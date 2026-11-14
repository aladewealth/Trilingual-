"""Data pipeline: stream EN/ES/FR web text from the Hugging Face Hub, train the
shared tokenizer, measure fertility, and write uint16 token shards.

Subcommands (run in order; each is restartable):
  sample     write a balanced raw-text sample per language for tokenizer training
  tokenizer  train the byte-level BPE on the sample
  shards     tokenize the streams into train/val shards (<|lang|> doc <|endoftext|>)
  fertility  tokens/word of our tokenizer vs GPT-2 on held-out docs (+ PNG chart)

Train/val split is deterministic: document i of a language's stream is a val
candidate iff i % val_every == 0. Val candidates are used for val until the
val quota is met and are otherwise dropped, so they never enter train shards or
the tokenizer sample.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Iterator, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Verified against the HF Hub API (configs + licenses) on 2026-11-14:
#   HuggingFaceFW/fineweb-edu  config sample-10BT (split train)         license odc-by
#   epfml/FineWeb2-HQ          configs spa_Latn / fra_Latn (split train) license odc-by
#   HuggingFaceFW/fineweb-2    configs spa_Latn / fra_Latn (train/test)  license odc-by
SOURCES = {
    "hq": {
        "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        "es": ("epfml/FineWeb2-HQ", "spa_Latn"),
        "fr": ("epfml/FineWeb2-HQ", "fra_Latn"),
    },
    "fineweb2": {
        "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        "es": ("HuggingFaceFW/fineweb-2", "spa_Latn"),
        "fr": ("HuggingFaceFW/fineweb-2", "fra_Latn"),
    },
}


def stream_docs(lang: str, source: str = "hq", start: int = 0, min_chars: int = 200,
                max_retries: int = 20) -> Iterator[Tuple[int, str]]:
    """Yields (doc_index, text). Re-opens the stream and skips ahead on network errors."""
    from datasets import load_dataset
    name, cfg = SOURCES[source][lang]
    i, retries = start, 0
    while True:
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True, columns=["text"])
            if i:
                ds = ds.skip(i)
            for ex in ds:
                idx = i
                i += 1
                t = ex["text"]
                if t and len(t) >= min_chars:
                    yield idx, t
            return
        except Exception as e:  # network hiccup during a long stream
            retries += 1
            if retries > max_retries:
                raise
            print(f"[{lang}] stream error at doc {i}: {e!r}; retry {retries}/{max_retries}", flush=True)
            time.sleep(min(60, 5 * retries))


def is_val_candidate(i: int, val_every: int) -> bool:
    return i % val_every == 0


# ---------------------------------------------------------------------------------
def cmd_sample(a):
    os.makedirs(a.out_dir, exist_ok=True)
    for lang in a.langs:
        path = os.path.join(a.out_dir, f"tok_sample_{lang}.jsonl")
        if os.path.exists(path + ".done"):
            print(f"[{lang}] sample exists, skipping"); continue
        target = int(a.mb_per_lang * 1e6)
        n_bytes, n_docs, t0 = 0, 0, time.time()
        with open(path, "w") as f:
            for i, text in stream_docs(lang, a.source):
                if is_val_candidate(i, a.val_every):
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n_bytes += len(text.encode("utf-8")); n_docs += 1
                if n_bytes >= target:
                    break
        open(path + ".done", "w").write(str(n_bytes))
        print(f"[{lang}] sample: {n_docs} docs, {n_bytes/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)


def _iter_sample(out_dir, langs):
    for lang in langs:
        with open(os.path.join(out_dir, f"tok_sample_{lang}.jsonl")) as f:
            for line in f:
                yield json.loads(line)["text"]


def cmd_tokenizer(a):
    from .tokenizer import train_tokenizer
    t0 = time.time()
    n = sum(sum(1 for _ in open(os.path.join(a.out_dir, f"tok_sample_{l}.jsonl"))) for l in a.langs)
    tok = train_tokenizer(_iter_sample(a.out_dir, a.langs), vocab_size=a.vocab_size,
                          min_frequency=a.min_frequency, length=n)
    path = a.tokenizer or os.path.join(a.out_dir, "tokenizer.json")
    tok.save(path)
    print(f"tokenizer: vocab {tok.get_vocab_size()} trained on {n} docs in {time.time()-t0:.0f}s -> {path}")


def cmd_shards(a):
    from .data import ShardWriter
    from .tokenizer import TrilinguaTokenizer
    tok = TrilinguaTokenizer.from_file(a.tokenizer or os.path.join(a.out_dir, "tokenizer.json"))
    assert tok.vocab_size <= 65536
    mix = dict(kv.split("=") for kv in a.mix.split(","))
    mix = {k: float(v) for k, v in mix.items()}
    s = sum(mix.values())
    for lang in a.langs:
        meta_path = os.path.join(a.out_dir, f"meta_{lang}.json")
        if os.path.exists(meta_path):
            print(f"[{lang}] shards exist, skipping"); continue
        quota = int(a.total_tokens * mix[lang] / s)
        val_quota = int(a.val_tokens)
        tw = ShardWriter(a.out_dir, "train", lang, a.shard_tokens)
        vw = ShardWriter(a.out_dir, "val", lang, a.shard_tokens)
        val_docs_f = open(os.path.join(a.out_dir, f"val_docs_{lang}.jsonl"), "w")
        n_val_docs_saved, t0, last = 0, time.time(), 0
        batch, n_docs = [], 0

        def flush(batch):
            nonlocal n_val_docs_saved
            ids = tok.encode_batch([t for _, t in batch])
            for (is_val, text), d in zip(batch, ids):
                doc = [tok.lang_ids[lang]] + d + [tok.eot_id]
                if is_val:
                    if vw.total < val_quota:
                        vw.add(doc)
                        if n_val_docs_saved < a.save_val_docs:
                            val_docs_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                            n_val_docs_saved += 1
                elif tw.total < quota:
                    tw.add(doc)

        for i, text in stream_docs(lang, a.source):
            v = is_val_candidate(i, a.val_every)
            if v and vw.total >= val_quota:
                continue
            batch.append((v, text)); n_docs += 1
            if len(batch) >= a.batch_docs:
                flush(batch); batch = []
                if tw.total - last >= a.log_every:
                    last = tw.total
                    el = time.time() - t0
                    print(f"[{lang}] {tw.total/1e6:.1f}M/{quota/1e6:.0f}M train tok, {vw.total/1e6:.2f}M val, "
                          f"{tw.total/el/1e3:.0f}k tok/s", flush=True)
            if tw.total >= quota and vw.total >= val_quota:
                break
        if batch:
            flush(batch)
        tw.close(); vw.close(); val_docs_f.close()
        meta = dict(lang=lang, source=SOURCES[a.source][lang], train_tokens=tw.total, val_tokens=vw.total,
                    train_shards=tw.idx, val_shards=vw.idx, docs_seen=n_docs, val_every=a.val_every,
                    seconds=round(time.time() - t0, 1))
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"[{lang}] done: {meta}", flush=True)
    # combine
    metas = {}
    for lang in ("en", "es", "fr"):
        p = os.path.join(a.out_dir, f"meta_{lang}.json")
        if os.path.exists(p):
            metas[lang] = json.load(open(p))
    combined = dict(langs=list(metas), per_lang=metas, vocab_size=tok.vocab_size,
                    total_train_tokens=sum(m["train_tokens"] for m in metas.values()),
                    format="uint16 shards, 256xint32 header; docs = <|lang|> text <|endoftext|>")
    json.dump(combined, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(combined, indent=2))


_WORDS = re.compile(r"\S+")


def cmd_fertility(a):
    from tokenizers import Tokenizer
    from .tokenizer import TrilinguaTokenizer
    tok = TrilinguaTokenizer.from_file(a.tokenizer or os.path.join(a.out_dir, "tokenizer.json"))
    others = {}
    for name in a.compare:
        try:
            others[name] = Tokenizer.from_pretrained(name)
        except Exception as e:
            print(f"could not load {name}: {e}")
    res = {}
    for lang in a.langs:
        texts = [json.loads(l)["text"] for l in open(os.path.join(a.data_dir or a.out_dir, f"val_docs_{lang}.jsonl"))]
        texts = texts[: a.max_docs]
        words = sum(len(_WORDS.findall(t)) for t in texts)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        row = {"docs": len(texts), "words": words, "bytes": nbytes}
        ours = sum(len(x) for x in tok.encode_batch(texts))
        row["trilingua"] = {"tokens": ours, "tokens_per_word": ours / words, "bytes_per_token": nbytes / ours}
        for name, t in others.items():
            n = sum(len(e.ids) for e in t.encode_batch(texts, add_special_tokens=False))
            row[name] = {"tokens": n, "tokens_per_word": n / words, "bytes_per_token": nbytes / n}
        res[lang] = row
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
    json.dump(res, open(a.json_out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    if a.png_out:
        from .plots import plot_fertility
        plot_fertility(res, a.png_out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["sample", "tokenizer", "shards", "fertility"])
    p.add_argument("--out_dir", default="data")
    p.add_argument("--data_dir", default=None, help="dir with val_docs_*.jsonl (fertility)")
    p.add_argument("--langs", nargs="+", default=["en", "es", "fr"])
    p.add_argument("--source", default="hq", choices=list(SOURCES))
    p.add_argument("--val_every", type=int, default=100)
    # sample / tokenizer
    p.add_argument("--mb_per_lang", type=float, default=150)
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--tokenizer", default=None, help="tokenizer.json path (default out_dir/tokenizer.json)")
    # shards
    p.add_argument("--total_tokens", type=float, default=1.3e9)
    p.add_argument("--mix", default="en=0.5,es=0.25,fr=0.25")
    p.add_argument("--val_tokens", type=float, default=2e6, help="per language")
    p.add_argument("--shard_tokens", type=int, default=100_000_000)
    p.add_argument("--batch_docs", type=int, default=1024)
    p.add_argument("--save_val_docs", type=int, default=500)
    p.add_argument("--log_every", type=int, default=20_000_000)
    # fertility
    p.add_argument("--compare", nargs="*", default=["gpt2"])
    p.add_argument("--max_docs", type=int, default=500)
    p.add_argument("--json_out", default="results/fertility.json")
    p.add_argument("--png_out", default="results/fertility.png")
    a = p.parse_args(argv)
    {"sample": cmd_sample, "tokenizer": cmd_tokenizer, "shards": cmd_shards, "fertility": cmd_fertility}[a.cmd](a)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # avoid a known pyarrow/streaming teardown crash at interpreter exit
