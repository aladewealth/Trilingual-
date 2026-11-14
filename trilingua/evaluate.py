"""Evaluation: per-language val loss / perplexity / bits-per-byte, GPT-2 comparison
(bits-per-byte on identical raw text, the only tokenizer-independent comparison),
language-ID accuracy of generations, qualitative samples and PNG charts.

  python -m trilingua.evaluate --model out/ckpt_latest.pt --data_dir data --out_dir results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import torch

from .data import ValSet
from .langid import langid
from .sampling import generate
from .tokenizer import TrilinguaTokenizer
from .utils import load_model

PROMPTS = {
    "en": ["The history of the printing press", "Once upon a time, in a small village by the sea,",
           "Climate change affects farmers because"],
    "es": ["La historia de la imprenta", "Había una vez, en un pequeño pueblo junto al mar,",
           "El cambio climático afecta a los agricultores porque"],
    "fr": ["L'histoire de l'imprimerie", "Il était une fois, dans un petit village au bord de la mer,",
           "Le changement climatique touche les agriculteurs parce que"],
}
CHAT_PROMPTS = {
    "en": ["Write a short paragraph about the ocean.", "Continue this story: The door creaked open and"],
    "es": ["Escribe un párrafo corto sobre el océano.", "Continúa esta historia: La puerta se abrió con un chirrido y"],
    "fr": ["Écris un court paragraphe sur l'océan.", "Continue cette histoire : La porte s'ouvrit en grinçant et"],
}


@torch.no_grad()
def shard_metrics(model, tok, data_dir, langs, max_chunks, batch, device):
    tb = torch.tensor(tok.token_bytes(), device=device)
    res = {}
    for l in langs:
        vs = ValSet(data_dir, l, model.cfg.max_seq_len, max_chunks)
        s = [0.0, 0, 0]
        for x, y in vs.batches(batch, device=device):
            _, loss = model(x, y, reduction="none")
            s[0] += loss.double().sum().item(); s[1] += y.numel(); s[2] += tb[y].sum().item()
        res[l] = dict(val_loss=s[0] / s[1], ppl=math.exp(s[0] / s[1]), bpb=s[0] / math.log(2) / s[2],
                      tokens=s[1], windows=vs.n)
    return res


@torch.no_grad()
def doc_bpb_ours(model, tok, texts, lang, device):
    tb = tok.token_bytes()
    nats, nbytes = 0.0, 0
    L = model.cfg.max_seq_len
    for t in texts:
        ids = [tok.lang_ids[lang]] + tok.encode(t)
        ids = ids[: L + 1]
        x = torch.tensor([ids[:-1]], device=device); y = torch.tensor([ids[1:]], device=device)
        _, loss = model(x, y, reduction="none")
        nats += loss.double().sum().item()
        nbytes += int(sum(tb[i] for i in ids[1:]))
    return nats / math.log(2) / nbytes


@torch.no_grad()
def doc_bpb_hf(name, texts, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    htok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    L = getattr(m.config, "n_positions", 1024)
    nats, nbytes = 0.0, 0
    for t in texts:
        ids = [htok.bos_token_id or htok.eos_token_id] + htok.encode(t)
        ids = ids[: L]
        x = torch.tensor([ids], device=device)
        logits = m(x).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum")
        nats += loss.item()
        nbytes += len(htok.decode(ids[1:]).encode("utf-8"))
    n_params = sum(p.numel() for p in m.parameters())
    del m
    return nats / math.log(2) / nbytes, n_params


def langid_eval(model, tok, langs, n, max_new, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    confusion = {l: Counter() for l in langs}
    examples = defaultdict(list)
    backend = None
    for l in langs:
        for i in range(n):
            ids = generate(model, [tok.lang_ids[l]], max_new_tokens=max_new, temperature=0.8, top_k=50,
                           top_p=0.95, repetition_penalty=1.1, stop_ids=[tok.eot_id], generator=g)
            text = tok.decode(ids)
            pred, conf, backend = langid(text)
            confusion[l][pred] += 1
            if i < 3:
                examples[l].append({"pred": pred, "text": text[:300]})
    acc = {l: confusion[l][l] / max(1, sum(confusion[l].values())) for l in langs}
    return dict(accuracy=acc, mean_accuracy=sum(acc.values()) / len(acc),
                confusion={l: dict(c) for l, c in confusion.items()}, backend=backend, n_per_lang=n,
                examples=dict(examples))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="training ckpt .pt, sft_final.pt, or exported dir")
    p.add_argument("--tokenizer", default=None, help="default: <data_dir>/tokenizer.json")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--langs", nargs="+", default=["en", "es", "fr"])
    p.add_argument("--max_chunks", type=int, default=None, help="val windows per language (default all)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--compare", nargs="*", default=[], help="HF causal LMs to compare by bpb, e.g. gpt2")
    p.add_argument("--n_docs", type=int, default=200, help="raw val docs per language for bpb comparison")
    p.add_argument("--langid_n", type=int, default=30)
    p.add_argument("--max_new", type=int, default=80)
    p.add_argument("--chat", action="store_true", help="also run instruction prompts (SFT model)")
    p.add_argument("--train_dir", default=None, help="dir with train_log.csv/val_log.csv for loss curve")
    p.add_argument("--tag", default="trilingua")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    tok = TrilinguaTokenizer.from_file(a.tokenizer or os.path.join(a.data_dir, "tokenizer.json"))
    model = load_model(a.model, device)
    t0 = time.time()
    out = dict(model=a.model, params=model.num_params(), config=model.cfg.to_dict())
    print(f"params {out['params']/1e6:.2f}M", flush=True)

    out["shard_metrics"] = shard_metrics(model, tok, a.data_dir, a.langs, a.max_chunks, a.batch, device)
    print("val:", json.dumps(out["shard_metrics"], indent=1), flush=True)

    docs = {}
    for l in a.langs:
        pth = os.path.join(a.data_dir, f"val_docs_{l}.jsonl")
        if os.path.exists(pth):
            docs[l] = [json.loads(x)["text"] for x in open(pth)][: a.n_docs]
    bpb = {a.tag: {l: doc_bpb_ours(model, tok, docs[l], l, device) for l in docs}}
    for name in a.compare:
        try:
            bpb[name] = {}
            for l in docs:
                bpb[name][l], npar = doc_bpb_hf(name, docs[l], device)
            out.setdefault("compare_params", {})[name] = npar
        except Exception as e:
            print(f"compare {name} failed: {e}")
            bpb.pop(name, None)
    out["doc_bpb"] = bpb
    print("bpb on raw val docs:", json.dumps(bpb, indent=1), flush=True)

    # language-ID sanity check of the classifier on real val text
    cls_acc = {}
    for l, ts in docs.items():
        preds = [langid(t[:1000])[0] for t in ts[:100]]
        cls_acc[l] = sum(p == l for p in preds) / max(1, len(preds))
    out["langid_classifier_accuracy_on_real_text"] = cls_acc
    out["langid_generations"] = langid_eval(model, tok, a.langs, a.langid_n, a.max_new, device)
    print("langid:", out["langid_generations"]["accuracy"], "backend", out["langid_generations"]["backend"], flush=True)

    g = torch.Generator(device=device).manual_seed(1)
    md = [f"# Samples ({a.tag})\n", "Settings: temperature 0.8, top-k 50, top-p 0.95, repetition penalty 1.1\n"]
    for l in a.langs:
        md.append(f"\n## {l.upper()}\n")
        for pr in PROMPTS[l]:
            ids = [tok.lang_ids[l]] + tok.encode(pr)
            gen = generate(model, ids, max_new_tokens=a.max_new, temperature=0.8, top_k=50, top_p=0.95,
                           repetition_penalty=1.1, stop_ids=[tok.eot_id], generator=g)
            md.append(f"**{pr}**{tok.decode(gen)}\n")
        if a.chat:
            for pr in CHAT_PROMPTS[l]:
                gen = generate(model, tok.chat_prompt_ids(l, pr), max_new_tokens=a.max_new * 2, temperature=0.7,
                               top_k=50, top_p=0.9, repetition_penalty=1.1, stop_ids=[tok.end_id, tok.eot_id],
                               generator=g)
                md.append(f"> **User:** {pr}\n>\n> **Trilingua:** {tok.decode(gen)}\n")
    open(os.path.join(a.out_dir, f"samples_{a.tag}.md"), "w").write("\n".join(md))

    out["eval_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out_dir, f"eval_{a.tag}.json"), "w"), indent=2, ensure_ascii=False)

    from .plots import plot_losses, plot_per_lang
    plot_per_lang({a.tag: {l: v["ppl"] for l, v in out["shard_metrics"].items()}}, "ppl",
                  os.path.join(a.out_dir, f"ppl_per_lang_{a.tag}.png"), "perplexity (val shards)",
                  "Per-language validation perplexity")
    plot_per_lang(bpb, "bpb", os.path.join(a.out_dir, f"bpb_{a.tag}.png"), "bits per byte (lower is better)",
                  "Bits per byte on identical held-out text")
    if a.train_dir and os.path.exists(os.path.join(a.train_dir, "train_log.csv")):
        plot_losses(os.path.join(a.train_dir, "train_log.csv"), os.path.join(a.train_dir, "val_log.csv"),
                    os.path.join(a.out_dir, f"loss_{a.tag}.png"))
    print(f"wrote results to {a.out_dir} in {out['eval_seconds']}s")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
