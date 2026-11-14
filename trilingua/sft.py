"""Stage 2 (optional): small supervised fine-tune into a trilingual writing assistant.

Data (built by `python -m trilingua.sft build`):
  * CohereLabs/aya_dataset (Apache-2.0), human-written prompt/response pairs,
    filtered to language_code in {eng, spa, fra}.
  * Self-supervised "continue this text" tasks made from pretraining-source
    documents (ODC-By), with instructions in the document's language. These are
    documents the model may have seen in pretraining; they teach the chat format
    and the continuation mode, they are not used for evaluation.
Format: <|lang|><|user|>{prompt}<|end|><|assistant|>{reply}<|end|><|endoftext|>,
loss only on the reply.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext

import torch

from .model import Transformer
from .config import ModelConfig
from .tokenizer import TrilinguaTokenizer
from .utils import CSVLogger, atomic_save, cosine_lr, strip_compile_prefix

AYA_CODES = {"eng": "en", "spa": "es", "fra": "fr"}
CONT_INSTR = {
    "en": ["Continue this text:", "Continue the following passage:", "Keep writing from here:"],
    "es": ["Continúa este texto:", "Continúa el siguiente pasaje:", "Sigue escribiendo a partir de aquí:"],
    "fr": ["Continue ce texte :", "Poursuis le passage suivant :", "Continue d'écrire à partir d'ici :"],
}


def build(a):
    from datasets import load_dataset
    from .prepare_data import stream_docs
    rng = random.Random(a.seed)
    rows = []
    aya = load_dataset("CohereLabs/aya_dataset", split="train")
    for e in aya:
        l = AYA_CODES.get(e["language_code"])
        if l and e["inputs"].strip() and e["targets"].strip():
            rows.append({"lang": l, "prompt": e["inputs"].strip(), "reply": e["targets"].strip(), "src": "aya"})
    n_aya = len(rows)
    for lang in a.langs:
        n = 0
        for i, text in stream_docs(lang, a.source):
            if i % 100 != 1:          # a fixed slice of train-side documents
                continue
            words = text.split(" ")
            if len(words) < 120:
                continue
            cut = rng.randint(40, 80)
            prefix = " ".join(words[:cut])
            cont = " ".join(words[cut:cut + rng.randint(80, 160)])
            rows.append({"lang": lang, "prompt": f"{rng.choice(CONT_INSTR[lang])}\n\n{prefix}",
                         "reply": cont, "src": "continuation"})
            n += 1
            if n >= a.cont_per_lang:
                break
    rng.shuffle(rows)
    n_val = max(50, int(len(rows) * a.val_frac))
    os.makedirs(a.out_dir, exist_ok=True)
    for name, part in (("sft_val.jsonl", rows[:n_val]), ("sft_train.jsonl", rows[n_val:])):
        with open(os.path.join(a.out_dir, name), "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {}
    for r in rows:
        k = f"{r['src']}_{r['lang']}"
        stats[k] = stats.get(k, 0) + 1
    json.dump(dict(total=len(rows), aya=n_aya, val=n_val, by_source_lang=stats),
              open(os.path.join(a.out_dir, "sft_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


def _load_examples(path, tok: TrilinguaTokenizer, max_len: int):
    out = []
    for line in open(path):
        r = json.loads(line)
        ids, labels = tok.chat_example(r["lang"], r["prompt"], r["reply"])
        if len(ids) > max_len:
            p = len(ids) - sum(1 for x in labels if x != -100)
            if p > max_len * 3 // 4:      # prompt alone too long: drop
                continue
            ids, labels = ids[:max_len], labels[:max_len]
        out.append((ids, labels))
    return out


def _collate(batch, pad_id, device):
    L = max(len(i) for i, _ in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for r, (ids, labels) in enumerate(batch):
        x[r, :len(ids)] = torch.tensor(ids)
        y[r, :len(ids) - 1] = torch.tensor(labels[1:])   # next-token targets
    return x.to(device), y.to(device)


def train(a):
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(a.seed)
    tok = TrilinguaTokenizer.from_file(a.tokenizer)
    ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    model.to(device)
    train_ex = _load_examples(os.path.join(a.data_dir, "sft_train.jsonl"), tok, cfg.max_seq_len)
    val_ex = _load_examples(os.path.join(a.data_dir, "sft_val.jsonl"), tok, cfg.max_seq_len)
    # sort-ish by length within buckets to reduce padding, then shuffle buckets each epoch
    dtype = a.dtype if a.dtype != "auto" else ("fp16" if dev_type == "cuda" and not torch.cuda.is_bf16_supported() else ("bf16" if dev_type == "cuda" else "fp32"))
    pt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    autocast = torch.autocast(dev_type, dtype=pt) if dtype != "fp32" else nullcontext()
    try:
        scaler = torch.amp.GradScaler(dev_type, enabled=dtype == "fp16")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=dtype == "fp16")
    opt = model.configure_optimizer(a.weight_decay, a.lr, (0.9, 0.95), dev_type)
    steps_per_epoch = math.ceil(len(train_ex) / (a.micro_batch * a.grad_accum))
    max_steps = a.max_steps or steps_per_epoch * a.epochs
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(a.out_dir, "sft_log.csv")):
        os.remove(os.path.join(a.out_dir, "sft_log.csv"))   # SFT is a single short run: start a fresh log
    log = CSVLogger(os.path.join(a.out_dir, "sft_log.csv"), ["step", "epoch", "loss", "val_loss", "lr"])
    print(f"SFT: {len(train_ex)} train / {len(val_ex)} val examples, {max_steps} steps, dtype {dtype}", flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(val_ex), a.micro_batch):
            x, y = _collate(val_ex[i:i + a.micro_batch], tok.pad_id, device)
            with autocast:
                _, loss = model(x, y, reduction="none")
            m = y != -100
            tot += loss[m].double().sum().item(); n += m.sum().item()
        model.train()
        return tot / max(1, n)

    v0 = evaluate()
    log.log(dict(step=0, epoch=0, loss="", val_loss=round(v0, 4), lr=0))
    print(f"step 0 val {v0:.4f}", flush=True)
    rng = random.Random(a.seed)
    order, pos, epoch, step = [], 0, 0, 0
    model.train()
    t0 = time.time()
    while step < max_steps:
        lr = cosine_lr(step, max_steps, a.lr, a.warmup_steps, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(a.grad_accum):
            if pos + a.micro_batch > len(order):
                # length-bucketed shuffle: sort chunks of 50*mb by length, then shuffle the batches
                order = list(range(len(train_ex))); rng.shuffle(order)
                chunk = a.micro_batch * 50
                order = [j for c in range(0, len(order), chunk)
                         for j in sorted(order[c:c + chunk], key=lambda k: len(train_ex[k][0]))]
                batches = [order[k:k + a.micro_batch] for k in range(0, len(order), a.micro_batch)]
                rng.shuffle(batches)
                order = [j for b in batches for j in b]
                pos = 0
                epoch += 1 if step else 0
            x, y = _collate([train_ex[j] for j in order[pos:pos + a.micro_batch]], tok.pad_id, device)
            pos += a.micro_batch
            # token-weighted mean across micro-batches would need a global count; per-micro mean is fine here
            with autocast:
                _, loss = model(x, y)
            scaler.scale(loss / a.grad_accum).backward()
            loss_acc += loss.item() / a.grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        vl = ""
        if step % a.eval_interval == 0 or step == max_steps:
            vl = round(evaluate(), 4)
        if step % a.log_interval == 0 or vl != "":
            log.log(dict(step=step, epoch=epoch, loss=round(loss_acc, 4), val_loss=vl, lr=f"{lr:.2e}"))
            print(f"step {step}/{max_steps} loss {loss_acc:.4f} val {vl} lr {lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
    atomic_save(dict(model=model.state_dict(), model_config=cfg.to_dict(), step=step, sft=True,
                     base_ckpt=os.path.abspath(a.init_ckpt)), os.path.join(a.out_dir, "sft_final.pt"))
    print("saved", os.path.join(a.out_dir, "sft_final.pt"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "train"])
    p.add_argument("--out_dir", default="sft_out")
    p.add_argument("--data_dir", default="sft_data")
    p.add_argument("--langs", nargs="+", default=["en", "es", "fr"])
    p.add_argument("--source", default="hq")
    p.add_argument("--cont_per_lang", type=int, default=1500)
    p.add_argument("--val_frac", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--init_ckpt", default="out/ckpt_latest.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    a = p.parse_args(argv)
    if a.cmd == "build":
        a.out_dir = a.data_dir if a.out_dir == "sft_out" else a.out_dir
        build(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
