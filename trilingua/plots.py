"""PNG charts for the write-up: loss curves, per-language perplexity / bpb, tokenizer fertility."""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"en": "#1f77b4", "es": "#d62728", "fr": "#2ca02c", "avg": "#444444"}


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_fertility(res: Dict, out: str, title: Optional[str] = None):
    langs = list(res)
    names = [k for k in res[langs[0]] if isinstance(res[langs[0]][k], dict)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        vals = [res[l][n]["tokens_per_word"] for l in langs]
        bars = ax.bar([i + j * w for i in range(len(langs))], vals, w, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks([i + w * (len(names) - 1) / 2 for i in range(len(langs))])
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_ylabel("tokens per word (lower is better)")
    ax.set_title(title or "Tokenizer fertility on held-out web text")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_losses(train_csv: str, val_csv: Optional[str], out: str, title: str = "Training loss"):
    tr = _read_csv(train_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [float(r["tokens"]) / 1e6 for r in tr]
    ax.plot(x, [float(r["loss"]) for r in tr], color="#999999", lw=1, label="train")
    if val_csv and os.path.exists(val_csv):
        va = _read_csv(val_csv)
        xv = [float(r["tokens"]) / 1e6 for r in va]
        for l in ("en", "es", "fr"):
            k = f"val_loss_{l}"
            if va and k in va[0] and va[0][k] != "":
                ax.plot(xv, [float(r[k]) for r in va], "o-", ms=3, color=COLORS[l], label=f"val {l}")
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_lang(metrics: Dict[str, Dict[str, float]], key: str, out: str, ylabel: str, title: str):
    """metrics: {model_name: {lang: value}}"""
    models = list(metrics)
    langs = [l for l in ("en", "es", "fr") if any(l in metrics[m] for m in models)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        vals = [metrics[m].get(l, float("nan")) for l in langs]
        bars = ax.bar([i + j * w for i in range(len(langs))], vals, w, label=m)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([i + w * (len(models) - 1) / 2 for i in range(len(langs))])
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
