"""Export a checkpoint to safetensors (+ config + tokenizer + model card) and optionally push to the HF Hub.

  python -m trilingua.export --ckpt out/ckpt_latest.pt --tokenizer data/tokenizer.json --out_dir export
  HF_TOKEN=... python -m trilingua.export ... --push aladewealth/trilingua
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from .config import ModelConfig, param_counts
from .utils import strip_compile_prefix

CARD = """---
license: mit
language: [en, es, fr]
tags: [trilingual, from-scratch, pytorch, text-generation]
datasets: [HuggingFaceFW/fineweb-edu, epfml/FineWeb2-HQ{extra_ds}]
---
# Trilingua ({kind})

A {params:.1f}M-parameter decoder-only language model for English, Spanish and French,
trained from scratch (tokenizer included).
Architecture: pre-norm transformer, RMSNorm, RoPE, SwiGLU, tied embeddings, {n_layer} layers,
d_model {d_model}, {n_head} heads, context {ctx}.

This is NOT a `transformers` model; load it with the code at the project repository:

```python
from trilingua.utils import load_model
from trilingua.tokenizer import TrilinguaTokenizer
model = load_model("path/to/this/folder")
tok = TrilinguaTokenizer.from_file("path/to/this/folder/tokenizer.json")
```

Prompt format: start with a language tag `<|en|>`, `<|es|>` or `<|fr|>`.
{chat}
Training step: {step}. See the repository README for data, metrics and limitations.
Small model: expect fluent-looking but often factually wrong text.
"""


def export(ckpt_path: str, tokenizer: str, out_dir: str) -> str:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["model_config"])
    sd = strip_compile_prefix(ck["model"])
    if cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)   # shared with tok_emb.weight; re-tied on load
    sd = {k: v.detach().float().contiguous() for k, v in sd.items()}
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt", "step": str(ck.get("step", 0))})
    cfg.save(os.path.join(out_dir, "config.json"))
    shutil.copy(tokenizer, os.path.join(out_dir, "tokenizer.json"))
    sft = bool(ck.get("sft"))
    json.dump({"sft": sft, "step": ck.get("step"), "eos_token": "<|endoftext|>", "end_token": "<|end|>",
               "temperature": 0.8, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.1},
              open(os.path.join(out_dir, "generation_config.json"), "w"), indent=2)
    chat = ("Instruction format (SFT): `<|lang|><|user|>your request<|end|><|assistant|>`; "
            "generation stops at `<|end|>`.\n") if sft else ""
    open(os.path.join(out_dir, "README.md"), "w").write(CARD.format(
        kind="SFT writing assistant" if sft else "base", params=param_counts(cfg)["total"] / 1e6,
        n_layer=cfg.n_layer, d_model=cfg.d_model, n_head=cfg.n_head, ctx=cfg.max_seq_len,
        chat=chat, step=ck.get("step"), extra_ds=", CohereLabs/aya_dataset" if sft else ""))
    return out_dir


def push(out_dir: str, repo_id: str, token: str, private: bool = False, path_in_repo: str = "."):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, path_in_repo=path_in_repo,
                      commit_message="Upload Trilingua weights")
    print(f"pushed {out_dir} -> https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out_dir", default="export")
    p.add_argument("--push", default=None, help="HF repo id, e.g. aladewealth/trilingua")
    p.add_argument("--path_in_repo", default=".", help="e.g. '.' for base or 'sft' for the SFT model")
    p.add_argument("--private", action="store_true")
    a = p.parse_args(argv)
    export(a.ckpt, a.tokenizer, a.out_dir)
    print(f"exported to {a.out_dir}: {os.listdir(a.out_dir)}")
    if a.push:
        token = os.environ.get("HF_TOKEN")
        assert token, "set HF_TOKEN"
        push(a.out_dir, a.push, token, a.private, a.path_in_repo)


if __name__ == "__main__":
    main()
