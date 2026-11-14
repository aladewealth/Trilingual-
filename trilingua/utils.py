"""Small shared helpers: LR schedule, CSV logging, checkpoint I/O, model loading."""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from typing import Optional

import torch

from .config import ModelConfig
from .model import Transformer


def cosine_lr(step: int, max_steps: int, lr: float, warmup: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return lr * min_lr_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class CSVLogger:
    def __init__(self, path: str, fields):
        self.path, self.fields = path, list(fields)
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def log(self, row: dict):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def find_latest_checkpoint(search_dirs, name: str = "ckpt_latest.pt") -> Optional[str]:
    """Searches dirs (recursively) for checkpoints and returns the one with the highest step."""
    best, best_step = None, -1
    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        cands = [d] if os.path.isfile(d) else glob.glob(os.path.join(d, "**", name), recursive=True)
        for c in cands:
            side = c[:-3] + ".json"
            try:
                if os.path.exists(side):
                    with open(side) as f:
                        step = json.load(f)["step"]
                else:
                    step = torch.load(c, map_location="cpu", weights_only=False, mmap=True)["step"]
            except Exception:
                continue
            if step > best_step:
                best, best_step = c, step
    return best


def strip_compile_prefix(sd: dict) -> dict:
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}


def load_model(path: str, device: str = "cpu") -> Transformer:
    """Loads from a training checkpoint (.pt) or an exported dir (config.json + model.safetensors)."""
    if os.path.isdir(path):
        cfg = ModelConfig.load(os.path.join(path, "config.json"))
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, "model.safetensors"))
        model = Transformer(cfg)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert set(missing) <= {"lm_head.weight"}, missing
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict(ck["model_config"])
        model = Transformer(cfg)
        model.load_state_dict(strip_compile_prefix(ck["model"]))
    return model.to(device).eval()


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
