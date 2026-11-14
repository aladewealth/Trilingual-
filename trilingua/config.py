"""Model configuration, presets, parameter and FLOP accounting."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 1024
    d_model: int = 640
    n_layer: int = 10
    n_head: int = 10
    n_kv_head: Optional[int] = None      # None -> n_head (plain MHA). Smaller -> GQA.
    ffn_hidden: Optional[int] = None     # None -> 8/3 * d_model rounded up to multiple_of
    multiple_of: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert (self.d_model // self.n_head) % 2 == 0, "head_dim must be even for RoPE"
        if self.ffn_hidden is None:
            h = int(8 * self.d_model / 3)
            self.ffn_hidden = self.multiple_of * math.ceil(h / self.multiple_of)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


PRESETS = {
    # ~1.4M params, for CPU smoke tests (uses a 4k smoke tokenizer)
    "tiny": dict(vocab_size=4096, max_seq_len=128, d_model=128, n_layer=4, n_head=4),
    # candidate configs for the real run (32k vocab, 1024 ctx)
    "small": dict(d_model=512, n_layer=8, n_head=8),
    "small12": dict(d_model=512, n_layer=12, n_head=8),
    "base": dict(d_model=640, n_layer=10, n_head=10),     # default
    "base12": dict(d_model=640, n_layer=12, n_head=10),
    "wide": dict(d_model=768, n_layer=8, n_head=12),
}


def get_config(name_or_path: str, **overrides) -> ModelConfig:
    if name_or_path in PRESETS:
        d = dict(PRESETS[name_or_path])
    else:
        with open(name_or_path) as f:
            d = json.load(f)
    d.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig.from_dict(d)


# ---- accounting ---------------------------------------------------------------

def param_counts(cfg: ModelConfig) -> dict:
    d, L, V, h = cfg.d_model, cfg.n_layer, cfg.vocab_size, cfg.ffn_hidden
    kv = cfg.n_kv_head * cfg.head_dim
    attn = d * d * 2 + d * kv * 2              # wq, wo, wk, wv
    mlp = 3 * d * h                            # w1, w2, w3
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = V * d
    head = 0 if cfg.tie_embeddings else V * d
    total = emb + head + L * per_layer + d     # + final norm
    return dict(embedding=emb, per_layer=per_layer, non_embedding=total - emb - head,
                total=total)


def flops_per_token(cfg: ModelConfig, seq_len: Optional[int] = None) -> float:
    """Training FLOPs/token (fwd+bwd) = 6 * matmul params + attention term.

    Matmul params include the LM head (it is a real matmul even when tied), and
    exclude the embedding lookup and norms. Attention: 12 * L * d * T (Kaplan/PaLM
    style, counts QK^T and AV for fwd+bwd, non-causal upper bound).
    """
    T = seq_len or cfg.max_seq_len
    pc = param_counts(cfg)
    matmul = pc["non_embedding"] - cfg.n_layer * 2 * cfg.d_model - cfg.d_model + cfg.vocab_size * cfg.d_model
    return 6 * matmul + 12 * cfg.n_layer * cfg.d_model * T
