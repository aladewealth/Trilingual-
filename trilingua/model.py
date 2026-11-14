"""Decoder-only transformer written from scratch.

Pre-norm blocks, RMSNorm, rotary position embeddings (RoPE), SwiGLU MLP,
optional grouped-query attention, PyTorch SDPA causal attention, tied
input/output embeddings and a simple KV cache for fast generation.
"""
from __future__ import annotations

import inspect
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def precompute_rope(head_dim: int, max_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd). cos/sin: (T, hd/2). Rotate-half convention, computed in fp32."""
    dtype = x.dtype
    x = x.float()
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(dtype)


class KVCache:
    """Per-layer key/value cache. Tensors are (B, n_kv_head, T, hd)."""

    def __init__(self, n_layer: int):
        self.k: List[Optional[torch.Tensor]] = [None] * n_layer
        self.v: List[Optional[torch.Tensor]] = [None] * n_layer

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv_head, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None, layer_idx: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache.k[layer_idx] is not None:
                k = torch.cat([cache.k[layer_idx], k], dim=2)
                v = torch.cat([cache.v[layer_idx], v], dim=2)
            cache.k[layer_idx], cache.v[layer_idx] = k, v
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        Tq, Tk = q.shape[2], k.shape[2]
        drop = self.dropout if self.training else 0.0
        if Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:  # chunked prefill on top of an existing cache
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.resid_drop(self.wo(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)  # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual output projections
        for name, p in self.named_parameters():
            if name.endswith("attn.wo.weight") or name.endswith("mlp.w2.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None, start_pos: int = 0,
                last_only: bool = False, reduction: str = "mean"):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, f"sequence {start_pos + T} > max_seq_len"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i)
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-100, reduction=reduction)
            if reduction == "none":
                loss = loss.view(B, T)
            return logits, loss
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x), None

    def configure_optimizer(self, weight_decay: float, lr: float, betas=(0.9, 0.95),
                            device_type: str = "cpu", eps: float = 1e-8):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in params.items() if p.dim() >= 2]
        no_decay = [p for n, p in params.items() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        extra = dict(fused=True) if fused_ok else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, **extra)
