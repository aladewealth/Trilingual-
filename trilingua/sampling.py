"""Autoregressive sampling with KV cache: temperature, top-k, top-p, repetition penalty, streaming."""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence

import torch

from .model import KVCache, Transformer


def apply_repetition_penalty(logits: torch.Tensor, prev: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style: divide positive logits / multiply negative logits of already-seen tokens."""
    if penalty == 1.0 or prev.numel() == 0:
        return logits
    uniq = torch.unique(prev)
    sel = logits[uniq]
    logits[uniq] = torch.where(sel > 0, sel / penalty, sel * penalty)
    return logits


def filter_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """logits: (V,). Returns logits with filtered entries set to -inf."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > top_p          # keep the smallest set with mass >= top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate_stream(model: Transformer, prompt_ids: Sequence[int], max_new_tokens: int = 200,
                    temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                    repetition_penalty: float = 1.1, penalty_window: int = 256,
                    stop_ids: Iterable[int] = (), no_penalty_ids: Iterable[int] = (),
                    generator: Optional[torch.Generator] = None) -> Iterator[int]:
    """Yields new token ids one by one. Uses a KV cache; re-prefills when the context is full."""
    model.eval()
    device = next(model.parameters()).device
    max_len = model.cfg.max_seq_len
    stop_ids = set(stop_ids)
    no_pen = torch.tensor(sorted(set(no_penalty_ids)), dtype=torch.long, device=device)
    tokens: List[int] = list(prompt_ids)[-(max_len - 1):]
    assert tokens, "empty prompt"

    def prefill(toks):
        cache = KVCache(model.cfg.n_layer)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        logits, _ = model(x, cache=cache, start_pos=0, last_only=True)
        return cache, logits[0, -1].float()

    cache, logits = prefill(tokens)
    for _ in range(max_new_tokens):
        recent = torch.tensor(tokens[-penalty_window:], dtype=torch.long, device=device)
        if no_pen.numel():
            recent = recent[~torch.isin(recent, no_pen)]
        logits = apply_repetition_penalty(logits.clone(), recent, repetition_penalty)
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            l = filter_top_k_top_p(logits / temperature, top_k, top_p)
            probs = torch.softmax(l, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        tokens.append(nxt)
        yield nxt
        if nxt in stop_ids:
            return
        if cache.seq_len >= max_len:      # context full: keep the last half and re-prefill
            tokens_ctx = tokens[-(max_len // 2):]
            cache, logits = prefill(tokens_ctx)
        else:
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            out, _ = model(x, cache=cache, start_pos=cache.seq_len)
            logits = out[0, -1].float()


def generate(model, prompt_ids, **kw) -> List[int]:
    return list(generate_stream(model, prompt_ids, **kw))
