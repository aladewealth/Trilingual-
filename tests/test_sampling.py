import torch

from trilingua.config import ModelConfig
from trilingua.model import Transformer
from trilingua.sampling import apply_repetition_penalty, filter_top_k_top_p, generate


def test_top_k():
    l = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    out = filter_top_k_top_p(l.clone(), top_k=2)
    assert torch.isfinite(out).sum() == 2 and torch.isfinite(out[1]) and torch.isfinite(out[3])


def test_top_p_keeps_minimal_nucleus():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    out = filter_top_k_top_p(probs.log(), top_p=0.75)
    assert torch.isfinite(out).tolist() == [True, True, False, False]
    out = filter_top_k_top_p(probs.log(), top_p=0.81)
    assert torch.isfinite(out).tolist() == [True, True, True, False]


def test_repetition_penalty():
    l = torch.tensor([2.0, -2.0, 1.0])
    out = apply_repetition_penalty(l.clone(), torch.tensor([0, 1, 1]), 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0]


def test_generate_greedy_deterministic_and_long():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=50, max_seq_len=16, d_model=32, n_layer=2, n_head=2))
    a = generate(m, [1, 2, 3], max_new_tokens=40, temperature=0)   # exceeds max_seq_len -> re-prefill
    b = generate(m, [1, 2, 3], max_new_tokens=40, temperature=0)
    assert a == b and len(a) == 40
    g = torch.Generator().manual_seed(0)
    s = generate(m, [1], max_new_tokens=10, temperature=1.0, top_k=5, generator=g)
    assert len(s) == 10 and all(0 <= t < 50 for t in s)


def test_stop_token():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=50, max_seq_len=32, d_model=32, n_layer=1, n_head=2))
    first = generate(m, [1, 2], max_new_tokens=1, temperature=0)[0]
    out = generate(m, [1, 2], max_new_tokens=10, temperature=0, stop_ids=[first])
    assert out == [first]
