import torch

from trilingua.model import apply_rope, precompute_rope


def test_rope_preserves_norm():
    cos, sin = precompute_rope(16, 64)
    x = torch.randn(2, 3, 64, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    cos, sin = precompute_rope(16, 8)
    x = torch.randn(1, 1, 1, 16)
    assert torch.allclose(apply_rope(x, cos[:1], sin[:1]), x)


def test_rope_relative_position_property():
    """<R_m q, R_n k> depends only on m - n."""
    hd, L = 32, 128
    cos, sin = precompute_rope(hd, L)
    q = torch.randn(hd)
    k = torch.randn(hd)

    def rot(v, pos):
        return apply_rope(v.view(1, 1, 1, hd), cos[pos:pos + 1], sin[pos:pos + 1]).view(hd)

    d1 = torch.dot(rot(q, 10), rot(k, 3))
    d2 = torch.dot(rot(q, 57), rot(k, 50))
    d3 = torch.dot(rot(q, 100), rot(k, 93))
    assert torch.allclose(d1, d2, atol=1e-4) and torch.allclose(d1, d3, atol=1e-4)
    d4 = torch.dot(rot(q, 10), rot(k, 4))
    assert not torch.allclose(d1, d4, atol=1e-3)


def test_rope_keeps_dtype():
    cos, sin = precompute_rope(8, 4)
    x = torch.randn(1, 1, 4, 8).half()
    assert apply_rope(x, cos, sin).dtype == torch.float16
