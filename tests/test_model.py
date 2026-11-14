import torch

from trilingua.config import ModelConfig, param_counts, get_config, PRESETS
from trilingua.model import KVCache, Transformer


def tiny(**kw):
    d = dict(vocab_size=97, max_seq_len=64, d_model=32, n_layer=2, n_head=4)
    d.update(kw)
    return ModelConfig(**d)


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    m = Transformer(tiny())
    x = torch.randint(0, 97, (3, 20))
    tgt = torch.randint(0, 97, (3, 20))  # independent targets (with tied embeddings, target==input is easy)
    logits, loss = m(x, tgt)
    assert logits.shape == (3, 20, 97)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # at init the loss should be close to ln(V)
    assert abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5
    logits, none = m(x)
    assert none is None and logits.shape == (3, 20, 97)
    last, _ = m(x, last_only=True)
    assert last.shape == (3, 1, 97)
    assert torch.allclose(last[:, 0], logits[:, -1], atol=1e-5)
    _, per_tok = m(x, x, reduction="none")
    assert per_tok.shape == (3, 20)


def test_param_count_matches_analytic():
    for name in ["tiny", "small", "base", "base12", "wide"]:
        cfg = get_config(name)
        if name != "tiny":
            cfg = get_config(name, max_seq_len=128)
        m = Transformer(cfg)
        assert m.num_params() == param_counts(cfg)["total"], name


def test_tied_embeddings():
    m = Transformer(tiny())
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    m2 = Transformer(tiny(tie_embeddings=False))
    assert m2.lm_head.weight.data_ptr() != m2.tok_emb.weight.data_ptr()
    assert m2.num_params() == param_counts(m2.cfg)["total"]


def test_causal_masking():
    """Changing future tokens must not change logits at earlier positions."""
    torch.manual_seed(0)
    m = Transformer(tiny()).eval()
    x = torch.randint(0, 97, (2, 32))
    y = x.clone()
    t = 17
    y[:, t:] = torch.randint(0, 97, (2, 32 - t))
    with torch.no_grad():
        lx, _ = m(x)
        ly, _ = m(y)
    assert torch.allclose(lx[:, :t], ly[:, :t], atol=1e-5)
    assert not torch.allclose(lx[:, t:], ly[:, t:], atol=1e-3)


def test_gradient_does_not_flow_from_future():
    torch.manual_seed(0)
    m = Transformer(tiny()).eval()
    x = torch.randint(0, 97, (1, 16))
    emb = m.tok_emb(x).detach().requires_grad_(True)
    # run blocks manually from embeddings
    h = emb
    cos, sin = m.rope_cos[:16], m.rope_sin[:16]
    for i, layer in enumerate(m.layers):
        h = layer(h, cos, sin)
    out = m.lm_head(m.norm(h))
    out[0, 5].sum().backward()
    g = emb.grad[0].abs().sum(-1)
    assert torch.all(g[6:] == 0) and torch.all(g[:6] > 0)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    for kv in (4, 2, 1):  # MHA, GQA, MQA
        m = Transformer(tiny(n_kv_head=kv)).eval()
        x = torch.randint(0, 97, (1, 24))
        with torch.no_grad():
            full, _ = m(x)
            cache = KVCache(m.cfg.n_layer)
            out, _ = m(x[:, :10], cache=cache, start_pos=0)
            steps = [out]
            for t in range(10, 24):
                o, _ = m(x[:, t:t + 1], cache=cache, start_pos=t)
                steps.append(o)
            inc = torch.cat(steps, dim=1)
            # chunked prefill on top of a cache (Tq > 1, Tq != Tk)
            cache2 = KVCache(m.cfg.n_layer)
            a, _ = m(x[:, :8], cache=cache2, start_pos=0)
            b, _ = m(x[:, 8:24], cache=cache2, start_pos=8)
        assert torch.allclose(full, inc, atol=1e-4), kv
        assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-4), kv


def test_ffn_hidden_rounding():
    cfg = ModelConfig(d_model=640, n_layer=1, n_head=10)
    assert cfg.ffn_hidden % 64 == 0 and cfg.ffn_hidden >= int(8 * 640 / 3)


def test_backward_runs_and_init_scaling():
    torch.manual_seed(0)
    m = Transformer(tiny(n_layer=4))
    x = torch.randint(0, 97, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    std_wo = m.layers[0].attn.wo.weight.std().item()
    std_wq = m.layers[0].attn.wq.weight.std().item()
    assert std_wo < std_wq  # residual projections use the scaled init
