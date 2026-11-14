import numpy as np
import pytest
import torch

from trilingua.data import MixtureLoader, ShardWriter, ValSet, list_shards, read_shard, write_shard


def test_shard_roundtrip(tmp_path):
    arr = np.arange(65000, dtype=np.int64) % 65535
    p = str(tmp_path / "train_en_0000.bin")
    write_shard(p, arr)
    back = read_shard(p)
    assert back.dtype == np.uint16 and np.array_equal(np.asarray(back), arr)


def test_shard_writer_splits(tmp_path):
    w = ShardWriter(str(tmp_path), "train", "fr", shard_tokens=1000)
    for i in range(25):
        w.add(list(range(i * 100, i * 100 + 100)))
    w.close()
    paths = list_shards(str(tmp_path), "train", "fr")
    assert len(paths) == 3
    allt = np.concatenate([np.asarray(read_shard(p)) for p in paths])
    assert np.array_equal(allt, np.arange(2500))


def test_loader_shapes_and_shift(synth_data):
    d, ranges = synth_data
    L = MixtureLoader(d, ["en", "es", "fr"], seq_len=32, micro_batch=4, seed=0)
    x, y, langs = L.next_batch()
    assert x.shape == y.shape == (4, 32) and x.dtype == torch.long
    # y is x shifted by one inside the same window
    assert torch.equal(x[:, 1:], y[:, :-1])
    # every row comes from a single language range
    for row, li in zip(x, langs):
        lo, hi = ranges[["en", "es", "fr"][li]]
        assert row.min() >= lo and row.max() < hi


def test_mixture_weights(synth_data):
    d, ranges = synth_data
    L = MixtureLoader(d, ["en", "es", "fr"], 16, 64, weights={"en": 0.5, "es": 0.25, "fr": 0.25}, seed=1)
    counts = np.zeros(3)
    for _ in range(50):
        _, _, langs = L.next_batch()
        counts += np.bincount(langs.numpy(), minlength=3)
    frac = counts / counts.sum()
    assert np.allclose(frac, [0.5, 0.25, 0.25], atol=0.03)


def test_epoch_covers_all_windows_without_repeats(synth_data):
    d, _ = synth_data
    L = MixtureLoader(d, ["fr"], 64, 1, seed=3)
    n = L.chunks["fr"].n_chunks
    seen = set()
    for _ in range(n):
        x, _, _ = L.next_batch()
        seen.add(tuple(x[0, :8].tolist()))
    assert len(seen) == n and L.epoch["fr"] == 0
    L.next_batch()
    assert L.epoch["fr"] == 1


def test_resume_state_equivalence(synth_data):
    d, _ = synth_data
    a = MixtureLoader(d, ["en", "es", "fr"], 32, 4, seed=7)
    for _ in range(13):
        a.next_batch()
    sd = a.state_dict()
    ref = [a.next_batch()[0] for _ in range(5)]
    b = MixtureLoader(d, ["en", "es", "fr"], 32, 4, seed=7)
    b.load_state_dict(sd)
    got = [b.next_batch()[0] for _ in range(5)]
    assert all(torch.equal(r, g) for r, g in zip(ref, got))


def test_ddp_ranks_partition_global_batch(synth_data):
    d, _ = synth_data
    single = MixtureLoader(d, ["en", "es", "fr"], 32, 8, seed=5, rank=0, world_size=1)
    r0 = MixtureLoader(d, ["en", "es", "fr"], 32, 4, seed=5, rank=0, world_size=2)
    r1 = MixtureLoader(d, ["en", "es", "fr"], 32, 4, seed=5, rank=1, world_size=2)
    for _ in range(3):
        s = single.next_batch()[0]
        a, b = r0.next_batch()[0], r1.next_batch()[0]
        assert torch.equal(torch.cat([a, b]), s)


def test_valset_sequential(synth_data):
    d, _ = synth_data
    v = ValSet(d, "es", 32, max_chunks=10)
    rows = [x for x, _ in v.batches(4)]
    assert sum(r.shape[0] for r in rows) == 10
    r0 = [x for x, _ in v.batches(4, rank=0, world_size=2)]
    r1 = [x for x, _ in v.batches(4, rank=1, world_size=2)]
    assert sum(r.shape[0] for r in r0 + r1) == 10
