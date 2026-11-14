import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def synth_data(tmp_path):
    """Tiny synthetic shards for en/es/fr (vocab 300) with per-language token ranges."""
    from trilingua.data import write_shard
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir()
    ranges = {"en": (10, 100), "es": (100, 200), "fr": (200, 300)}
    for lang, (lo, hi) in ranges.items():
        n_train = 20000 if lang == "en" else 10000
        write_shard(str(d / f"train_{lang}_0000.bin"), rng.integers(lo, hi, n_train))
        write_shard(str(d / f"train_{lang}_0001.bin"), rng.integers(lo, hi, 3000))
        write_shard(str(d / f"val_{lang}_0000.bin"), rng.integers(lo, hi, 2000))
    return str(d), ranges
