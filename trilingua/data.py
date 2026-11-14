"""Token shards (uint16 .bin) and a resumable, DDP-aware language-mixture loader.

Shard format: 256 x int32 header [MAGIC, VERSION, n_tokens, 0...] followed by
n_tokens uint16 token ids. File names: {split}_{lang}_{idx:04d}.bin

The loader treats each language's training tokens as non-overlapping windows
of seq_len+1 tokens (stride seq_len), shuffles them with a seeded permutation
per epoch, and picks the language of every sample with a seeded RNG according
to the mixture weights. All ranks hold the identical state and draw the whole
global micro-batch, then keep their own slice, so resuming only needs
{rng state, per-language cursor/epoch} which goes into the checkpoint.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

MAGIC = 20261220
VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4


def write_shard(path: str, tokens: np.ndarray) -> None:
    tokens = np.asarray(tokens)
    assert tokens.max(initial=0) < 2 ** 16, "token id does not fit in uint16"
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())
    os.replace(tmp, path)


def read_shard(path: str) -> np.memmap:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC, f"bad magic in {path}"
    assert header[1] == VERSION, f"unsupported version in {path}"
    n = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(n,))


class ShardWriter:
    """Accumulates tokens and flushes fixed-size shards."""

    def __init__(self, out_dir: str, split: str, lang: str, shard_tokens: int = 100_000_000):
        self.out_dir, self.split, self.lang = out_dir, split, lang
        self.shard_tokens = shard_tokens
        self.buf: List[np.ndarray] = []
        self.buf_len = 0
        self.idx = 0
        self.total = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, ids: Sequence[int]) -> None:
        a = np.asarray(ids, dtype=np.uint16)
        self.buf.append(a)
        self.buf_len += len(a)
        self.total += len(a)
        while self.buf_len >= self.shard_tokens:
            allt = np.concatenate(self.buf)
            self._flush(allt[: self.shard_tokens])
            rest = allt[self.shard_tokens:]
            self.buf, self.buf_len = ([rest] if len(rest) else []), len(rest)

    def _flush(self, arr: np.ndarray) -> None:
        path = os.path.join(self.out_dir, f"{self.split}_{self.lang}_{self.idx:04d}.bin")
        write_shard(path, arr)
        self.idx += 1

    def close(self) -> None:
        if self.buf_len:
            self._flush(np.concatenate(self.buf))
            self.buf, self.buf_len = [], 0


def list_shards(data_dir: str, split: str, lang: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, f"{split}_{lang}_*.bin")))


class _LangChunks:
    def __init__(self, paths: List[str], seq_len: int):
        assert paths, "no shards"
        self.shards = [read_shard(p) for p in paths]
        self.seq_len = seq_len
        counts = [max(0, (len(s) - 1) // seq_len) for s in self.shards]
        self.cum = np.cumsum([0] + counts)
        self.n_chunks = int(self.cum[-1])
        self.n_tokens = int(sum(len(s) for s in self.shards))
        assert self.n_chunks > 0, "shards too small for seq_len"

    def get(self, k: int) -> np.ndarray:
        s = int(np.searchsorted(self.cum, k, side="right") - 1)
        off = (k - int(self.cum[s])) * self.seq_len
        return np.asarray(self.shards[s][off: off + self.seq_len + 1], dtype=np.int64)


class MixtureLoader:
    def __init__(self, data_dir: str, langs: Sequence[str], seq_len: int, micro_batch: int,
                 weights: Optional[Dict[str, float]] = None, rank: int = 0, world_size: int = 1,
                 seed: int = 1337, split: str = "train", device: str = "cpu"):
        self.langs = list(langs)
        self.seq_len, self.micro_batch = seq_len, micro_batch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.device = device
        self.chunks = {l: _LangChunks(list_shards(data_dir, split, l), seq_len) for l in self.langs}
        if weights is None:  # proportional to available tokens
            weights = {l: self.chunks[l].n_tokens for l in self.langs}
        w = np.array([weights[l] for l in self.langs], dtype=np.float64)
        self.weights = w / w.sum()
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.cursor = {l: 0 for l in self.langs}
        self.epoch = {l: 0 for l in self.langs}
        self._perm = {l: self._make_perm(i, l, 0) for i, l in enumerate(self.langs)}
        self.samples_drawn = 0

    def _make_perm(self, li: int, lang: str, epoch: int) -> np.ndarray:
        r = np.random.Generator(np.random.PCG64([self.seed, li, epoch]))
        return r.permutation(self.chunks[lang].n_chunks)

    def _next_chunk(self, lang: str) -> np.ndarray:
        if self.cursor[lang] >= len(self._perm[lang]):
            self.epoch[lang] += 1
            self.cursor[lang] = 0
            self._perm[lang] = self._make_perm(self.langs.index(lang), lang, self.epoch[lang])
        k = int(self._perm[lang][self.cursor[lang]])
        self.cursor[lang] += 1
        return self.chunks[lang].get(k)

    def next_batch(self):
        """Returns (x, y, lang_idx) for this rank: (micro_batch, seq_len) int64 tensors."""
        n = self.micro_batch * self.world_size
        choice = self.rng.choice(len(self.langs), size=n, p=self.weights)
        rows = [self._next_chunk(self.langs[c]) for c in choice]
        lo, hi = self.rank * self.micro_batch, (self.rank + 1) * self.micro_batch
        batch = torch.from_numpy(np.stack(rows[lo:hi]))
        langs = torch.from_numpy(choice[lo:hi].astype(np.int64))
        self.samples_drawn += n
        x, y = batch[:, :-1], batch[:, 1:]
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y, langs

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "cursor": dict(self.cursor),
                "epoch": dict(self.epoch), "samples_drawn": self.samples_drawn,
                "seed": self.seed, "langs": self.langs}

    def load_state_dict(self, sd: dict) -> None:
        assert sd["langs"] == self.langs and sd["seed"] == self.seed, "loader config changed"
        self.rng.bit_generator.state = sd["rng"]
        self.cursor, self.epoch = dict(sd["cursor"]), dict(sd["epoch"])
        self.samples_drawn = sd["samples_drawn"]
        self._perm = {l: self._make_perm(i, l, self.epoch[l]) for i, l in enumerate(self.langs)}

    def tokens_per_lang(self) -> Dict[str, int]:
        return {l: self.chunks[l].n_tokens for l in self.langs}


class ValSet:
    """Deterministic sequential windows from a language's val shards."""

    def __init__(self, data_dir: str, lang: str, seq_len: int, max_chunks: Optional[int] = None):
        self.c = _LangChunks(list_shards(data_dir, "val", lang), seq_len)
        self.n = self.c.n_chunks if max_chunks is None else min(max_chunks, self.c.n_chunks)

    def batches(self, batch_size: int, rank: int = 0, world_size: int = 1, device: str = "cpu"):
        idx = list(range(rank, self.n, world_size))
        for i in range(0, len(idx), batch_size):
            rows = np.stack([self.c.get(k) for k in idx[i:i + batch_size]])
            t = torch.from_numpy(rows).to(device)
            yield t[:, :-1], t[:, 1:]


def load_meta(data_dir: str) -> dict:
    p = os.path.join(data_dir, "meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}
