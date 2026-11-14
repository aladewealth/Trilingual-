"""Shared EN/ES/FR byte-level BPE tokenizer (HF `tokenizers`), trained by us.

Text is NFC-normalised first so that precomposed and decomposed accents
("é" vs "e\u0301") map to the same tokens; decode(encode(s)) == NFC(s).
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

EOT = "<|endoftext|>"
PAD = "<|pad|>"
LANG_TOKENS = {"en": "<|en|>", "es": "<|es|>", "fr": "<|fr|>"}
SYSTEM, USER, ASSISTANT, END = "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"
# Order fixes the ids: <|endoftext|> = 0, <|pad|> = 1, <|en|> = 2, <|es|> = 3, <|fr|> = 4, ...
SPECIAL_TOKENS = [EOT, PAD, LANG_TOKENS["en"], LANG_TOKENS["es"], LANG_TOKENS["fr"],
                  SYSTEM, USER, ASSISTANT, END]
LANGS = ("en", "es", "fr")

# GPT-4-style split, adapted: Unicode letters keep accented words together
# (\p{L} covers é, ñ, ç), contractions case-insensitive, numbers in groups of <=3 digits,
# punctuation runs (incl. ¿ ¡ « ») isolated from words.
PRETOKENIZE_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def build_untrained_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], vocab_size: int = 32768, min_frequency: int = 2,
                    show_progress: bool = True, length: Optional[int] = None) -> Tokenizer:
    tok = build_untrained_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    for i, s in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(s) == i, f"special token {s} has id {tok.token_to_id(s)} != {i}"
    return tok


class TrilinguaTokenizer:
    """Thin wrapper with language tags, chat formatting and per-token byte lengths."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.eot_id = tok.token_to_id(EOT)
        self.pad_id = tok.token_to_id(PAD)
        self.lang_ids = {l: tok.token_to_id(t) for l, t in LANG_TOKENS.items()}
        self.user_id = tok.token_to_id(USER)
        self.assistant_id = tok.token_to_id(ASSISTANT)
        self.end_id = tok.token_to_id(END)
        self.special_ids = {tok.token_to_id(s) for s in SPECIAL_TOKENS}
        self._token_bytes = None

    @classmethod
    def from_file(cls, path: str) -> "TrilinguaTokenizer":
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        self.tok.save(path)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def encode_document(self, text: str, lang: str) -> List[int]:
        """Pretraining format: <|lang|> text <|endoftext|>"""
        return [self.lang_ids[lang]] + self.encode(text) + [self.eot_id]

    # ---- chat / instruction format (used for SFT and the demo) ------------------
    def chat_prompt_ids(self, lang: str, user_msg: str) -> List[int]:
        """<|lang|><|user|>{msg}<|end|><|assistant|>  -> model writes reply then <|end|>"""
        return ([self.lang_ids[lang], self.user_id] + self.encode(user_msg) +
                [self.end_id, self.assistant_id])

    def chat_example(self, lang: str, user_msg: str, reply: str):
        """Returns (ids, labels) with loss only on the reply and its <|end|>."""
        prompt = self.chat_prompt_ids(lang, user_msg)
        answer = self.encode(reply) + [self.end_id, self.eot_id]
        ids = prompt + answer
        labels = [-100] * len(prompt) + answer
        return ids, labels

    def token_bytes(self) -> np.ndarray:
        """Number of UTF-8 bytes each token decodes to (0 for special tokens). Used for bits-per-byte."""
        if self._token_bytes is None:
            byte_decoder = {c: b for b, c in _bytes_to_unicode().items()}
            vocab = self.tok.get_vocab()
            arr = np.zeros(self.vocab_size, dtype=np.int32)
            for t, i in vocab.items():
                if i in self.special_ids:
                    continue
                arr[i] = sum(1 for ch in t if ch in byte_decoder)
            self._token_bytes = arr
        return self._token_bytes


def _bytes_to_unicode():
    """GPT-2's reversible byte -> printable unicode map (same as tokenizers' ByteLevel)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
