"""Model loading from a local dir / checkpoint / HF Hub repo, and prompt construction for the demo + CLI."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from .sampling import generate_stream
from .tokenizer import TrilinguaTokenizer
from .utils import load_model

WRITE_TEMPLATES = {
    "en": "Write a paragraph about {topic}.",
    "es": "Escribe un párrafo sobre {topic}.",
    "fr": "Écris un paragraphe sur {topic}.",
}


@dataclass
class Bundle:
    model: torch.nn.Module
    tok: TrilinguaTokenizer
    sft: bool
    source: str


def load_bundle(source: str, tokenizer: Optional[str] = None, subfolder: Optional[str] = None,
                device: str = "cpu", revision: Optional[str] = None) -> Bundle:
    """source: exported dir, a training .pt checkpoint, or an HF repo id like 'aladewealth/trilingua'."""
    if os.path.isdir(source):
        d = os.path.join(source, subfolder) if subfolder else source
        model = load_model(d, device)
        tok = TrilinguaTokenizer.from_file(tokenizer or os.path.join(d, "tokenizer.json"))
        gc = os.path.join(d, "generation_config.json")
        sft = json.load(open(gc)).get("sft", False) if os.path.exists(gc) else False
    elif os.path.isfile(source):
        assert tokenizer, "--tokenizer is required with a raw checkpoint"
        model = load_model(source, device)
        tok = TrilinguaTokenizer.from_file(tokenizer)
        sft = bool(torch.load(source, map_location="cpu", weights_only=False, mmap=True).get("sft", False))
    else:
        from huggingface_hub import hf_hub_download
        files = ["config.json", "model.safetensors", "tokenizer.json", "generation_config.json"]
        paths = [hf_hub_download(source, f, subfolder=subfolder, revision=revision) for f in files]
        return load_bundle(os.path.dirname(paths[0]), device=device)
    return Bundle(model, tok, sft, source)


def build_prompt(b: Bundle, lang: str, mode: str, text: str) -> List[int]:
    """mode: 'continue' (raw continuation), 'write' (write about a topic), 'chat' (free instruction, SFT only)."""
    tok = b.tok
    text = text.strip()
    if mode == "continue" or (mode == "chat" and not b.sft):
        return [tok.lang_ids[lang]] + tok.encode(text)
    if mode == "write":
        if b.sft:
            return tok.chat_prompt_ids(lang, WRITE_TEMPLATES[lang].format(topic=text))
        # base model: use the topic as a title and let the model write the article body
        return [tok.lang_ids[lang]] + tok.encode(text[:1].upper() + text[1:] + "\n\n")
    if mode == "chat":
        return tok.chat_prompt_ids(lang, text)
    raise ValueError(mode)


def stream_text(b: Bundle, prompt_ids: List[int], max_new_tokens: int = 200, temperature: float = 0.8,
                top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.1,
                seed: Optional[int] = None) -> Iterator[str]:
    """Yields the cumulative decoded text (handles multi-byte characters split across tokens)."""
    tok = b.tok
    g = None
    if seed is not None:
        g = torch.Generator(device=next(b.model.parameters()).device).manual_seed(seed)
    stops = [tok.eot_id, tok.end_id]
    out: List[int] = []
    for t in generate_stream(b.model, prompt_ids, max_new_tokens, temperature, top_k, top_p,
                             repetition_penalty, stop_ids=stops, no_penalty_ids=tok.special_ids, generator=g):
        if t in stops:
            break
        out.append(t)
        text = tok.decode(out)
        if not text.endswith("\ufffd"):    # wait for the rest of a multi-byte char
            yield text
    yield tok.decode(out)
