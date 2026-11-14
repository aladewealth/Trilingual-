"""Language identification for EN/ES/FR generations.

Primary: fastText lid.176 (if `fasttext` is importable and the model file is available).
Fallback: a transparent stopword + orthography heuristic (no dependencies).
"""
from __future__ import annotations

import os
import re
import urllib.request
from collections import Counter
from typing import Optional, Tuple

_STOP = {
    "en": "the of and to in is that for it was on with as are be this by at from or have an not which but they his her you were their has had we can all will would there been more also its who into when other some what".split(),
    "es": "de la que el en y los del se las por un para con una no su al es lo como más pero sus le ya o este fue ha sí porque esta son entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante uno les ni contra otros ese eso".split(),
    "fr": "de la le et les des en un une du est que pour qui dans par sur pas au plus il ne se ce sont avec à son elle nous vous mais ou cette ont été aux leur fait comme tout être sa ses lui ces y bien sans peut".split(),
}
_STOPSETS = {k: set(v) for k, v in _STOP.items()}
_CHARS = {"es": set("ñ¿¡áíóú"), "fr": set("çàèêëîïôûùœâ«»"), "en": set()}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def heuristic_langid(text: str) -> Tuple[str, float]:
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return "unk", 0.0
    scores = Counter()
    for l, s in _STOPSETS.items():
        scores[l] = sum(1 for w in words if w in s) / len(words)
    low = text.lower()
    for l, cs in _CHARS.items():
        if cs:
            scores[l] += 0.5 * sum(low.count(c) for c in cs) / max(1, len(words))
    lang, best = scores.most_common(1)[0]
    total = sum(scores.values()) or 1.0
    return (lang, best / total) if best > 0 else ("unk", 0.0)


_FT = None
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def _load_fasttext(cache_dir: str = os.path.expanduser("~/.cache/trilingua")):
    global _FT
    if _FT is not None:
        return _FT if _FT is not False else None
    try:
        import fasttext  # type: ignore
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "lid.176.ftz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(FT_URL, path)
        fasttext.FastText.eprint = lambda *a, **k: None
        _FT = fasttext.load_model(path)
    except Exception:
        _FT = False
        return None
    return _FT


def langid(text: str, backend: str = "auto") -> Tuple[str, float, str]:
    """Returns (lang, confidence, backend_used)."""
    text = text.replace("\n", " ").strip()
    if backend in ("auto", "fasttext"):
        ft = _load_fasttext()
        if ft is not None and text:
            try:
                # low-level call avoids the fasttext + numpy>=2 incompatibility in FastText.predict()
                prob, label = ft.f.predict(text + "\n", 1, 0.0, "strict")[0]
                return label.replace("__label__", ""), float(prob), "fasttext"
            except Exception:
                pass
        if backend == "fasttext":
            raise RuntimeError("fasttext unavailable")
    l, c = heuristic_langid(text)
    return l, c, "heuristic"
