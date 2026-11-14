import os
import unicodedata

import pytest

from trilingua.tokenizer import (LANG_TOKENS, SPECIAL_TOKENS, TrilinguaTokenizer, train_tokenizer)

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's a well-known pangram, isn't it?",
    "El pingüino Wenceslao hizo kilómetros bajo exhaustiva lluvia y frío, añoraba a su querido cachorro.",
    "¿Dónde está la biblioteca? ¡Qué día tan bonito! El niño comió piña con su señora madre.",
    "Le cœur déçu mais l'âme plutôt naïve, Louÿs rêva de crapaüter en canoë au delà des îles.",
    "« Bonjour », dit le garçon à l'hôtel. Ça va très bien, merci. Où est la gare ?",
    "Numbers: 1234567, 3.14159, 2026-12-20. Emoji: 🙂. Tabs\tand\nnewlines\n\n  indented.",
] * 30

TESTS = [
    "Hello, world! It's 2026 and we're training models.",
    "¿Cómo estás? ¡Muy bien! El señor Muñoz tiene un año más; mañana será piña.",
    "« Ça va ? » demanda-t-elle. Le garçon répondit : « Très bien, à l'école. » Noël, maïs, œuvre.",
    "Mixed: naïve café — déjà vu… ñandú ç Ç Ñ É È Ê À Ù Û Ü Ÿ Œ æ ß",
    "   leading spaces, trailing spaces   \n\nand blank lines\t\ttabs",
    "Unseen chars round-trip too: 你好 Привет ☃ 🚀",
]


@pytest.fixture(scope="module")
def tok():
    t = train_tokenizer(CORPUS, vocab_size=600, min_frequency=1, show_progress=False)
    return TrilinguaTokenizer(t)


def test_special_token_ids(tok):
    for i, s in enumerate(SPECIAL_TOKENS):
        assert tok.tok.token_to_id(s) == i
    assert tok.eot_id == 0
    assert set(tok.lang_ids) == {"en", "es", "fr"}


@pytest.mark.parametrize("text", TESTS)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_nfd_input_normalised_to_nfc(tok):
    nfd = unicodedata.normalize("NFD", "café señor garçon")
    assert nfd != "café señor garçon"
    assert tok.decode(tok.encode(nfd)) == "café señor garçon"
    assert tok.encode(nfd) == tok.encode("café señor garçon")


def test_specials_not_split_and_skipped_on_decode(tok):
    ids = tok.encode_document("Hola, ¿qué tal?", "es")
    assert ids[0] == tok.lang_ids["es"] and ids[-1] == tok.eot_id
    assert tok.decode(ids) == "Hola, ¿qué tal?"
    # special-token strings typed inside text are recognised as specials
    assert tok.lang_ids["fr"] in tok.encode("<|fr|>Bonjour")


def test_token_bytes_sum_equals_utf8_length(tok):
    tb = tok.token_bytes()
    for text in TESTS:
        ids = tok.encode(text)
        assert sum(tb[i] for i in ids) == len(unicodedata.normalize("NFC", text).encode("utf-8"))
    assert all(tb[tok.tok.token_to_id(s)] == 0 for s in SPECIAL_TOKENS)


def test_chat_example_masks_prompt(tok):
    ids, labels = tok.chat_example("fr", "Écris un poème.", "Voici un poème.")
    assert len(ids) == len(labels)
    n_prompt = len(tok.chat_prompt_ids("fr", "Écris un poème."))
    assert all(l == -100 for l in labels[:n_prompt])
    assert labels[n_prompt:] == ids[n_prompt:]
    assert ids[-2:] == [tok.end_id, tok.eot_id]


TRAINED = os.environ.get("TRILINGUA_TOKENIZER")


@pytest.mark.skipif(not TRAINED or not os.path.exists(TRAINED or ""), reason="set TRILINGUA_TOKENIZER to test a trained tokenizer")
@pytest.mark.parametrize("text", TESTS)
def test_trained_tokenizer_roundtrip(text):
    t = TrilinguaTokenizer.from_file(TRAINED)
    assert t.decode(t.encode(text)) == text
