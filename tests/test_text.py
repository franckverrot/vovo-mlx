"""The text front-end must produce exactly the ids the Swift reference produces (`vovo g2p`)."""

import json
from pathlib import Path

import pytest

from vovo_mlx.text import G2P, normalize, tokenize_ipa, encode, SYMBOLS

GOLDEN = json.loads((Path(__file__).parent / "golden_phones.json").read_text())


@pytest.fixture(scope="module")
def g2p():
    return G2P()


@pytest.mark.parametrize("case", GOLDEN, ids=[c["text"][:40] for c in GOLDEN])
def test_matches_swift_reference(g2p, case):
    assert normalize(case["text"]) == case["normalized"]
    assert g2p.encode(case["text"]) == case["ids"]


def test_phone_inventory_is_stable():
    assert len(SYMBOLS) == 67
    assert SYMBOLS[:5] == ["<pad>", "<bos>", "<eos>", "<unk>", " "]


def test_tokenize_multicodepoint():
    assert tokenize_ipa("ɛ̃") == ["ɛ̃"]
    assert encode(tokenize_ipa("ˈvoʊvoʊ")) == encode(["ˈ", "v", "o", "ʊ", "v", "o", "ʊ"])


def test_lexicon_and_overrides(g2p):
    assert "".join(g2p.pronounce("vovo")) == "ˈvoʊvoʊ"
    assert g2p.lexicon.override_count >= 3
    assert len(g2p.lexicon) > 100_000
