"""The Vovo phone inventory. The order *is* the token id: never reorder, only append.

Mirrors `Sources/VovoText/PhoneSet.swift` in the Swift reference implementation.
"""

from __future__ import annotations

PAD, BOS, EOS, UNK, SPACE = 0, 1, 2, 3, 4

SYMBOLS: list[str] = [
    "<pad>", "<bos>", "<eos>", "<unk>", " ",
    # punctuation (kept: it carries pause/intonation information)
    ",", ".", "?", "!", ";", ":", "—", "…",
    # stress
    "ˈ", "ˌ",
    # English consonants (ipa-dict en_US inventory)
    "b", "d", "f", "ɡ", "h", "j", "k", "ɫ", "l", "m", "n", "ŋ", "p", "ɹ", "s", "ʃ", "t", "θ", "ð", "v", "w", "z", "ʒ", "ʔ",
    # English vowels
    "æ", "ɑ", "ɔ", "ə", "ɛ", "ɪ", "i", "ʊ", "u", "ʌ", "ɚ", "ɝ", "e", "o", "a",
    # reserved for French
    "ʁ", "y", "ø", "œ", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃", "ɥ", "ɲ", "ː",
    # language tags
    "<lang:en>", "<lang:fr>",
]

ID_OF: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = len(SYMBOLS)


def tokenize_ipa(ipa: str) -> list[str]:
    """Longest-match tokenization of an IPA string (handles multi-codepoint tokens like "ɛ̃").

    Unknown marks (length marks, ties...) are dropped, exactly like the Swift implementation.
    """
    out: list[str] = []
    scalars = list(ipa)  # Python str iterates over code points, like Swift's unicodeScalars
    i = 0
    while i < len(scalars):
        matched = False
        for length in range(min(3, len(scalars) - i), 0, -1):
            s = "".join(scalars[i : i + length])
            if s in ID_OF:
                out.append(s)
                i += length
                matched = True
                break
        if not matched:
            i += 1
    return out


def encode(tokens: list[str]) -> list[int]:
    return [ID_OF.get(t, UNK) for t in tokens]


def decode(ids: list[int]) -> str:
    return "".join(SYMBOLS[i] if i < len(SYMBOLS) else "?" for i in ids)
