"""Text -> phone tokens: lexicon first, then morphological fallbacks, then letter spelling.
Port of `Sources/VovoText/G2P.swift`."""

from __future__ import annotations

from . import phones
from .lexicon import Lexicon, english
from .normalizer import cardinal, normalize

PUNCTUATION_MAP: dict[str, str] = {
    ",": ",", ".": ".", "?": "?", "!": "!", ";": ";", ":": ":", "—": "—", "…": "…",
    "(": ",", ")": ",", "[": ",", "]": ",",
}
_PUNCT = set(PUNCTUATION_MAP.values())
_VOICELESS = {"p", "t", "k", "f", "θ", "s", "ʃ", "h"}


class G2P:
    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or english()
        self.oov_words: dict[str, int] = {}

    def phonemize(self, text: str, normalize_text: bool = True) -> list[str]:
        """Phone tokens for a sentence, including space and punctuation tokens."""
        s = normalize(text) if normalize_text else text
        tokens: list[str] = []
        word = ""

        def flush() -> None:
            nonlocal word
            if not word:
                return
            if tokens and tokens[-1] not in _PUNCT:
                tokens.append(" ")
            tokens.extend(self.pronounce(word))
            word = ""

        for ch in s:
            if ch.isalpha() or ch == "'" or ch.isnumeric():
                word += ch
            else:
                flush()  # hyphens act as a word boundary; quotes and other symbols are dropped
                p = PUNCTUATION_MAP.get(ch)
                if p is not None:
                    if tokens and tokens[-1] in _PUNCT:
                        if tokens[-1] == ",":
                            tokens[-1] = p  # keep the stronger mark
                    elif tokens:
                        tokens.append(p)
        flush()
        while tokens and tokens[-1] in (" ", ","):
            tokens.pop()
        return tokens

    def encode(self, text: str, normalize_text: bool = True) -> list[int]:
        return phones.encode(self.phonemize(text, normalize_text))

    def punctuation_only(self, text: str) -> list[str]:
        """Punctuation tokens for a fragment with no words — `phonemize` drops leading punctuation, but an
        SSML span can be a mark on its own and that mark still carries a pause."""
        out: list[str] = []
        for ch in text:
            p = PUNCTUATION_MAP.get(ch)
            if p is None:
                continue
            if out:
                if out[-1] == ",":
                    out[-1] = p
            else:
                out.append(p)
        return out

    def pronounce(self, raw: str) -> list[str]:
        """Phones for one word (letters, digits and apostrophes only)."""
        lex = self.lexicon
        w = raw.lower().strip("'")
        if not w:
            return []
        if (p := lex[w]) is not None:
            return p
        if (p := lex[raw.lower()]) is not None:
            return p
        # Possessive / contractions
        if w.endswith("'s") and (base := lex[w[:-2]]) is not None:
            return base + _s_suffix(base)
        if w.endswith("s'") and (base := lex[w[:-2]]) is not None:
            return base + _s_suffix(base)
        # Plurals / 3rd person
        if w.endswith("es") and (base := lex[w[:-2]]) is not None:
            return base + ["ɪ", "z"]
        if w.endswith("s") and (base := lex[w[:-1]]) is not None:
            return base + _s_suffix(base)
        if w.endswith("ed") and (base := lex[w[:-2]]) is not None:
            return base + _ed_suffix(base)
        if w.endswith("ed") and (base := lex[w[:-1]]) is not None:
            return base + ["d"]
        if w.endswith("ing") and (base := lex[w[:-3]]) is not None:
            return base + ["ɪ", "ŋ"]
        if w.endswith("ing") and (base := lex[w[:-3] + "e"]) is not None:
            return base + ["ɪ", "ŋ"]
        if w.endswith("ly") and (base := lex[w[:-2]]) is not None:
            return base + ["ɫ", "i"]
        if w.endswith("ness") and (base := lex[w[:-4]]) is not None:
            return base + ["n", "ə", "s"]
        # Numbers that slipped through normalization: spell digits
        if all(ch.isnumeric() for ch in w):
            out: list[str] = []
            for ch in w:
                out.extend(lex[cardinal(int(ch))] or [])
            return out
        # Compound split: longest prefix in lexicon + suffix in lexicon
        if len(w) >= 6:
            for i in range(len(w) - 3, 2, -1):
                a, b = w[:i], w[i:]
                pa, pb = lex[a], lex[b]
                if pa is not None and pb is not None:
                    return pa + pb
        self.oov_words[w] = self.oov_words.get(w, 0) + 1
        # Spell it out
        out = []
        for ch in w:
            out.extend(lex[ch] or [])
        return out


def _s_suffix(p: list[str]) -> list[str]:
    if not p:
        return ["z"]
    if p[-1] in ("s", "z", "ʃ", "ʒ"):
        return ["ɪ", "z"]
    return ["s"] if p[-1] in _VOICELESS else ["z"]


def _ed_suffix(p: list[str]) -> list[str]:
    if not p:
        return ["d"]
    if p[-1] in ("t", "d"):
        return ["ɪ", "d"]
    return ["t"] if p[-1] in _VOICELESS else ["d"]
