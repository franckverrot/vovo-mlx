"""Word -> IPA tokens from the open-dict-data `ipa-dict` en_US file (MIT). First pronunciation wins;
`overrides.txt` (same format, `#` comments) wins over the dictionary."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from .phones import tokenize_ipa


class Lexicon:
    def __init__(self, text: str, overrides: str | None = None) -> None:
        entries: dict[str, list[str]] = {}
        self.override_count = 0
        if overrides:
            for line in overrides.split("\n"):
                if line.startswith("#") or "\t" not in line:
                    continue
                word, ipa = line.split("\t", 1)
                entries[word.lower()] = tokenize_ipa(ipa.strip("/ "))
                self.override_count += 1
        for line in text.split("\n"):
            if "\t" not in line:
                continue
            word, rest = line.split("\t", 1)
            rest = rest.split(",", 1)[0]
            word = word.lower()
            if word not in entries:
                entries[word] = tokenize_ipa(rest.strip("/ "))
        self.entries = entries

    def __getitem__(self, word: str) -> list[str] | None:
        return self.entries.get(word)

    def __contains__(self, word: str) -> bool:
        return word in self.entries

    def __len__(self) -> int:
        return len(self.entries)


@lru_cache(maxsize=1)
def english() -> Lexicon:
    """The bundled English lexicon (ipa-dict en_US + Vovo overrides)."""
    data = resources.files("vovo_mlx.text") / "data"
    return Lexicon((data / "en_US.txt").read_text(encoding="utf-8"), (data / "overrides.txt").read_text(encoding="utf-8"))
