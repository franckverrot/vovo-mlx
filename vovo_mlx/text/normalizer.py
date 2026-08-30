"""English text normalization: unicode cleanup, abbreviations, numbers, currency, ordinals, times.

A line-by-line port of `Sources/VovoText/Normalizer.swift`; the two must agree token for token,
which `tests/test_text.py` checks against ids produced by the Swift `vovo g2p` command.
"""

from __future__ import annotations

import re
from collections.abc import Callable

ABBREVIATIONS: dict[str, str] = {
    "mr": "mister", "mrs": "misess", "ms": "miss", "dr": "doctor", "st": "saint", "mt": "mount",
    "jr": "junior", "sr": "senior", "prof": "professor", "gen": "general", "col": "colonel",
    "lt": "lieutenant", "sgt": "sergeant", "capt": "captain", "hon": "honorable", "rev": "reverend",
    "vs": "versus", "etc": "et cetera", "no": "number", "co": "company", "inc": "incorporated",
    "ltd": "limited", "dept": "department", "est": "established", "approx": "approximately",
    "ft": "feet", "lb": "pounds", "lbs": "pounds", "oz": "ounces", "hr": "hour", "hrs": "hours",
    "min": "minutes", "sec": "seconds", "vol": "volume", "ave": "avenue", "blvd": "boulevard",
}
# Abbreviations that need a trailing period to be expanded (otherwise they are real words).
PERIOD_ONLY = {"no", "st", "co", "est", "min", "sec", "gen", "col", "vol"}

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def normalize(text: str) -> str:
    s = text
    for a, b in [
        ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "—"), ("—", "—"),
        ("--", "—"), ("…", "…"), (" ", " "), ("\t", " "), ("\n", " "),
    ]:
        s = s.replace(a, b)
    s = _expand_symbols(s)
    s = _expand_abbreviations(s)
    s = _expand_numbers(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ")


def _expand_symbols(s: str) -> str:
    r = s.replace(" & ", " and ").replace("&", " and ").replace("%", " percent").replace(" @ ", " at ")
    r = re.sub(r"(\d)\s*\+\s*(\d)", r"\1 plus \2", r)
    r = re.sub(r"(\d)\s*=\s*(\d)", r"\1 equals \2", r)
    r = re.sub(r"#(\d)", r"number \1", r)
    return r


def _expand_abbreviations(s: str) -> str:
    out = []
    last = 0
    for m in re.finditer(r"\b([A-Za-z]+)\.?(?=\s|$|[,;:!?)])", s):
        whole, word = m.group(0), m.group(1)
        key = word.lower()
        has_period = whole.endswith(".")
        out.append(s[last : m.start()])
        exp = ABBREVIATIONS.get(key)
        if exp is not None and (has_period or key not in PERIOD_ONLY) and (word[0].isupper() or has_period or key in ("etc", "vs")):
            out.append(exp)
            if has_period and m.end() >= len(s):  # keep a sentence-final period
                out.append(".")
        else:
            out.append(whole)
        last = m.end()
    out.append(s[last:])
    return "".join(out)


def _replace(s: str, pattern: str, f: Callable[[list[str]], str]) -> str:
    def sub(m: re.Match) -> str:
        return f([m.group(0)] + [g if g is not None else "" for g in m.groups()])
    return re.sub(pattern, sub, s)


def _expand_numbers(s: str) -> str:
    r = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    r = re.sub(r"(\d),(\d{3})", r"\1\2", r)

    def money(g: list[str]) -> str:
        d, c = int(g[1]), int(g[2])
        t = f"{cardinal(d)} {'dollar' if d == 1 else 'dollars'}"
        if c > 0:
            t += f" {cardinal(c)} {'cent' if c == 1 else 'cents'}"
        return t

    r = _replace(r, r"\$(\d+)\.(\d{2})\b", money)
    r = _replace(r, r"\$(\d+)", lambda g: f"{cardinal(int(g[1]))} {'dollar' if int(g[1]) == 1 else 'dollars'}")
    r = _replace(r, r"£(\d+)", lambda g: f"{cardinal(int(g[1]))} {'pound' if int(g[1]) == 1 else 'pounds'}")

    def clock(g: list[str]) -> str:
        h, m = int(g[1]), int(g[2])
        if m == 0:
            return f"{cardinal(h)} o'clock"
        return f"{cardinal(h)} {'oh ' if m < 10 else ''}{cardinal(m)}"

    r = _replace(r, r"\b(\d{1,2}):(\d{2})\b", clock)
    r = _replace(r, r"\b(\d+)(st|nd|rd|th)\b", lambda g: ordinal(int(g[1])))
    r = _replace(r, r"\b(\d+)\.(\d+)\b", lambda g: f"{cardinal(int(g[1]))} point " + " ".join(cardinal(int(ch)) for ch in g[2]))
    r = _replace(r, r"\b(1[0-9]{3}|20[0-9]{2})\b", lambda g: _year(int(g[1])))
    r = _replace(r, r"-(\d+)\b", lambda g: f"minus {cardinal(int(g[1]))}")

    def number(g: list[str]) -> str:
        digits = g[1]
        if len(digits) > 15:
            return " ".join(cardinal(int(ch)) for ch in digits)
        return cardinal(int(digits))

    r = _replace(r, r"\b(\d+)\b", number)
    return r


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + cardinal(n % 100))
    for value, name in [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]:
        if n >= value:
            return cardinal(n // value) + " " + name + ("" if n % value == 0 else " " + cardinal(n % value))
    return ONES[0]


def ordinal(n: int) -> str:
    irregular = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}
    if n in irregular:
        return irregular[n]
    if n < 20:
        return ONES[n] + "th"
    if n < 100 and n % 10 == 0:
        return TENS[n // 10][:-1] + "ieth"
    c = cardinal(n)
    parts = c.split(" ")
    last_word = parts[-1]
    if last_word in ("hundred", "thousand", "million", "billion"):
        return c + "th"
    last_n = n % 100 if n % 100 < 20 else n % 10
    return " ".join(parts[:-1]) + (" " if len(parts) > 1 else "") + ordinal(last_n)


def _year(y: int) -> str:
    if y % 100 == 0:
        return cardinal(y // 100) + " hundred"
    if y < 2000 or y >= 2010:
        hi, lo = y // 100, y % 100
        return cardinal(hi) + " " + ("oh " + ONES[lo] if lo < 10 else cardinal(lo))
    return cardinal(y)
