"""A small SSML subset — `<speak>`, `<prosody>`, `<emphasis>`, `<break>`, `<p>`, `<s>`, `<sub>` — parsed into
per-span prosody control. Port of `vovo-core/Sources/VovoText/SSML.swift`; `tests/golden_ssml.json` keeps the
two implementations byte-identical.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

PUNCTUATION = {",", ".", "?", "!", ";", ":", "—", "…"}


@dataclass(frozen=True)
class Control:
    """Prosody for one run of text, in the model's units."""
    pitch_shift: float = 0.0   # semitones
    pitch_scale: float = 1.0
    energy_shift: float = 0.0  # dB
    speed: float = 1.0         # > 1 is faster

    def applying(self, o: "Control") -> "Control":
        return Control(self.pitch_shift + o.pitch_shift, self.pitch_scale * o.pitch_scale,
                       self.energy_shift + o.energy_shift, self.speed * o.speed)


@dataclass(frozen=True)
class Text:
    text: str
    control: Control


@dataclass(frozen=True)
class Pause:
    milliseconds: int


Span = Text | Pause


class SSMLError(ValueError):
    pass


def looks_like_markup(s: str) -> bool:
    t = s.strip()
    return t.startswith("<speak") or "<prosody" in t or "<emphasis" in t or "<break" in t


def semitones(v: str) -> float:
    s = v.strip().lower()
    named = {"x-low": -6.0, "low": -3.0, "medium": 0.0, "default": 0.0, "normal": 0.0, "high": 3.0, "x-high": 6.0}
    if s in named:
        return named[s]
    try:
        if s.endswith("st"):
            return float(s[:-2])
        if s.endswith("%"):
            return 12 * math.log2(1 + float(s[:-1]) / 100)
        if s.endswith("hz"):
            return float(s[:-2]) / 20
        return float(s)
    except ValueError:
        return 0.0


def range_scale(v: str) -> float:
    s = v.strip().lower()
    named = {"none": 0.0, "monotone": 0.0, "x-low": 0.25, "low": 0.5, "reduced": 0.5,
             "medium": 1.0, "default": 1.0, "normal": 1.0, "high": 1.5, "x-high": 2.0}
    if s in named:
        return named[s]
    try:
        if s.endswith("%"):
            return max(0.0, 1 + float(s[:-1]) / 100)
        if s.endswith("st"):
            return max(0.0, 1 + float(s[:-2]) / 6)
        return float(s)
    except ValueError:
        return 1.0


def decibels(v: str) -> float:
    s = v.strip().lower()
    named = {"silent": -60.0, "x-soft": -12.0, "soft": -6.0, "medium": 0.0, "default": 0.0, "normal": 0.0,
             "loud": 6.0, "x-loud": 12.0}
    if s in named:
        return named[s]
    try:
        return float(s[:-2]) if s.endswith("db") else float(s)
    except ValueError:
        return 0.0


def rate(v: str) -> float:
    s = v.strip().lower()
    named = {"x-slow": 0.5, "slow": 0.8, "medium": 1.0, "default": 1.0, "normal": 1.0, "fast": 1.25, "x-fast": 1.5}
    if s in named:
        return named[s]
    try:
        return max(0.1, float(s[:-1]) / 100) if s.endswith("%") else max(0.1, float(s))
    except ValueError:
        return 1.0


def emphasis(level: str) -> Control:
    l = (level or "moderate").strip().lower()
    if l == "strong":
        return Control(2.0, 1.3, 3.0, 0.92)
    if l == "reduced":
        return Control(-1.0, 0.7, -3.0, 1.08)
    if l == "none":
        return Control()
    return Control(1.0, 1.15, 1.5, 0.96)


def break_milliseconds(attrs: dict[str, str]) -> int:
    t = attrs.get("time")
    if t:
        t = t.strip().lower()
        try:
            if t.endswith("ms"):
                return int(float(t[:-2]))
            if t.endswith("s"):
                return int(float(t[:-1]) * 1000)
            return int(float(t))
        except ValueError:
            pass
    return {"none": 0, "x-weak": 100, "weak": 150, "strong": 500, "x-strong": 800}.get((attrs.get("strength") or "").lower(), 300)


_ATTR = re.compile(r"([A-Za-z_:][-\w:.]*)\s*=\s*(\"[^\"]*\"|'[^']*')")


def attributes(s: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2)[1:-1] for m in _ATTR.finditer(s)}


def parse(markup: str) -> list[Span]:
    """Markup → spans. Unknown tags keep their text; prosody shifts add and scales multiply."""
    spans: list[Span] = []
    stack: list[tuple[str, Control]] = []
    current = Control()
    text = ""
    sub_depth = 0
    i = 0

    def flush() -> None:
        nonlocal text
        t = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if t:
            spans.append(Text(t, current))
        text = ""

    while i < len(markup):
        ch = markup[i]
        if ch != "<":
            if sub_depth == 0:
                text += ch
            i += 1
            continue
        close = markup.find(">", i)
        if close < 0:
            raise SSMLError(f"SSML: malformed markup near {markup[i:i + 20]!r}")
        raw = markup[i + 1:close].strip()
        i = close + 1
        if raw.startswith("!") or raw.startswith("?"):
            continue

        if raw.startswith("/"):
            tag = raw[1:].strip().lower()
            if tag == "speak":
                continue
            if tag in ("p", "s"):
                flush()
                spans.append(Pause(500 if tag == "p" else 250))
                continue
            if tag == "sub":
                sub_depth = max(0, sub_depth - 1)
            if not stack:
                if tag in ("prosody", "emphasis"):
                    raise SSMLError(f"SSML: </{tag}> closes a tag that is not open")
                continue
            top_tag, top_control = stack[-1]
            if top_tag == tag:
                if tag in ("prosody", "emphasis"):
                    flush()
                current = top_control
                stack.pop()
            continue

        self_closing = raw.endswith("/")
        body = raw[:-1] if self_closing else raw
        name = body.split(" ", 1)[0].lower()
        attrs = attributes(body[len(name):])

        if name == "speak":
            continue
        if name == "break":
            flush()
            spans.append(Pause(break_milliseconds(attrs)))
        elif name in ("prosody", "emphasis"):
            if name == "emphasis":
                delta = emphasis(attrs.get("level", "moderate"))
            else:
                delta = Control(semitones(attrs["pitch"]) if "pitch" in attrs else 0.0,
                                range_scale(attrs["range"]) if "range" in attrs else 1.0,
                                decibels(attrs["volume"]) if "volume" in attrs else 0.0,
                                rate(attrs["rate"]) if "rate" in attrs else 1.0)
            if self_closing:
                continue
            flush()
            stack.append((name, current))
            current = current.applying(delta)
        elif name == "sub":
            if "alias" in attrs:
                text += attrs["alias"]
            if not self_closing:
                stack.append((name, current))
                sub_depth += 1
        elif name in ("p", "s"):
            continue
        elif not self_closing:
            stack.append((name, current))

    flush()
    for tag, _ in stack:
        if tag in ("prosody", "emphasis"):
            raise SSMLError(f"SSML: <{tag}> is never closed")
    return spans
