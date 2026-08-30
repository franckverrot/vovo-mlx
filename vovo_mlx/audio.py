"""WAV output without extra dependencies."""

from __future__ import annotations

import wave

import numpy as np


def write_wav(path: str, samples, sample_rate: int = 24000) -> None:
    """Write float samples in [-1, 1] as 16-bit PCM."""
    x = np.asarray(samples, dtype=np.float32)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
