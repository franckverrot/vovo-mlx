"""Render a text file, one sentence per line, into numbered WAVs — and show the sampler knobs."""

import sys
import time
from pathlib import Path

from vovo_mlx import VovoTTS, SAMPLE_RATE
from vovo_mlx.audio import write_wav

lines = [l.strip() for l in Path(sys.argv[1] if len(sys.argv) > 1 else "sentences.txt").read_text().splitlines() if l.strip()]
out = Path(sys.argv[2] if len(sys.argv) > 2 else "out")
out.mkdir(exist_ok=True)

tts = VovoTTS.from_pretrained()
t0 = time.time()
total = 0.0
for i, text in enumerate(lines):
    # guidance 2 / 16 Euler steps is the default trade-off; try steps=8, sway=-1, midpoint=True for speed,
    # temperature < 0.667 for a steadier (but flatter) voice, speed > 1 to talk faster.
    wav = tts.say(text, steps=16, guidance=2.0, seed=i)
    write_wav(out / f"{i:02d}.wav", wav, SAMPLE_RATE)
    total += len(wav) / SAMPLE_RATE
print(f"{len(lines)} sentences, {total:.1f} s of audio in {time.time() - t0:.1f} s")
