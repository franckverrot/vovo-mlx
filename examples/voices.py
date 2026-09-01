"""Browse the voices a multi-voice checkpoint carries.

    python examples/voices.py            # one line per voice, with its pitch
    python examples/voices.py p226 p310  # render just these

Vovo2 carries 110: `ljspeech` (the original voice, 24 h of audiobook) and 109 from VCTK. Pick by name or
by id — `tts.say(text, speaker="p226")` and `speaker=2` are the same voice.
"""

import math
import sys
from pathlib import Path

from vovo_mlx import VovoTTS, SAMPLE_RATE
from vovo_mlx.audio import write_wav

TEXT = "The quick brown fox jumps over the lazy dog."


def main() -> None:
    tts = VovoTTS.from_pretrained()
    if not tts.voices:
        print("This checkpoint is single-voice. Load Vovo2 for the 110-voice model.")
        return

    wanted = sys.argv[1:]
    if not wanted:
        print(f"{len(tts.voices)} voices — pass names to render some, e.g. python examples/voices.py p226 p310\n")
        for i, name in enumerate(tts.voices):
            # f0Mean is stored in log-Hz per speaker; it is what the model normalizes pitch against.
            hz = math.exp(tts.config.f0Mean[i]) if i < len(tts.config.f0Mean) else 0
            print(f"  {i:3d}  {name:<10}{f'~{hz:.0f} Hz' if hz else ''}")
        return

    out = Path("voices_out")
    out.mkdir(exist_ok=True)
    for name in wanted:
        wav = tts.say(TEXT, speaker=name, seed=0)   # same seed: the voice is the only thing that differs
        path = out / f"{name}.wav"
        write_wav(path, wav, SAMPLE_RATE)
        print(f"  {path}  {len(wav) / SAMPLE_RATE:.2f} s")


if __name__ == "__main__":
    main()
