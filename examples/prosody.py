"""The prosody knobs, one at a time: render the same sentence several ways and hear each control.

    python examples/prosody.py [out_dir]

Needs a checkpoint with the variance adaptor (Vovo1.5 and later). Without it the model has no pitch or
energy predictor, the knobs are ignored, and every file below comes out identical — the script says so
rather than letting you wonder.
"""

import sys
from pathlib import Path

import os

from vovo_mlx import VovoTTS, SAMPLE_RATE
from vovo_mlx.audio import write_wav

SENTENCE = "I told you the meeting was moved to Thursday."

# (filename, kwargs, what to listen for)
VARIANTS = [
    ("plain",         dict(),                      "the model's own reading"),
    ("higher",        dict(pitch_shift=3),         "same delivery, three semitones up"),
    ("lower",         dict(pitch_shift=-3),        "three semitones down"),
    ("flat",          dict(pitch_scale=0.4),       "contour squashed toward the mean — bored, robotic"),
    ("animated",      dict(pitch_scale=1.8),       "contour exaggerated — the peaks get peakier"),
    ("louder",        dict(energy_shift=6),        "+6 dB of vocal effort (not volume: the voice pushes)"),
    ("softer",        dict(energy_shift=-6),       "-6 dB — closer to a murmur"),
    ("fast",          dict(speed=1.25),            "25 % quicker, pitch unchanged"),
    ("slow",          dict(speed=0.8),             "20 % slower"),
]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "prosody_out")
    out.mkdir(exist_ok=True)

    # VOVO_WEIGHTS can point at a local directory holding model.safetensors + vocoder.safetensors
    tts = VovoTTS.from_pretrained(os.environ.get("VOVO_WEIGHTS") or "franckverrot/vovo")
    if not tts.config.varianceAdaptor:
        print("! this checkpoint has no variance adaptor, so pitch_shift / pitch_scale / energy_shift do")
        print("! nothing here. Only `speed` will change. Load a Vovo1.5+ checkpoint to hear the rest.")

    print(f'"{SENTENCE}"\n')
    for name, kwargs, note in VARIANTS:
        # seed is fixed so the only difference between files is the knob, not a different noise draw
        wav = tts.say(SENTENCE, seed=0, **kwargs)
        path = out / f"{name}.wav"
        write_wav(path, wav, SAMPLE_RATE)
        args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "defaults"
        print(f"  {path}  ({args:28}) — {note}")

    print(f"\n{len(VARIANTS)} files in {out}/. The knobs compose: pitch_shift=2, pitch_scale=1.4,")
    print("energy_shift=3 is a plausible 'excited' preset.")


if __name__ == "__main__":
    main()
