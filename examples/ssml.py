"""SSML: steer prosody per word instead of per utterance — and see what the markup actually did.

    python examples/ssml.py [out_dir]

`tts.say()` detects markup on its own, so SSML is just text you pass in. The interesting part is that you
can inspect the plan before synthesizing: which phones each span produced, and what control landed on them.

Needs a checkpoint with the variance adaptor (Vovo1.5 and later) for the pitch and emphasis tags; `<break>`
works on any checkpoint because it is a real pause token with a pinned duration.
"""

import sys
from pathlib import Path

import os

from vovo_mlx import VovoTTS, SAMPLE_RATE, plan_ssml
from vovo_mlx.audio import write_wav

EXAMPLES = [
    ("emphasis",
     '<speak>I said <emphasis level="strong">red</emphasis>, not blue.</speak>',
     "one word pushed up in pitch and energy"),
    ("question",
     '<speak>You moved it to <prosody pitch="+4st">Thursday</prosody>?</speak>',
     "a rise on the last content word — the shape of a question"),
    ("pause",
     '<speak>Wait.<break time="700ms"/>Say that again.</speak>',
     "a pause of an exact length, not a guess from punctuation"),
    ("aside",
     '<speak>The build passed <prosody pitch="-2st" volume="-4dB" rate="1.15">(finally)</prosody> '
     'and we shipped.</speak>',
     "lower, quieter, quicker — a parenthetical said as one"),
    ("flat_then_lively",
     '<speak><prosody range="0.3">This part is monotone.</prosody> '
     '<prosody range="1.8">This part is not!</prosody></speak>',
     "`range` scales the contour's variance, per span"),
    ("substitution",
     '<speak>Ship it by <sub alias="November third">Nov 3</sub>.</speak>',
     "say something other than what is written"),
]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "ssml_out")
    out.mkdir(exist_ok=True)

    # VOVO_WEIGHTS can point at a local directory holding model.safetensors + vocoder.safetensors
    tts = VovoTTS.from_pretrained(os.environ.get("VOVO_WEIGHTS") or "franckverrot/vovo")
    if not tts.config.varianceAdaptor:
        print("! this checkpoint has no variance adaptor: <prosody pitch/range> and <emphasis> will be")
        print("! ignored. <break>, <sub> and rate still work. Load a Vovo1.5+ checkpoint for the rest.\n")

    for name, markup, note in EXAMPLES:
        phones, control, spans = plan_ssml(markup, tts.g2p)
        wav = tts.say(markup, seed=0)
        write_wav(out / f"{name}.wav", wav, SAMPLE_RATE)

        print(f"{name}: {note}")
        print(f"  {markup}")
        print(f"  {len(spans)} spans -> {len(phones)} phones, {len(wav) / SAMPLE_RATE:.2f} s")
        # Show only the phones the markup actually touched — the rest are at their defaults.
        steered = [(i, c) for i, c in enumerate(control)
                   if c.pitch_shift or c.pitch_scale != 1 or c.energy_shift or c.speed != 1 or c.duration_frames]
        if steered:
            lo, hi = steered[0][0], steered[-1][0]
            c = steered[0][1]
            print(f"  control on phones {lo}-{hi}: pitch {c.pitch_shift:+.2f}st, scale {c.pitch_scale:.2f}, "
                  f"energy {c.energy_shift:+.1f}dB, speed {c.speed:.2f}"
                  + (f", pinned to {c.duration_frames} frames" if c.duration_frames else ""))
        print()

    print(f"{len(EXAMPLES)} files in {out}/.")
    print("Tags compose with the scalar knobs: tts.say(markup, pitch_shift=2) shifts everything, and the")
    print("markup still moves individual spans relative to that.")


if __name__ == "__main__":
    main()
