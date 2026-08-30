"""`vovo-mlx` command line: say a sentence, list phones, or print a checkpoint's config."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import SAMPLE_RATE, VovoTTS, hub
from .audio import write_wav
from .text import G2P, normalize


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vovo-mlx", description="Vovo text-to-speech on Apple silicon (MLX).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("say", help="Synthesize a sentence to a WAV file.")
    s.add_argument("text")
    s.add_argument("-o", "--out", default="out.wav")
    s.add_argument("--repo", default=hub.DEFAULT_REPO, help="Hub repo id or local directory with model.safetensors + vocoder.safetensors")
    s.add_argument("--model-file", default=hub.MODEL_FILE)
    s.add_argument("--vocoder-file", default=hub.VOCODER_FILE)
    s.add_argument("--steps", type=int, default=16)
    s.add_argument("--guidance", type=float, default=2.0)
    s.add_argument("--temperature", type=float, default=0.667)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--sway", type=float, default=0.0)
    s.add_argument("--midpoint", action="store_true")
    s.add_argument("--seed", type=int)
    s.add_argument("--play", action="store_true", help="Play with afplay when done.")

    g = sub.add_parser("phones", help="Show normalization and phonemization of text.")
    g.add_argument("text")

    args = p.parse_args(argv)
    if args.command == "phones":
        print("normalized:", normalize(args.text))
        toks = G2P().phonemize(args.text)
        print("phones:    ", "".join(toks))
        return 0

    t0 = time.time()
    tts = VovoTTS.from_pretrained(args.repo, model_file=args.model_file, vocoder_file=args.vocoder_file)
    t1 = time.time()
    wav = tts.say(args.text, steps=args.steps, guidance=args.guidance, temperature=args.temperature,
                  speed=args.speed, sway=args.sway, midpoint=args.midpoint, seed=args.seed)
    t2 = time.time()
    write_wav(args.out, wav, SAMPLE_RATE)
    secs = len(wav) / SAMPLE_RATE
    print(f"{secs:.2f}s of audio in {t2 - t1:.2f}s (RTF {(t2 - t1) / secs:.3f}); load {t1 - t0:.2f}s; wrote {args.out}")
    if args.play:
        subprocess.run(["afplay", args.out], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
