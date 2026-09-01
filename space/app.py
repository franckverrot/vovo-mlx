"""Hugging Face Space for Vovo: type a sentence, hear the voice. Runs vovo-mlx on the Space's CPU."""

from __future__ import annotations

import math
import os
import time

import gradio as gr
import mlx.core as mx
import numpy as np

from vovo_mlx import SAMPLE_RATE, VovoTTS, plan_ssml, __version__
from vovo_mlx.text.phones import decode as decode_phones
from vovo_mlx.text.ssml import SSMLError, looks_like_markup

MODEL_REPO = "franckverrot/vovo"
tts = VovoTTS.from_pretrained(MODEL_REPO)
tts.say("Warming up.", steps=2)  # first call compiles the graphs; do it before the first visitor
if os.environ.get("VOVO_PROFILE", "1") == "1":
    import space_profile as _profile  # noqa: E402  (timings into the Space logs)

    _profile.run(tts)

def voice_choices() -> list[tuple[str, str]]:
    """(label, value) for the dropdown — the name alone means nothing, the pitch makes it browsable."""
    names = tts.voices
    if not names:
        return []
    out = []
    for i, n in enumerate(names):
        hz = math.exp(tts.config.f0Mean[i]) if i < len(tts.config.f0Mean) else 0
        label = f"{n} · {hz:.0f} Hz" if hz else n
        out.append((label if n != "ljspeech" else f"{label}  (the original voice)", n))
    return out


VOICES = voice_choices()

EXAMPLES = [
    ["The quick brown fox jumps over the lazy dog."],
    ["Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the exhibition."],
    ["Dr. Smith paid $12.50 on the 3rd of May, 1999, at 4:05."],
    ["Vovo is a text-to-speech model written in Swift, with hand-written Metal kernels, trained in thirteen minutes."],
    ["It's 7:30; she'll arrive at 10:00, maybe 10:15, with 1,250 acres of paperwork."],
    ['<speak>I said <emphasis level="strong">red</emphasis>,<break time="700ms"/>not <prosody pitch="+4st">blue</prosody>.</speak>'],
    ['<speak><prosody range="0.3">This part is monotone.</prosody> <prosody range="1.8">This part is not!</prosody></speak>'],
    ['<speak>The build passed <prosody pitch="-2st" volume="-4dB" rate="1.15">(finally)</prosody> and we shipped.</speak>'],
]


def synthesize(text: str, steps: int, guidance: float, temperature: float, speed: float, seed: int, sway: float,
               midpoint: bool, pitch_shift: float = 0.0, pitch_scale: float = 1.0, energy_shift: float = 0.0,
               voice: str = "ljspeech"):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a sentence first.")
    if len(text) > 600:
        raise gr.Error("Keep it under 600 characters — this Space runs on a small CPU.")
    t0 = time.time()
    try:
        wav = tts.say(text, speaker=(voice or "ljspeech") if VOICES else 0, steps=int(steps), guidance=float(guidance), temperature=float(temperature), speed=float(speed),
                      sway=float(sway), midpoint=bool(midpoint), seed=int(seed) if seed >= 0 else None,
                      pitch_shift=float(pitch_shift), pitch_scale=float(pitch_scale), energy_shift=float(energy_shift))
    except SSMLError as e:
        raise gr.Error(f"SSML: {e}")
    dt = time.time() - t0
    secs = len(wav) / SAMPLE_RATE
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    if looks_like_markup(text):
        ids, _, spans = plan_ssml(text, tts.g2p)
        phones = decode_phones(ids) + f"   ({len(spans)} SSML spans)"
    else:
        phones = "".join(tts.phonemize(text))
    device = "GPU" if mx.default_device().type == mx.DeviceType.gpu else "CPU"
    who = f"{voice} · " if VOICES else ""
    info = f"{who}{secs:.2f} s of audio in {dt:.1f} s on {device} (RTF {dt / secs:.2f}) · {int(steps)} steps, guidance {guidance:g}, seed {int(seed)}"
    return (SAMPLE_RATE, pcm), phones, info


with gr.Blocks(title="Vovo — text to speech") as demo:
    gr.Markdown(
        f"""
# Vovo — a from-scratch text-to-speech model

Vovo is a 21 M-parameter English text-to-speech model written in **Swift with hand-written Metal kernels** — its own
tensor engine, autograd, optimizer and trainer — trained on a laptop in about three hours. It carries **110 voices** and
predicts pitch and energy per sound, so you can steer the delivery with the sliders **or with SSML** (try the last three
examples). This Space runs the
Python/MLX port ([`vovo-mlx`](https://github.com/franckverrot/vovo-mlx) v{__version__}) on a CPU, so expect a few
seconds per sentence; on Apple silicon it is ~20× faster than real time.
Weights: [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}) · MIT.
"""
    )
    with gr.Row():
        with gr.Column(scale=3):
            voice = gr.Dropdown(VOICES, value="ljspeech", label="Voice",
                                info=f"{len(VOICES)} voices — the original plus VCTK's speakers, labelled by pitch",
                                visible=bool(VOICES), filterable=True)
            text = gr.Textbox(label="Text", value=EXAMPLES[0][0], lines=3, max_lines=6)
            with gr.Row():
                steps = gr.Slider(2, 32, value=16, step=2, label="ODE steps", info="fewer = faster, blurrier")
                guidance = gr.Slider(1.0, 4.0, value=2.0, step=0.5, label="Guidance", info="1 = off; higher = crisper, brighter")
            with gr.Row():
                temperature = gr.Slider(0.2, 1.0, value=0.667, step=0.05, label="Temperature")
                speed = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label="Speed")
            with gr.Row():
                pitch_shift = gr.Slider(-12, 12, value=0, step=0.5, label="Pitch (semitones)", info="whole utterance")
                pitch_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Range", info="< 1 flatter, > 1 more animated")
                energy_shift = gr.Slider(-12, 12, value=0, step=1, label="Effort (dB)", info="the voice pushes or eases")
            with gr.Accordion("More", open=False):
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed (−1 = random)")
                    sway = gr.Slider(-1.0, 0.0, value=0.0, step=0.25, label="Sway", info="< 0 packs steps near t = 0")
                    midpoint = gr.Checkbox(value=False, label="Midpoint solver")
            button = gr.Button("Speak", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Output (24 kHz)", type="numpy", autoplay=True)
            phones = gr.Textbox(label="Phones the model saw", interactive=False)
            info = gr.Markdown()
    inputs = [text, steps, guidance, temperature, speed, seed, sway, midpoint, pitch_shift, pitch_scale, energy_shift, voice]
    outputs = [audio, phones, info]
    button.click(synthesize, inputs, outputs, api_name="synthesize")
    text.submit(synthesize, inputs, outputs)
    gr.Examples(EXAMPLES, inputs=[text], outputs=outputs, fn=lambda t: synthesize(t, 16, 2.0, 0.667, 1.0, 0, 0.0, False, 0.0, 1.0, 0.0, "ljspeech"), cache_examples=False)
    gr.Markdown(
        """
**Knobs.** *Steps*: Euler ODE steps of the flow-matching decoder. *Guidance*: classifier-free guidance — 1 is off,
2 is the default trade-off, above 3 gets bright and hissy. *Temperature*: scale of the starting noise (lower = steadier,
flatter). *Speed*: > 1 talks faster. Same seed + same settings = same audio.

**Voices.** 110 of them: `ljspeech` is the original (24 h of one audiobook reader), the rest come from VCTK and are
labelled with their pitch so you can find a register. A single-voice checkpoint scores better on word error rate —
21 M parameters shared across 110 voices leave less for any one of them — so this Space trades accuracy for choice.

**Prosody.** *Pitch* shifts the whole utterance in semitones. *Range* scales the pitch contour about its own mean —
0.3 is a monotone, 1.8 is theatrical. *Effort* is vocal effort in dB, which is not the same as volume: the voice
pushes or eases rather than just getting louder.

**SSML.** Paste markup and it is detected automatically: `<prosody pitch|range|rate|volume>`, `<emphasis>`,
`<break time="700ms"/>`, `<p>`, `<s>`, `<sub alias="…">`. A break is a real pause token with its duration pinned,
not a guess from punctuation. Tags steer individual words and compose with the sliders above.

**Text.** Numbers, money, times, ordinals and common abbreviations are normalized ("$12.50", "4:05", "3rd", "Dr.");
words outside the lexicon are spelled by rules, so unusual names may come out odd. English only, one voice.
"""
    )

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
