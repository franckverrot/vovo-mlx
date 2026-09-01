# vovo-mlx

Vovo is an English-only TTS model built from scratch.  This repository is the Python MLX implementation for it.

You will find weights on [HuggingFace](https://huggingface.co/franckverrot/vovo).

Vovo was written in Swift with hand-written Metal kernels.  The entirety of the training toolchain is home-made: the tensor engine is new, autograd/optimizer/etc are also custom, and the broader data pipeline and trainer as well.  The result is a compact 20M parameters voice model that was trained in about 13 minutes on an M2 Max with 96GB of RAM (the training consumed way less than that).

This repository is the Python port of the *inference* path so the weights can be used from Python: it reproduces the Swift implementation numerically.

## Install

Apple silicon + Python ≥ 3.10. Pick one:

```
pip install vovo-mlx                                        # from PyPI
pip install git+https://github.com/franckverrot/vovo-mlx    # straight from GitHub (latest main)
```

For development (editable install with the test dependencies):

```
git clone git@github.com:franckverrot/vovo-mlx.git && cd vovo-mlx
pip install -e ".[dev]"       # or: uv venv && uv pip install -e ".[dev]"
pytest
```

The weights are downloaded from the Hub on first use and cached by `huggingface_hub` (~135 MB).

## Models

Two checkpoints live in the same Hub repo. They are the same architecture and the same size; they differ in
what they were trained on, and one is not simply better than the other.

| | **Vovo2-21M** (default) | **Vovo1.5-21M** |
| --- | --- | --- |
| voices | **110** — `ljspeech` plus 109 from VCTK | 1 |
| best for | picking a voice, accents, range | the cleanest single voice |
| word error rate | 10.0 % | **3.7 %** |
| pitch error | 107 cents | **74 cents** |
| prosody controls | yes | yes |
| download | 137 MB (model 83 MB + vocoder 54 MB) | same |

```python
tts = VovoTTS.from_pretrained()                                          # Vovo2, 110 voices
tts = VovoTTS.from_pretrained(revision="vovo1.5-21m")                    # the single voice
```

```
vovo-mlx voices                                    # list them, with each voice's pitch
vovo-mlx say "Hello." --speaker p226 -o p226.wav   # by name, or --speaker 2 by id
```

Vovo2's word error rate is genuinely worse, and the reason is interesting: 21 M parameters shared across
110 voices leave less for any one of them. If you want one voice and nothing else, take 1.5.

## Use

```
vovo-mlx say "The quick brown fox jumps over the lazy dog." -o fox.wav --play
```

```python
from vovo_mlx import VovoTTS, SAMPLE_RATE
from vovo_mlx.audio import write_wav

tts = VovoTTS.from_pretrained()                      # downloads + caches the weights from the Hub
wav = tts.say("Hello from Vovo.", seed=0)            # float32 numpy, 24 kHz
write_wav("hello.wav", wav, SAMPLE_RATE)
```



## What's in the box

- **Text front-end**: English normalizer, rule-based fallbacks for out-of-vocabulary words, and Vovo's fixed 67-symbol phone inventory
- **Acoustic model** (~21M). Vovo1.5 and later carry a **variance adaptor**: pitch and energy are
  predicted per phone, which is what the prosody controls below steer
- **Vocoder**: [Vocos](https://github.com/gemelo-ai/vocos) mel-24kHz, fine-tuned by Vovo on its own predicted mels
- Outputs 24 kHz mono.



## Sampler knobs

`tts.say(text, speaker=0, steps=16, guidance=2.0, temperature=0.667, speed=1.0, sway=0.0, midpoint=False,
seed=None, pitch_shift=0.0, pitch_scale=1.0, energy_shift=0.0)`


| argument      | what it does                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `speaker`     | voice id or name (`2`, `"p226"`); `tts.voices` lists them, single-voice models ignore it |
| `steps`       | ODE steps (16 is the default; 8 with `sway=-1, midpoint=True` is faster and close)                    |
| `guidance`    | classifier-free guidance scale; 1 = off, 2 = default (crisper, slightly brighter), 3 = brighter still |
| `temperature` | scale of the starting noise; lower = steadier, flatter                                                |
| `speed`       | > 1 talks faster (durations divided by `speed`)                                                       |
| `sway`        | < 0 packs ODE steps near t = 0 where the flow bends most (try −1)                                     |
| `midpoint`    | 2nd-order solver (two velocity evaluations per step)                                                  |
| `pitch_shift` | semitones up or down, whole utterance (variance-adaptor models only)                                  |
| `pitch_scale` | scales the contour's variance about its mean: < 1 flatter, > 1 more animated                          |
| `energy_shift` | vocal effort in dB — the voice pushes or eases, which is not the same as volume                      |


Lower-level access: `tts.synthesize(text, **knobs)` returns the log-mel (`.mel`), the encoder prior
(`.prior`), the per-phone durations and the starting noise; `tts.vocode(mel)` turns any `[T, 100]` log-mel
(24 kHz, n_fft 1024, hop 256, HTK mel scale, `log(clamp(x, 1e-7))`) into a waveform; `tts.phonemize(text)`
shows the phones the model will see.


## Prosody and SSML

Two ways to control delivery, and they compose.

**Scalar knobs** apply to the whole utterance — `pitch_shift`, `pitch_scale` and `energy_shift` from the
table above, on `tts.say()` and on the CLI:

```
vovo-mlx say "We shipped it." --pitch-shift 2 --pitch-scale 1.4 --energy-shift 3 -o excited.wav
```

**SSML** steers individual words. Markup is detected automatically, so it is just text you pass in:

```python
tts.say('<speak>I said <emphasis level="strong">red</emphasis>,'
        '<break time="700ms"/>not <prosody pitch="+4st">blue</prosody>.</speak>')
```

| tag | what it does |
| --- | --- |
| `<prosody pitch="+4st">` | shift a span, in semitones or `high`/`low`/`+10%` |
| `<prosody range="1.8">` | scale the contour's variance for that span |
| `<prosody rate="1.15">`, `<prosody volume="-4dB">` | speed and vocal effort for that span |
| `<emphasis level="strong">` | a preset of pitch, energy and rate (`moderate`, `reduced` too) |
| `<break time="700ms"/>` | a pause of an exact length — a real pause token with its duration pinned |
| `<p>`, `<s>` | paragraph and sentence pauses (500 ms / 250 ms) |
| `<sub alias="November third">Nov 3</sub>` | say something other than what is written |

To see what markup will do *before* rendering it, `plan_ssml` returns the phones, the per-phone control and
the spans — `examples/ssml.py` prints all three, and the CLI has `vovo-mlx ssml '<speak>…</speak>'`.

The pitch and emphasis controls need a checkpoint **with the variance adaptor**; without one they are
ignored (`<break>`, `<sub>` and `rate` still work). Check with `tts.config.varianceAdaptor`.


## Examples

- [examples/say.py](examples/say.py): two lines, text to WAV
- [examples/voices.py](examples/voices.py): list the 110 voices, render the ones you name
- [examples/prosody.py](examples/prosody.py): every scalar knob, one at a time — the same sentence higher,
  lower, flatter, livelier, louder, softer, faster, slower, from a fixed seed so only the knob differs
- [examples/ssml.py](examples/ssml.py): markup that steers prosody per word, and how to *inspect the plan*
  (which phones each span produced, what control landed on them) before synthesizing
- [examples/batch.py](examples/batch.py): text file to numbered WAVs with some options
- [examples/inspect_mel.py](examples/inspect_mel.py): plot prior μ next to the decoded mel
- `vovo-mlx phones "Dr. Smith paid $12.50 at 4:05."`: see the normalization and phonemization

The prosody and SSML examples need a checkpoint **with the variance adaptor** (Vovo1.5 and later) — pitch
and energy are predicted per phone there, and the knobs move those predictions. On a checkpoint without it
the knobs are ignored and both scripts say so up front rather than leaving you guessing. Point them at a
local export with `VOVO_WEIGHTS=/path/to/dir python examples/prosody.py`.


## Weights

`VovoTTS.from_pretrained("franckverrot/vovo")` fetches two files:


| file                  | what                                                                                |
| --------------------- | ----------------------------------------------------------------------------------- |
| `model.safetensors`   | acoustic model, EMA weights, config in the safetensors metadata                     |
| `vocoder.safetensors` | Vocos mel-24kHz, PyTorch key layout (also loadable by the original `vocos` package) |


A local directory with the same two files works too: `VovoTTS.from_pretrained("/path/to/dir")`.

Pick a version with `revision=`: `vovo2-21m` (default, 110 voices), `vovo1.5-21m` (one voice, cleanest),
`vovo1-20m` (the first release, no prosody controls).

Trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (public domain, one female speaker, ~24 h)
and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) (CC-BY-4.0, 109 speakers, ~41 h).

## How it was trained

Not in one go, and no longer in thirteen minutes — that figure belongs to the first release and has been
quoted at us ever since. Each version starts from the previous one's weights, so the cost accumulates:

| | what changed | steps | time |
| --- | --- | --- | --- |
| | one voice, long run | 30,000 | 63 min |
| | a discriminator on the decoder, for sharper detail | 4,000 | 18 min |
| | 109 more voices added to the mix | 30,000 | 60 min |
| **Vovo2-21M** | the original voice weighted back up | 8,000 | 37 min |
| | | **72,000** | **≈ 3 h** |

Three hours on one laptop, and the extra voices are what bought the biggest single improvement in the
model's pitch: the error fell by a third once it had heard 109 other people speak, which no amount of
further training on one voice had managed. Data, not steps.

## Limitations

English only, and no streaming support just yet.  Out-of-vocabulary words are spelled by rules, there's no neural G2P, so unusual names come out odd.  Intelligible but not studio-clean: 10 % word error rate for Vovo2 on my test set with Apple's ASR, 3.7 % for the single-voice Vovo1.5 — the 110 voices cost real accuracy, and three hours of training on a laptop is three hours of training on a laptop.  Pitch is predicted from the text alone, so it guesses a reading; use the prosody controls or SSML when it guesses wrong.

## License

MIT for the code and the weights. The lexicon is ipa-dict (MIT); the vocoder is derived from Vocos (MIT);
the training data is LJSpeech (public domain).