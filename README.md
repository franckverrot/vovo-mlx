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

`tts.say(text, steps=16, guidance=2.0, temperature=0.667, speed=1.0, sway=0.0, midpoint=False, seed=None, pitch_shift=0.0, pitch_scale=1.0, energy_shift=0.0)`


| argument      | what it does                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------- |
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
Trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (public domain, one female speaker, ~24 h).

## Limitations

Currently single voice, English only, and no streaming support just yet.  Out-of-vocabulary words are spelled by rules, there's no neural G2P.  The voice is intelligible (2 % word error rate on Vovo's test set with Apple's ASR) but far from being studio-clean: I let it train for 13 minutes.

## License

MIT for the code and the weights. The lexicon is ipa-dict (MIT); the vocoder is derived from Vocos (MIT);
the training data is LJSpeech (public domain).