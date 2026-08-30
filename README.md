<!-- DRAFT — Franck reviews and has the final word on this file before anything is published. -->

# vovo-mlx

Inference for **Vovo**, a small English text-to-speech model built from scratch, running on Apple silicon
with [MLX](https://github.com/ml-explore/mlx). Weights are on the Hugging Face Hub:
[`franckverrot/vovo-ljspeech`](https://huggingface.co/franckverrot/vovo-ljspeech).

Vovo itself is written in Swift with hand-written Metal kernels — its own tensor engine, autograd, optimizer,
data pipeline and trainer, no PyTorch anywhere — and trains a 20 M-parameter voice in about 13 minutes on an
M2 Max. This repository is the Python port of the *inference* path so the weights can be used from Python.
It reproduces the Swift implementation numerically (see [Parity](#parity)).

```
pip install vovo-mlx
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

- **Text front-end**: English normalizer (numbers, currency, ordinals, times, abbreviations), the
  [ipa-dict](https://github.com/open-dict-data/ipa-dict) en_US lexicon (MIT) with rule-based fallbacks for
  out-of-vocabulary words, and Vovo's fixed 67-symbol phone inventory.
- **Acoustic model** (20 M params): phone encoder (conv prenet + 6-layer RoPE transformer, d=192) → per-phone
  mel prior μ and durations → length regulation → flow-matching DiT decoder (6 layers, d=384, adaLN-zero)
  sampled with an Euler or midpoint ODE solver, optional F5-style "sway" time grid and classifier-free guidance.
- **Vocoder**: [Vocos](https://github.com/gemelo-ai/vocos) mel-24kHz (MIT), fine-tuned by Vovo on its own
  predicted mels, in MLX (ConvNeXt backbone + iSTFT head).
- Outputs 24 kHz mono. On an M2 Max: real-time factor ≈ 0.04–0.06 (a 9-second sentence in 0.3 s), 0.5 s to load.

## Sampler knobs

`tts.say(text, steps=16, guidance=2.0, temperature=0.667, speed=1.0, sway=0.0, midpoint=False, seed=None)`

| argument | what it does |
|---|---|
| `steps` | ODE steps (16 is the default; 8 with `sway=-1, midpoint=True` is faster and close) |
| `guidance` | classifier-free guidance scale; 1 = off, 2 = default (crisper, slightly brighter), 3 = brighter still |
| `temperature` | scale of the starting noise; lower = steadier, flatter |
| `speed` | > 1 talks faster (durations divided by `speed`) |
| `sway` | < 0 packs ODE steps near t = 0 where the flow bends most (try −1) |
| `midpoint` | 2nd-order solver (two velocity evaluations per step) |

Lower-level access: `tts.synthesize(text, **knobs)` returns the log-mel (`.mel`), the encoder prior
(`.prior`), the per-phone durations and the starting noise; `tts.vocode(mel)` turns any `[T, 100]` log-mel
(24 kHz, n_fft 1024, hop 256, HTK mel scale, `log(clamp(x, 1e-7))`) into a waveform; `tts.phonemize(text)`
shows the phones the model will see.

## Examples

- [`examples/say.py`](examples/say.py) — the two-line version.
- [`examples/batch.py`](examples/batch.py) — a text file to numbered WAVs, with the knobs commented.
- [`examples/inspect_mel.py`](examples/inspect_mel.py) — plot the prior μ next to the decoded mel.
- `vovo-mlx phones "Dr. Smith paid $12.50 at 4:05."` — see the normalization and phonemization.

## Weights

`VovoTTS.from_pretrained("franckverrot/vovo-ljspeech")` fetches two files (cached by `huggingface_hub`):

| file | what |
|---|---|
| `model.safetensors` | acoustic model, EMA weights, config in the safetensors metadata |
| `vocoder.safetensors` | Vocos mel-24kHz, PyTorch key layout (also loadable by the original `vocos` package) |

A local directory with the same two files works too: `VovoTTS.from_pretrained("/path/to/dir")`.
Trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (public domain, one female speaker, ~24 h).

## Parity

The port is checked against the Swift implementation, not against the ear: the Swift CLI can dump the phone
ids, the starting noise, the prior, the decoded mel and the waveform for a sentence (`vovo say --dump`), and
`tests/test_parity.py` re-runs the same phones from the same noise in MLX and compares. On three sentences
(16 Euler steps with guidance 2; 8 midpoint steps with sway −1) the prior μ agrees to 1e-5, the decoded mel
to 3e-4 and the waveform to 2e-4 (max absolute difference, log-mel units / sample amplitude). The text
front-end is checked token-for-token against the Swift `vovo g2p` output on 28 sentences
(`tests/golden_phones.json`).

```
VOVO_WEIGHTS_DIR=… VOVO_PARITY_DIR=… pytest      # parity tests skip without the dumps
```

## Limitations

Single voice, English only, no streaming yet; out-of-vocabulary words are spelled by rules, not by a neural
G2P. The voice is intelligible (2 % word error rate on Vovo's test set with on-device ASR) but not yet
studio-clean — it is a 13-minute training run. Apple silicon only (MLX).

## License

MIT for the code and the weights. The lexicon is ipa-dict (MIT); the vocoder is derived from Vocos (MIT);
the training data is LJSpeech (public domain).
