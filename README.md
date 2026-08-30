# vovo-mlx

Vovo is an English-only TTS model built from scratch.  This repository is the Python MLX implementation for it.

You will find weights on [HuggingFace](https://huggingface.co/franckverrot/vovo).

Vovo was written in Swift with hand-written Metal kernels.  The entirety of the training toolchain is home-made: the tensor engine is new, autograd/optimizer/etc are also custom, and the broader data pipeline and trainer as well.  The result is a compact 20M parameters voice model that was trained in about 13 minutes on an M2 Max with 96B of RAM (the training consumed way less than that).

This repository is the Python port of the *inference* path so the weights can be used from Python: it reproduces the Swift implementation numerically (see [Parity](#parity)).

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

- **Text front-end**: English normalizer, rule-based fallbacks for out-of-vocabulary words, and Vovo's fixed 67-symbol phone inventory
- **Acoustic model** (20M)
- **Vocoder**: [Vocos](https://github.com/gemelo-ai/vocos) mel-24kHz, fine-tuned by Vovo on its own predicted mels
- Outputs 24 kHz mono.



## Sampler knobs

`tts.say(text, steps=16, guidance=2.0, temperature=0.667, speed=1.0, sway=0.0, midpoint=False, seed=None)`


| argument      | what it does                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `steps`       | ODE steps (16 is the default; 8 with `sway=-1, midpoint=True` is faster and close)                    |
| `guidance`    | classifier-free guidance scale; 1 = off, 2 = default (crisper, slightly brighter), 3 = brighter still |
| `temperature` | scale of the starting noise; lower = steadier, flatter                                                |
| `speed`       | > 1 talks faster (durations divided by `speed`)                                                       |
| `sway`        | < 0 packs ODE steps near t = 0 where the flow bends most (try −1)                                     |
| `midpoint`    | 2nd-order solver (two velocity evaluations per step)                                                  |


Lower-level access: `tts.synthesize(text, **knobs)` returns the log-mel (`.mel`), the encoder prior
(`.prior`), the per-phone durations and the starting noise; `tts.vocode(mel)` turns any `[T, 100]` log-mel
(24 kHz, n_fft 1024, hop 256, HTK mel scale, `log(clamp(x, 1e-7))`) into a waveform; `tts.phonemize(text)`
shows the phones the model will see.

## Examples

- `[examples/say.py](examples/say.py)` — the two-line version.
- `[examples/batch.py](examples/batch.py)` — a text file to numbered WAVs, with the knobs commented.
- `[examples/inspect_mel.py](examples/inspect_mel.py)` — plot the prior μ next to the decoded mel.
- `vovo-mlx phones "Dr. Smith paid $12.50 at 4:05."` — see the normalization and phonemization.



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