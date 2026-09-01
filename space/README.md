---
title: Vovo
emoji: 🗣️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
short_description: 110-voice TTS built from scratch in Swift + Metal
models:
- franckverrot/vovo
---

# Vovo

Type a sentence, pick one of 110 voices, hear Vovo — a 21 M-parameter English text-to-speech model written from
scratch in Swift with hand-written Metal kernels and trained on a laptop in about three hours. It predicts pitch and
energy per sound, so the delivery can be steered with sliders or with SSML (`<prosody>`, `<emphasis>`, `<break>`).
This Space runs the Python/MLX port
[`vovo-mlx`](https://github.com/franckverrot/vovo-mlx) on CPU; weights are at
[`franckverrot/vovo`](https://huggingface.co/franckverrot/vovo).
