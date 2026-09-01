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
short_description: From-scratch TTS (Swift + Metal), steerable prosody
models:
- franckverrot/vovo
---

# Vovo

Type a sentence, hear Vovo — a 21 M-parameter English text-to-speech model written from scratch in Swift with
hand-written Metal kernels, trained on an M2 Max. It predicts pitch and energy per sound, so the delivery can be
steered with sliders or with SSML (`<prosody>`, `<emphasis>`, `<break>`). This Space runs the Python/MLX port
[`vovo-mlx`](https://github.com/franckverrot/vovo-mlx) on CPU; weights are at
[`franckverrot/vovo`](https://huggingface.co/franckverrot/vovo).
