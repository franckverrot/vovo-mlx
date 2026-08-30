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
short_description: A from-scratch TTS voice (Swift + Metal), served via MLX
models:
- franckverrot/vovo
---

# Vovo

Type a sentence, hear Vovo — a 20 M-parameter English text-to-speech model written from scratch in Swift with
hand-written Metal kernels and trained in 13 minutes on an M2 Max. This Space runs the Python/MLX port
[`vovo-mlx`](https://github.com/franckverrot/vovo-mlx) on CPU; weights are at
[`franckverrot/vovo`](https://huggingface.co/franckverrot/vovo).
