"""Vocos (mel-24khz) neural vocoder in MLX: ConvNeXt backbone + iSTFT head. Loads the PyTorch-layout
safetensors of `charactr/vocos-mel-24khz` (MIT) and Vovo's fine-tuned exports of it."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import VocosConfig


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate)
        self.pwconv2 = nn.Linear(intermediate, dim)
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pwconv2(nn.gelu(self.pwconv1(h)))
        return x + h * self.gamma


class Vocos(nn.Module):
    def __init__(self, cfg: VocosConfig = VocosConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Conv1d(cfg.n_mels, cfg.dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.convnext = [ConvNeXtBlock(cfg.dim, cfg.intermediate) for _ in range(cfg.layers)]
        self.final_layer_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.out = nn.Linear(cfg.dim, cfg.n_fft + 2)
        n = cfg.n_fft
        self._window = 0.5 - 0.5 * mx.cos(2 * mx.pi * mx.arange(n) / n)   # periodic Hann

    def __call__(self, log_mel: mx.array) -> mx.array:
        """log-mel [T, n_mels] → waveform [(T-1)·hop] at 24 kHz."""
        x = self.embed(log_mel[None])
        x = self.norm(x)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x)
        out = self.out(x)[0]                                     # [T, n_fft + 2]
        half = self.cfg.n_fft // 2 + 1
        mag = mx.minimum(mx.exp(out[:, :half]), 100.0)
        phase = out[:, half:]
        return self.istft(mag * mx.cos(phase), mag * mx.sin(phase))

    def istft(self, re: mx.array, im: mx.array) -> mx.array:
        """Inverse STFT with center=True semantics: frames [T, n_fft/2+1] → (T-1)·hop samples."""
        n, hop = self.cfg.n_fft, self.cfg.hop
        T = re.shape[0]
        spec = re + 1j * im
        frames = mx.fft.irfft(spec, n=n, axis=-1) * self._window          # [T, n]
        total = (T - 1) * hop + n
        # overlap-add via a scatter: sample s of frame t lands at t·hop + s
        idx = (mx.arange(T)[:, None] * hop + mx.arange(n)[None, :]).reshape(-1)
        out = mx.zeros((total,)).at[idx].add(frames.reshape(-1))
        norm = mx.zeros((total,)).at[idx].add(mx.broadcast_to(self._window * self._window, (T, n)).reshape(-1))
        wav = mx.where(norm > 1e-6, out / mx.maximum(norm, 1e-6), 0.0)
        half = n // 2
        return wav[half : half + total - n]


def load_vocos(path: str, cfg: VocosConfig = VocosConfig()) -> Vocos:
    """Load a PyTorch-layout Vocos checkpoint (`backbone.*`, `head.*` keys)."""
    tensors = mx.load(path)
    model = Vocos(cfg)
    weights = []
    for k, v in tensors.items():
        if k.startswith("feature_extractor.") or k.startswith("head.istft."):
            continue  # mel filterbank / the iSTFT window: buffers, recomputed here
        name = k.replace("backbone.", "").replace("head.", "")
        if name.endswith("dwconv.weight"):
            v = v.transpose(0, 2, 1)                # torch [C, 1, 7] → mlx [C, 7, 1]
        elif name == "embed.weight":
            v = v.transpose(0, 2, 1)                # torch [out, in, k] → mlx [out, k, in]
        weights.append((name, v))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model
