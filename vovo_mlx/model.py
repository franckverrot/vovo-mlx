"""The Vovo acoustic model in MLX: phone encoder (conv prenet + RoPE transformer) → mel prior μ and
durations → flow-matching DiT decoder. Mirrors `Sources/VovoModel/{TextEncoder,Decoder,VovoTTS}.swift`;
weight layouts are converted in `load_checkpoint`."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig

MEL_FLOOR = -16.118095  # log(1e-7): what the vocoder was trained on; nothing below it is meaningful


# --- building blocks -------------------------------------------------------------------------------

def rope(x: mx.array, base: float = 10000.0) -> mx.array:
    """Rotary embedding on [B, T, H, D] with half-rotation pairing (i, i + D/2)."""
    B, T, H, D = x.shape
    half = D // 2
    i = mx.arange(half, dtype=mx.float32)
    freq = base ** (-2.0 * i / D)                             # [half]
    ang = mx.arange(T, dtype=mx.float32)[:, None] * freq[None]  # [T, half]
    c, s = mx.cos(ang)[None, :, None, :], mx.sin(ang)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def self_attention(qkv: mx.array, heads: int) -> mx.array:
    """Multi-head self-attention from a fused projection [B, T, 3·H·D] (layout [q | k | v], head-major)."""
    B, T, C3 = qkv.shape
    D = C3 // (3 * heads)
    q, k, v = (t.reshape(B, T, heads, D) for t in mx.split(qkv, 3, axis=-1))
    q, k = rope(q), rope(k)
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))    # [B, H, T, D]
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))
    return out.transpose(0, 2, 1, 3).reshape(B, T, heads * D)


class Conv1d(nn.Module):
    """'Same' channels-last convolution; weight stored MLX-style [Cout, K, Cin]."""

    def __init__(self, cin: int, cout: int, k: int) -> None:
        super().__init__()
        self.weight = mx.zeros((cout, k, cin))
        self.bias = mx.zeros((cout,))
        self.padding = (k - 1) // 2

    def __call__(self, x: mx.array) -> mx.array:
        return mx.conv1d(x, self.weight, padding=self.padding) + self.bias


class Linear(nn.Module):
    """y = x @ W + b with W stored [in, out] (the Vovo checkpoint layout)."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.weight = mx.zeros((cin, cout))
        self.bias = mx.zeros((cout,))

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = mx.ones((dim,))
        self.beta = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.gamma, self.beta, self.eps)


class TransformerBlock(nn.Module):
    """Pre-norm block: RoPE self-attention + GELU MLP."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = LayerNorm(dim), LayerNorm(dim)
        self.qkv, self.proj = Linear(dim, 3 * dim), Linear(dim, dim)
        self.fc1, self.fc2 = Linear(dim, 4 * dim), Linear(4 * dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.proj(self_attention(self.qkv(self.ln1(x)), self.heads))
        return x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))


# --- encoder ----------------------------------------------------------------------------------------

class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.encDim)
        self.spkProj = Linear(cfg.spkDim, cfg.encDim)
        self.prenetConvs = [Conv1d(cfg.encDim, cfg.encDim, cfg.prenetKernel) for _ in range(cfg.prenetLayers)]
        self.prenetNorms = [LayerNorm(cfg.encDim) for _ in range(cfg.prenetLayers)]
        self.blocks = [TransformerBlock(cfg.encDim, cfg.encHeads) for _ in range(cfg.encLayers)]
        self.outNorm = LayerNorm(cfg.encDim)
        self.projMu = Linear(cfg.encDim, cfg.nMels)
        self.durConvs = [Conv1d(cfg.encDim, cfg.durHidden, cfg.durKernel), Conv1d(cfg.durHidden, cfg.durHidden, cfg.durKernel)]
        self.durNorms = [LayerNorm(cfg.durHidden), LayerNorm(cfg.durHidden)]
        self.durOut = Linear(cfg.durHidden, 1)
        if cfg.varianceAdaptor:
            self.pitchConvs = [Conv1d(cfg.encDim, cfg.durHidden, cfg.durKernel), Conv1d(cfg.durHidden, cfg.durHidden, cfg.durKernel)]
            self.pitchNorms = [LayerNorm(cfg.durHidden), LayerNorm(cfg.durHidden)]
            self.pitchOut = Linear(cfg.durHidden, 1)
            self.pitchEmb = Linear(1, cfg.nMels)
            self.energyConvs = [Conv1d(cfg.encDim, cfg.durHidden, cfg.durKernel), Conv1d(cfg.durHidden, cfg.durHidden, cfg.durKernel)]
            self.energyNorms = [LayerNorm(cfg.durHidden), LayerNorm(cfg.durHidden)]
            self.energyOut = Linear(cfg.durHidden, 1)
            self.energyEmb = Linear(1, cfg.nMels)

    def prosody_embed(self, mu: mx.array, pitch: mx.array, energy: mx.array) -> mx.array:
        """μ + embeddings of the per-phone pitch/energy values (normalized units), [B, N]."""
        return mu + self.pitchEmb(pitch[..., None]) + self.energyEmb(energy[..., None])

    def __call__(self, phones: mx.array, spk: mx.array) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        """phones [B, N] int32, spk [B, spkDim] → (μ, log-durations, pitch, energy); the last two are
        None without the variance adaptor."""
        x = self.emb(phones) * math.sqrt(self.cfg.encDim) + self.spkProj(spk)[:, None, :]
        h = x
        for conv, norm in zip(self.prenetConvs, self.prenetNorms):
            h = nn.relu(norm(conv(h)))
        x = x + h
        for block in self.blocks:
            x = block(x)
        x = self.outNorm(x)
        mu = self.projMu(x)
        d = x
        for conv, norm in zip(self.durConvs, self.durNorms):
            d = norm(nn.relu(conv(d)))
        logw = self.durOut(d)[..., 0]
        if not self.cfg.varianceAdaptor:
            return mu, logw, None, None

        def predict(convs, norms, out):
            v = x
            for conv, norm in zip(convs, norms):
                v = norm(nn.relu(conv(v)))
            return out(v)[..., 0]

        return mu, logw, predict(self.pitchConvs, self.pitchNorms, self.pitchOut), predict(self.energyConvs, self.energyNorms, self.energyOut)


# --- decoder ----------------------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """DiT block with adaLN-zero modulation from the conditioning vector (flow time + speaker)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.dim, self.heads = dim, heads
        self.qkv, self.proj = Linear(dim, 3 * dim), Linear(dim, dim)
        self.fc1, self.fc2 = Linear(dim, 4 * dim), Linear(4 * dim, dim)
        self.adaLN = Linear(dim, 6 * dim)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        mod = self.adaLN(cond)[:, None, :]
        shift1, scale1, gate1, shift2, scale2, gate2 = mx.split(mod, 6, axis=-1)
        h = mx.fast.layer_norm(x, None, None, 1e-5) * (1 + scale1) + shift1
        x = x + gate1 * self.proj(self_attention(self.qkv(h), self.heads))
        h = mx.fast.layer_norm(x, None, None, 1e-5) * (1 + scale2) + shift2
        return x + gate2 * self.fc2(nn.gelu(self.fc1(h)))


def time_embedding(t: float, dim: int) -> mx.array:
    """Sinusoidal embedding of the flow time (t·1000), sin in the first half, cos in the second."""
    half = dim // 2
    i = mx.arange(half, dtype=mx.float32)
    freq = mx.exp(-math.log(10000.0) * i / (half - 1))
    return mx.concatenate([mx.sin(t * 1000 * freq), mx.cos(t * 1000 * freq)])[None]


class FlowDecoder(nn.Module):
    """Velocity estimator v(x_t, t | μ_up, speaker)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inProj = Linear(2 * cfg.nMels, cfg.decDim)
        self.time1, self.time2 = Linear(cfg.decDim, cfg.decDim), Linear(cfg.decDim, cfg.decDim)
        self.spkProj = Linear(cfg.spkDim, cfg.decDim)
        self.blocks = [DiTBlock(cfg.decDim, cfg.decHeads) for _ in range(cfg.decLayers)]
        self.outProj = Linear(cfg.decDim, cfg.nMels)

    def __call__(self, xt: mx.array, mu: mx.array, t_emb: mx.array, spk: mx.array) -> mx.array:
        cond = nn.silu(self.time2(nn.silu(self.time1(t_emb))) + self.spkProj(spk))
        x = self.inProj(mx.concatenate([xt, mu], axis=-1))
        cfg = self.cfg
        skip = None
        T = xt.shape[1]
        for i, block in enumerate(self.blocks):
            if cfg.decDownTo > cfg.decDownFrom and i == cfg.decDownFrom:
                skip, x = x, _downsample2(x)
            x = block(x, cond)
            if cfg.decDownTo > cfg.decDownFrom and i == cfg.decDownTo - 1:
                x = _upsample2(x, T) + skip
        x = mx.fast.layer_norm(x, None, None, 1e-5)
        return self.outProj(x)


def _downsample2(x: mx.array) -> mx.array:
    B, T, C = x.shape
    if T % 2:
        x = mx.concatenate([x, x[:, -1:, :]], axis=1)
    return x.reshape(B, -1, 2, C).mean(axis=2)


def _upsample2(x: mx.array, T: int) -> mx.array:
    return mx.repeat(x, 2, axis=1)[:, :T, :]


# --- the model ---------------------------------------------------------------------------------------

@dataclass
class PhoneControl:
    """Per-phone synthesis control (what SSML tags steer). `duration_frames` overrides the predicted
    duration outright, which is how `<break time="…"/>` gets its exact length."""
    pitch_shift: float = 0.0
    pitch_scale: float = 1.0
    energy_shift: float = 0.0
    speed: float = 1.0
    duration_frames: int | None = None


@dataclass
class Synthesis:
    mel: mx.array          # [T, nMels] decoded log-mel
    prior: mx.array        # [T, nMels] μ upsampled (the encoder's blurry estimate)
    durations: list[int]   # frames per phone
    x0: mx.array           # the noise the ODE started from
    pitch: mx.array = None      # [N] per-phone pitch after the knobs (variance adaptor)
    energy: mx.array = None     # [N] per-phone energy after the knobs


def time_grid(steps: int, sway: float) -> list[float]:
    """Uniform ODE grid, or F5-TTS "sway" (s < 0 packs steps near t = 0)."""
    out = []
    for i in range(steps + 1):
        u = i / steps
        out.append(u if sway == 0 else u + sway * (math.cos(math.pi / 2 * u) - 1 + u))
    return out


class VovoModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg)
        self.decoder = FlowDecoder(cfg)
        self.spkEmb = nn.Embedding(cfg.nSpeakers, cfg.spkDim)

    def synthesize(self, phones: list[int], *, speaker: int = 0, steps: int = 16, guidance: float = 2.0,
                   temperature: float = 0.667, speed: float = 1.0, sway: float = 0.0, midpoint: bool = False,
                   noise: mx.array | None = None, pitch_shift: float = 0.0, pitch_scale: float = 1.0,
                   energy_shift: float = 0.0, control: list["PhoneControl"] | None = None) -> Synthesis:
        """Phone ids → log-mel. `guidance` is classifier-free guidance (1 = off); `speed` > 1 talks faster."""
        cfg = self.cfg
        ph = mx.array(phones, dtype=mx.int32)[None]
        spk = self.spkEmb(mx.array([speaker], dtype=mx.int32))
        mu, logw, pitch_pred, energy_pred = self.encoder(ph, spk)
        pitch_out, energy_out = mx.array([]), mx.array([])
        if cfg.varianceAdaptor and pitch_pred is not None:
            f0_std = max(cfg.f0Std[speaker], 1e-3) if speaker < len(cfg.f0Std) else 1.0
            e_std = max(cfg.energyStd[speaker], 1e-3) if speaker < len(cfg.energyStd) else 1.0
            p, e = pitch_pred, energy_pred
            mean = p.mean()
            st, db = math.log(2) / 12 / f0_std, math.log(10) / 20 / e_std
            if control:
                # Per-phone: the scalar knobs compose with each span's values.
                scales = mx.array([pitch_scale * c.pitch_scale for c in control])[None]
                shifts = mx.array([(pitch_shift + c.pitch_shift) * st for c in control])[None]
                energies = mx.array([(energy_shift + c.energy_shift) * db for c in control])[None]
                p = mean + (p - mean) * scales + shifts
                e = e + energies
            else:
                if pitch_scale != 1:
                    p = mean + (p - mean) * pitch_scale
                if pitch_shift != 0:
                    p = p + pitch_shift * st
                if energy_shift != 0:
                    e = e + energy_shift * db
            pitch_out, energy_out = p[0], e[0]
            mu = self.encoder.prosody_embed(mu, p, e)
        if control is not None and len(control) != len(phones):
            raise ValueError("control must have one entry per phone")
        w = []
        for i, v in enumerate(logw[0].tolist()):
            c = control[i] if control else None
            if c is not None and c.duration_frames is not None:
                w.append(max(1, c.duration_frames))
            else:
                w.append(max(1, math.ceil(math.exp(float(v)) / (speed * (c.speed if c else 1.0)))))
        T = sum(w)
        idx = mx.array([n for n, count in enumerate(w) for _ in range(count)], dtype=mx.int32)
        mu_up = mu[:, idx, :]                                     # [1, T, nMels]
        x = noise[None] if noise is not None else mx.random.normal((1, T, cfg.nMels)) * temperature
        x0 = x[0]
        grid = time_grid(steps, sway)
        mu_zero = mx.zeros_like(mu_up) if guidance != 1 else None

        def velocity(xt: mx.array, t: float) -> mx.array:
            te = time_embedding(t, cfg.decDim)
            vc = self.decoder(xt, mu_up, te, spk)
            if mu_zero is None:
                return vc
            vu = self.decoder(xt, mu_zero, te, spk)
            return vu + (vc - vu) * guidance

        for i in range(steps):
            t, dt = grid[i], grid[i + 1] - grid[i]
            v = velocity(x, t)
            if midpoint:
                x = x + velocity(x + v * (dt / 2), t + dt / 2) * dt
            else:
                x = x + v * dt
        mel = mx.maximum(x[0], MEL_FLOOR)
        mx.eval(mel)
        return Synthesis(mel=mel, prior=mx.maximum(mu_up[0], MEL_FLOOR), durations=w, x0=x0, pitch=pitch_out, energy=energy_out)


# --- checkpoint loading -----------------------------------------------------------------------------

def load_checkpoint(path: str, prefer_ema: bool = True) -> VovoModel:
    """Load a Vovo safetensors checkpoint (training or exported). EMA weights are preferred when present."""
    tensors, meta = _load_safetensors(path)
    cfg = ModelConfig.from_json(meta["config"])
    model = VovoModel(cfg)
    weights = {k: v for k, v in tensors.items() if not k.startswith(("opt.", "ema."))}
    if prefer_ema:
        for k, v in tensors.items():
            if k.startswith("ema."):
                weights[k[4:]] = v
    converted = []
    for name, value in weights.items():
        if ".prenetConvs." in name or ".durConvs." in name or ".pitchConvs." in name or ".energyConvs." in name:
            if name.endswith(".weight"):
                value = value.transpose(2, 0, 1)                  # [K, Cin, Cout] → [Cout, K, Cin]
        converted.append((name, value))
    model.load_weights(converted, strict=True)
    mx.eval(model.parameters())
    return model


def _load_safetensors(path: str) -> tuple[dict[str, mx.array], dict[str, str]]:
    tensors, meta = mx.load(path, return_metadata=True)
    return tensors, meta


def plan_ssml(markup: str, g2p, frame_rate: float = 93.75) -> tuple[list[int], list[PhoneControl], list]:
    """SSML → (phone ids, per-phone control, spans). Each span is phonemized on its own so its prosody
    applies to exactly the phones it produced; a `<break>` becomes the pause token the model was trained
    on, with its duration pinned. Mirrors `VovoTTS.plan(ssml:g2p:)` in Swift."""
    from .text import ssml as ssml_mod
    from .text.phones import ID_OF, encode

    spans = ssml_mod.parse(markup)
    punctuation_ids = {ID_OF[p] for p in ssml_mod.PUNCTUATION if p in ID_OF}
    phones: list[int] = []
    control: list[PhoneControl] = []
    for span in spans:
        if isinstance(span, ssml_mod.Pause):
            if span.milliseconds <= 0:
                continue
            frames = max(1, round(span.milliseconds / 1000 * frame_rate))
            if phones and phones[-1] in punctuation_ids:
                control[-1].duration_frames = frames
            else:
                phones.append(ID_OF[","])
                control.append(PhoneControl(duration_frames=frames))
            continue
        tokens = g2p.phonemize(span.text) or g2p.punctuation_only(span.text)
        if not tokens:
            continue
        starts_with_punctuation = tokens[0] in ssml_mod.PUNCTUATION
        if phones and not starts_with_punctuation and phones[-1] != ID_OF[" "] and tokens[0] != " ":
            phones.append(ID_OF[" "])
            control.append(PhoneControl())
        ids = encode(tokens)
        phones += ids
        control += [PhoneControl(span.control.pitch_shift, span.control.pitch_scale,
                                 span.control.energy_shift, span.control.speed) for _ in ids]
    return phones, control, spans
