"""Startup profile for the Space: log where the time goes on this machine (read it in the Space logs)."""

from __future__ import annotations

import os
import time

import mlx.core as mx


def run(tts) -> None:
    from vovo_mlx.model import time_embedding

    print(f"[profile] mlx {mx.__version__} device {mx.default_device()} cpus {os.cpu_count()}", flush=True)
    phones = tts.g2p.encode("The quick brown fox jumps over the lazy dog.")
    t = time.time(); s = tts.model.synthesize(phones, steps=1, guidance=1.0); t_enc1 = time.time() - t
    T = s.mel.shape[0]
    mu = mx.zeros((1, T, 100)); x = mx.zeros((1, T, 100)); spk = tts.model.spkEmb(mx.array([0]))
    te = time_embedding(0.5, tts.model.cfg.decDim)
    t = time.time(); v = tts.model.decoder(x, mu, te, spk); mx.eval(v); t_dec = time.time() - t
    t = time.time(); v = tts.model.decoder(x, mu, te, spk); mx.eval(v); t_dec2 = time.time() - t
    t = time.time(); s = tts.model.synthesize(phones, steps=16, guidance=2.0); t_ode = time.time() - t
    t = time.time(); w = tts.vocoder(s.mel); mx.eval(w); t_voc = time.time() - t
    blk = tts.model.encoder.blocks[0]
    xx = mx.random.normal((1, T, 192)); t = time.time(); y = blk(xx); mx.eval(y); t_blk = time.time() - t
    a = mx.random.normal((T, 384)); b = mx.random.normal((384, 1536)); t = time.time(); c = a @ b; mx.eval(c); t_mm = time.time() - t
    q = mx.random.normal((1, 6, T, 64)); t = time.time(); o = mx.fast.scaled_dot_product_attention(q, q, q, scale=0.125); mx.eval(o); t_sdpa = time.time() - t
    g = mx.random.normal((T, 1536)); t = time.time(); o = mx.erf(g); mx.eval(o); t_erf = time.time() - t
    print(f"[profile] T={T} frames; encoder+1 step {t_enc1:.2f}s; decoder pass {t_dec:.2f}s then {t_dec2:.2f}s; "
          f"16 steps g2 {t_ode:.1f}s; vocoder {t_voc:.2f}s; enc block {t_blk:.3f}s; matmul {T}x384x1536 {t_mm * 1000:.1f} ms; "
          f"sdpa {t_sdpa * 1000:.1f} ms; erf {T}x1536 {t_erf * 1000:.1f} ms", flush=True)
