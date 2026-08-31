"""Numerical parity with the Swift reference: same phones, same starting noise → same mel and waveform.

Needs the dumps written by `vovo say --dump` and the exported weights; set VOVO_PARITY_DIR and
VOVO_WEIGHTS_DIR (see README, "Development"). Skipped otherwise.
"""

import glob
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

PARITY_DIR = os.environ.get("VOVO_PARITY_DIR")
WEIGHTS_DIR = os.environ.get("VOVO_WEIGHTS_DIR")
DUMPS = sorted(glob.glob(os.path.join(PARITY_DIR, "*.safetensors"))) if PARITY_DIR else []

pytestmark = pytest.mark.skipif(not (DUMPS and WEIGHTS_DIR), reason="set VOVO_PARITY_DIR and VOVO_WEIGHTS_DIR")


@pytest.fixture(scope="module")
def tts():
    from vovo_mlx import VovoTTS

    return VovoTTS.from_pretrained(WEIGHTS_DIR, vocoder_file="vocoder_base.safetensors")


@pytest.mark.parametrize("dump", DUMPS, ids=[os.path.basename(d) for d in DUMPS])
def test_mel_and_wav_match_swift(tts, dump):
    t, meta = mx.load(dump, return_metadata=True)
    phones = [int(v) for v in t["phones"].tolist()]
    assert tts.g2p.encode(meta["text"]) == phones, "text front-end disagrees with Swift"
    s = tts.model.synthesize(phones, steps=int(meta["steps"]), guidance=float(meta["guidance"]),
                             temperature=float(meta["temperature"]), sway=float(meta["sway"]),
                             midpoint=meta["midpoint"] == "true", speed=float(meta["speed"]), noise=t["x0"],
                             pitch_shift=float(meta.get("pitchShift", 0)), pitch_scale=float(meta.get("pitchScale", 1)),
                             energy_shift=float(meta.get("energyShift", 0)))
    assert s.durations == [int(v) for v in t["durations"].tolist()], "durations differ"
    if "pitch" in t and s.pitch is not None and s.pitch.size == t["pitch"].size and t["pitch"].size > 1:
        assert float(mx.abs(s.pitch - t["pitch"]).max()) < 1e-4, "per-phone pitch differs from Swift"
        assert float(mx.abs(s.energy - t["energy"]).max()) < 1e-4, "per-phone energy differs from Swift"
    prior_err = float(mx.abs(s.prior - t["mu_up"]).max())
    mel_err = float(mx.abs(s.mel - t["mel"]).max())
    assert prior_err < 1e-4, f"prior μ max abs diff {prior_err}"
    assert mel_err < 5e-3, f"decoded mel max abs diff {mel_err}"
    if meta.get("vocoder", "").endswith("assets/vocos/model.safetensors"):
        wav = tts.vocode(s.mel)
        ref = np.array(t["wav"])
        n = min(len(wav), len(ref))
        assert n == len(ref) == len(wav)
        err = float(np.abs(wav[:n] - ref[:n]).max())
        assert err < 5e-3, f"waveform max abs diff {err}"
