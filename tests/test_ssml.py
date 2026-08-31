"""The Python SSML parser must produce exactly what the Swift one produces (`vovo ssml --json`)."""

import json
from pathlib import Path

import pytest

from vovo_mlx.model import PhoneControl, plan_ssml
from vovo_mlx.text import G2P, ssml

GOLDEN = json.loads((Path(__file__).parent / "golden_ssml.json").read_text())


@pytest.fixture(scope="module")
def g2p():
    return G2P()


@pytest.mark.parametrize("case", GOLDEN, ids=[c["markup"][:38] for c in GOLDEN])
def test_matches_swift(g2p, case):
    phones, control, _ = plan_ssml(case["markup"], g2p)
    assert phones == case["phones"]
    assert [round(c.pitch_shift, 5) for c in control] == [round(v, 5) for v in case["pitchShift"]]
    assert [round(c.pitch_scale, 5) for c in control] == [round(v, 5) for v in case["pitchScale"]]
    assert [round(c.energy_shift, 5) for c in control] == [round(v, 5) for v in case["energyShift"]]
    assert [round(c.speed, 5) for c in control] == [round(v, 5) for v in case["speed"]]
    assert [c.duration_frames if c.duration_frames is not None else -1 for c in control] == case["durationFrames"]


def test_nesting_and_units():
    (span,) = ssml.parse('<prosody pitch="+2st" rate="0.5"><prosody pitch="+3st" rate="2">x</prosody></prosody>')
    assert span.control.pitch_shift == 5 and span.control.speed == pytest.approx(1.0)
    assert ssml.semitones("+100%") == pytest.approx(12, abs=1e-4)
    assert ssml.range_scale("monotone") == 0
    assert ssml.rate("x-slow") == 0.5 and ssml.decibels("x-loud") == 12


def test_errors_and_detection():
    with pytest.raises(ssml.SSMLError):
        ssml.parse('<prosody pitch="+2st">never closed')
    with pytest.raises(ssml.SSMLError):
        ssml.parse("</prosody>")
    assert ssml.looks_like_markup("<speak>hi</speak>")
    assert not ssml.looks_like_markup("a < b and c > d")
