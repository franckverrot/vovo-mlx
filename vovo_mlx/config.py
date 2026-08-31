"""Model hyper-parameters, read from the `config` metadata of a Vovo checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields


@dataclass
class ModelConfig:
    vocab: int = 67
    nMels: int = 100
    encDim: int = 192
    encLayers: int = 6
    encHeads: int = 2
    prenetKernel: int = 5
    prenetLayers: int = 3
    durHidden: int = 256
    durKernel: int = 3
    decDim: int = 384
    decLayers: int = 6
    decHeads: int = 6
    nSpeakers: int = 1
    spkDim: int = 64
    decDownFrom: int = 0
    decDownTo: int = 0
    varianceAdaptor: bool = False
    f0Mean: list[float] = field(default_factory=list)
    f0Std: list[float] = field(default_factory=list)
    energyMean: list[float] = field(default_factory=list)
    energyStd: list[float] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        raw = json.loads(text)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass(frozen=True)
class VocosConfig:
    n_mels: int = 100
    dim: int = 512
    intermediate: int = 1536
    layers: int = 8
    n_fft: int = 1024
    hop: int = 256
    sample_rate: int = 24000
