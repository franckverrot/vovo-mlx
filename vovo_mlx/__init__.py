"""vovo-mlx — inference for Vovo, a from-scratch English TTS model, on Apple silicon with MLX.

    from vovo_mlx import VovoTTS
    tts = VovoTTS.from_pretrained()            # weights from the Hugging Face Hub
    wav = tts.say("Hello from Vovo.")          # numpy float32 at 24 kHz
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from . import hub
from .config import ModelConfig, VocosConfig
from .model import PhoneControl, Synthesis, VovoModel, load_checkpoint, plan_ssml
from .text import G2P
from .vocos import Vocos, load_vocos

__version__ = "0.3.0"
SAMPLE_RATE = 24000


class VovoTTS:
    """Text → waveform: G2P, the acoustic model and the vocoder in one object."""

    def __init__(self, model: VovoModel, vocoder: Vocos, g2p: G2P | None = None) -> None:
        self.model = model
        self.vocoder = vocoder
        self.g2p = g2p or G2P()

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = hub.DEFAULT_REPO, *, model_file: str = hub.MODEL_FILE,
                        vocoder_file: str = hub.VOCODER_FILE, revision: str | None = None) -> "VovoTTS":
        """Load from a Hub repo id (downloaded and cached) or a local directory containing the two files."""
        model = load_checkpoint(hub.resolve(repo_or_dir, model_file, revision))
        vocoder = load_vocos(hub.resolve(repo_or_dir, vocoder_file, revision))
        return cls(model, vocoder)

    @property
    def config(self) -> ModelConfig:
        return self.model.cfg

    @property
    def voices(self) -> list[str]:
        """Names of the voices this checkpoint carries, in speaker-id order. Empty on single-voice models
        and on checkpoints published before names were recorded."""
        return list(self.model.cfg.speakerNames)

    def speaker_id(self, speaker: int | str) -> int:
        """Accepts an id or a voice name ('p226'). Raises rather than quietly rendering the wrong voice."""
        cfg = self.model.cfg
        if isinstance(speaker, int):
            if not 0 <= speaker < cfg.nSpeakers:
                raise ValueError(f"speaker {speaker} is out of range: ids 0…{cfg.nSpeakers - 1}")
            return speaker
        try:
            return cfg.speakerNames.index(speaker)
        except ValueError:
            if not cfg.speakerNames:
                raise ValueError(f"this checkpoint has no voice names — use an id (0…{cfg.nSpeakers - 1})") from None
            raise ValueError(f"no voice named {speaker!r}. Known: {' '.join(cfg.speakerNames)}") from None

    def phonemize(self, text: str) -> list[str]:
        """The phone tokens the model will see (normalized text → IPA)."""
        return self.g2p.phonemize(text)

    def synthesize(self, text: str, *, speaker: int | str = 0, **kwargs) -> Synthesis:
        """Text → log-mel. SSML markup (`<speak>`, `<prosody>`, `<emphasis>`, `<break>`) is detected and
        parsed automatically; see `VovoModel.synthesize` for the sampler arguments."""
        from .text import ssml as ssml_mod

        sid = self.speaker_id(speaker)
        if ssml_mod.looks_like_markup(text):
            phones, control, _ = plan_ssml(text, self.g2p)
            return self.model.synthesize(phones, speaker=sid, control=control, **kwargs)
        return self.model.synthesize(self.g2p.encode(text), speaker=sid, **kwargs)

    def vocode(self, log_mel: mx.array) -> np.ndarray:
        wav = self.vocoder(log_mel)
        mx.eval(wav)
        return np.array(wav, dtype=np.float32)

    def say(self, text: str, *, speaker: int | str = 0, steps: int = 16, guidance: float = 2.0,
            temperature: float = 0.667, speed: float = 1.0, sway: float = 0.0, midpoint: bool = False,
            seed: int | None = None, pitch_shift: float = 0.0, pitch_scale: float = 1.0,
            energy_shift: float = 0.0) -> np.ndarray:
        """Text → float32 waveform at 24 kHz. With a variance-adaptor model, `pitch_shift` (semitones),
        `pitch_scale` (contour variance) and `energy_shift` (dB) reshape the prosody; SSML markup in `text`
        (`<prosody>`, `<emphasis>`, `<break>`) steers it per span and composes with these knobs."""
        if seed is not None:
            mx.random.seed(seed)
        s = self.synthesize(text, speaker=speaker, steps=steps, guidance=guidance, temperature=temperature, speed=speed,
                            sway=sway, midpoint=midpoint,
                            pitch_shift=pitch_shift, pitch_scale=pitch_scale, energy_shift=energy_shift)
        return self.vocode(s.mel)


__all__ = ["VovoTTS", "VovoModel", "Vocos", "Synthesis", "PhoneControl", "plan_ssml", "ModelConfig", "VocosConfig", "G2P", "load_checkpoint", "load_vocos", "SAMPLE_RATE", "__version__"]
