"""Minimal example: text → WAV."""

from vovo_mlx import VovoTTS, SAMPLE_RATE
from vovo_mlx.audio import write_wav

tts = VovoTTS.from_pretrained()  # downloads and caches the weights from the Hugging Face Hub
wav = tts.say("The quick brown fox jumps over the lazy dog.", seed=0)
write_wav("fox.wav", wav, SAMPLE_RATE)
print(f"wrote fox.wav: {len(wav) / SAMPLE_RATE:.2f} s")
