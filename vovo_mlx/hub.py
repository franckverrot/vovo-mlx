"""Weights from the Hugging Face Hub (or a local directory laid out the same way)."""

from __future__ import annotations

import os

DEFAULT_REPO = "franckverrot/vovo"
MODEL_FILE = "model.safetensors"
VOCODER_FILE = "vocoder.safetensors"


def resolve(repo_or_dir: str, filename: str, revision: str | None = None) -> str:
    """Path to `filename` inside a local directory, or downloaded from the Hub repo (cached)."""
    local = os.path.join(repo_or_dir, filename)
    if os.path.isdir(repo_or_dir):
        if not os.path.exists(local):
            raise FileNotFoundError(local)
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_or_dir, filename, revision=revision)
