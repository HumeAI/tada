import os

import pytest
import torch
from huggingface_hub import hf_hub_download


def get_sample_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "samples")


def get_weight_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "weights")


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def skip_if_hub_model_unavailable(repo_id: str, subfolder: str | None = None, filename: str = "config.json") -> None:
    """Skip integration tests when model artifacts cannot be fetched from Hugging Face Hub."""
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
    except Exception as exc:  # pragma: no cover - best effort network check
        pytest.skip(f"Hugging Face model '{repo_id}' is unavailable in this environment: {exc}")
