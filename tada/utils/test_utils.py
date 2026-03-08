import os

import torch


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
