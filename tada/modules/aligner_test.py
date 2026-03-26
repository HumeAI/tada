import os

import pytest
import torch
import torchaudio

from ..utils.test_utils import get_sample_dir, skip_if_hub_model_unavailable
from .aligner import Aligner


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        # os.path.join(get_weight_dir(), "uploads")
        "HumeAI/tada-codec"
    ],
)
def test_aligner(model_name_or_path: str):
    device = "cpu"
    skip_if_hub_model_unavailable(model_name_or_path, subfolder="aligner")
    aligner = Aligner.from_pretrained(model_name_or_path, subfolder="aligner").to(device)
    audio, sample_rate = torchaudio.load(os.path.join(get_sample_dir(), "ljspeech.wav"))
    audio = audio.to(device)
    text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
    token_positions, _ = aligner(
        audio, text=[text], audio_length=torch.tensor([audio.shape[1]], device=device), sample_rate=sample_rate
    )

    text_tokens = aligner.tokenizer.convert_ids_to_tokens(aligner.tokenizer.encode(text, add_special_tokens=False))
    # list of tuples of (text token, position)
    assert list(
        zip(
            text_tokens,
            token_positions[0].cpu().numpy().tolist(),
        )
    ) == [
        ("The", 2),
        ("Ġexamination", 31),
        ("Ġand", 47),
        ("Ġtestimony", 58),
        ("Ġof", 85),
        ("Ġthe", 92),
        ("Ġexperts", 114),
        (",", 149),
        ("Ġenabled", 156),
        ("Ġthe", 176),
        ("Ġcommission", 189),
        ("Ġto", 208),
        ("Ġconclude", 228),
        ("Ġthat", 266),
        ("Ġfive", 280),
        ("Ġshots", 298),
        ("Ġmay", 316),
        ("Ġhave", 323),
        ("Ġbeen", 332),
        ("Ġfired", 347),
        (".", 377),
    ]
