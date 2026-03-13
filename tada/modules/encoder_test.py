import os

import torch
import torchaudio

from ..utils.test_utils import get_sample_dir, skip_if_hub_model_unavailable
from .encoder import Encoder


def test_encoder(model_name_or_path: str = "HumeAI/tada-codec", device: str = "cuda"):
    skip_if_hub_model_unavailable(model_name_or_path, subfolder="encoder")
    encoder = Encoder.from_pretrained(model_name_or_path, subfolder="encoder").to(device)
    audio, sample_rate = torchaudio.load(os.path.join(get_sample_dir(), "ljspeech.wav"))
    audio = audio.to(device)
    text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
    out = encoder(
        audio, text=[text], audio_length=torch.tensor([audio.shape[1]], device=device), sample_rate=sample_rate
    )
    assert out.encoded_expanded.shape == (1, 381, 512)


def test_token_values_are_local(model_name_or_path: str = "HumeAI/tada-codec", device: str = "cuda"):
    """Test that token values are local.
    A change in one audio sample should only affect the immediately preceding and following token values.
    """
    skip_if_hub_model_unavailable(model_name_or_path, subfolder="encoder")
    encoder = Encoder.from_pretrained(model_name_or_path, subfolder="encoder").to(device)
    encoder.eval()
    torch.manual_seed(42)

    audio1 = torch.randn(1, 24000 * 10).to(device)
    audio2 = audio1.clone()
    audio2[:, int(24000 * 4.5)] *= 10000000

    text_tokens = torch.tensor([[1, 2, 3]], device=device)
    text_token_len = torch.tensor([3], device=device)
    token_positions = torch.tensor([[50, 100, 150, 200, 250, 300, 350, 400, 450]], device=device)
    token_masks = torch.zeros(1, token_positions.max(), device=device).long()
    token_masks[0][token_positions[0] - 1] = 1
    out1, out2 = [
        encoder(
            audio,
            text_tokens=text_tokens,
            text_token_len=text_token_len,
            token_positions=token_positions,
            token_masks=token_masks,
            sample=False,
        )
        for audio in [audio1, audio2]
    ]

    token_vectors1 = out1.encoded_expanded[0, token_positions[0] - 1]
    token_vectors2 = out2.encoded_expanded[0, token_positions[0] - 1]
    diff = (token_vectors1 - token_vectors2).abs().mean(-1)

    assert torch.all(diff[[0, 1, 2, 5, 6, 7, 8]] < 1e-3)
    assert torch.all(diff[[3, 4]] > 1e-3)
