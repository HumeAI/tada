import os

import pytest
import torch
import torchaudio

from ..utils.test_utils import get_sample_dir, skip_if_hub_model_unavailable
from .decoder import Decoder, DecoderConfig
from .encoder import Encoder, EncoderConfig


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        # None,
        # os.path.join(get_weight_dir(), "uploads")
        "HumeAI/tada-codec"
    ],
)
def test_decoder(model_name_or_path: str | None):
    device = "cpu"
    if model_name_or_path is None:
        encoder = Encoder(EncoderConfig())
        decoder = Decoder(DecoderConfig())
    else:
        skip_if_hub_model_unavailable(model_name_or_path, subfolder="encoder")
        skip_if_hub_model_unavailable(model_name_or_path, subfolder="decoder")
        encoder = Encoder.from_pretrained(model_name_or_path, subfolder="encoder").to(device)
        decoder = Decoder.from_pretrained(model_name_or_path, subfolder="decoder").to(device)
    audio, sample_rate = torchaudio.load(os.path.join(get_sample_dir(), "ljspeech.wav"))
    audio = audio.to(device)
    text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
    out = encoder(
        audio, text=[text], audio_length=torch.tensor([audio.shape[1]], device=device), sample_rate=sample_rate
    )
    out = decoder(out.encoded_expanded, out.text_emb_expanded, out.token_masks)
    assert out.shape == (1, 1, 182874)
    torchaudio.save(os.path.join(get_sample_dir(), "ljspeech_decoded.wav"), out.detach().squeeze(1), sample_rate)
