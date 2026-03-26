import os

import pytest
import torch
import torchaudio

from ..utils.test_utils import get_sample_dir, skip_if_hub_model_unavailable
from .encoder import Encoder, EncoderConfig
from .tada import TadaConfig, TadaForCausalLM


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        # None,
        # os.path.join(get_weight_dir(), "uploads")
        "HumeAI/TADA"
    ],
)
def test_tada_for_causal_lm(model_name_or_path: str | None):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    if model_name_or_path is None:
        encoder = Encoder(EncoderConfig())
        config = TadaConfig(
            num_hidden_layers=1, vocab_size=128256, hidden_size=8, num_attention_heads=1, num_time_classes=8
        )
        model = TadaForCausalLM(config)
    else:
        skip_if_hub_model_unavailable(model_name_or_path, subfolder="encoder")
        skip_if_hub_model_unavailable(model_name_or_path, subfolder="llm")
        encoder = Encoder.from_pretrained("HumeAI/TADA", subfolder="encoder").to(device)
        model = TadaForCausalLM.from_pretrained(model_name_or_path, subfolder="llm").to(device)
    encoder.to(device).eval()
    model.to(device).eval()

    audio, sample_rate = torchaudio.load(os.path.join(get_sample_dir(), "ljspeech.wav"))
    audio = audio.to(device)
    prompt_text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
    prompt = encoder(
        audio, text=[prompt_text], audio_length=torch.tensor([audio.shape[1]], device=device), sample_rate=sample_rate
    )

    output = model.generate(
        prompt=prompt,
        text="Please call Stella. Ask her to bring these things with her from the store.",
    )
    torchaudio.save(
        os.path.join(get_sample_dir(), "ljspeech_generated.wav"), output.audio[0].detach().cpu().unsqueeze(0), 24000
    )
