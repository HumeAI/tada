import os

import torch

from tada.modules.encoder import Encoder, EncoderConfig
from tada.utils.test_utils import get_weight_dir


def test_convert_checkpoint():
    encoder = Encoder(EncoderConfig())
    ckpt_path = os.path.join(get_weight_dir(), "tokenization.ckpt")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    encoder.to(torch.bfloat16)
    encoder.save_pretrained(os.path.join(get_weight_dir(), "uploads", "codec", "encoder"))

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=os.path.join(get_weight_dir(), "uploads", "codec"), repo_id="HumeAI/tada-codec", repo_type="model"
    )
