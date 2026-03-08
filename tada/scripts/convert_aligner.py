import os

import torch

from tada.modules.aligner import Aligner, AlignerConfig
from tada.utils.test_utils import get_weight_dir


def test_convert_checkpoint():
    aligner = Aligner(AlignerConfig())
    ckpt_path = os.path.join(get_weight_dir(), "aligner.ckpt")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    aligner.load_state_dict(state_dict, strict=False)
    aligner.eval()
    aligner.to(torch.bfloat16)
    aligner.save_pretrained(os.path.join(get_weight_dir(), "uploads", "codec", "aligner"))

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=os.path.join(get_weight_dir(), "uploads", "codec"), repo_id="HumeAI/tada-codec", repo_type="model"
    )
