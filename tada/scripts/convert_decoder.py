import os

import torch

from tada.modules.decoder import Decoder, DecoderConfig
from tada.utils.test_utils import get_weight_dir

if __name__ == "__main__":
    decoder = Decoder(DecoderConfig())
    ckpt_path = "/mnt/weka/models/outputs/cont_tokens/v2_text_tok_512_sem_loss_gaussian_combined_causal_v2/epoch=0-step=200000.ckpt"
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()
    decoder.to(torch.bfloat16)
    decoder.save_pretrained(os.path.join(get_weight_dir(), "uploads", "codec", "decoder"))

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=os.path.join(get_weight_dir(), "uploads", "codec"), repo_id="HumeAI/tada-codec", repo_type="model"
    )
