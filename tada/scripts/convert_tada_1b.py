import os
import subprocess
from pathlib import Path

import torch
from huggingface_hub import HfApi

from tada.modules.tada import TadaConfig, TadaForCausalLM
from tada.utils.test_utils import get_weight_dir


def consolidate_checkpoint(ckpt_path):
    """
    Consolidate an unconsolidated FSDP checkpoint using Lightning's utility.

    Args:
        ckpt_path: Path to the unconsolidated checkpoint directory

    Returns:
        Path to the consolidated checkpoint file
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.is_dir():
        print(f"{ckpt_path} is not a directory, assuming it's already consolidated")
        return ckpt_path

    # Check if consolidated version already exists
    consolidated_path = Path(str(ckpt_path) + ".consolidated")
    if consolidated_path.exists():
        print(f"Found existing consolidated checkpoint at {consolidated_path}")
        return consolidated_path

    print(f"Consolidating FSDP checkpoint from {ckpt_path}")
    print("This may take a few minutes...")

    try:
        # Run Lightning's consolidation command
        cmd = ["python", "-m", "lightning.pytorch.utilities.consolidate_checkpoint", str(ckpt_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if consolidated_path.exists():
            print(f"Successfully consolidated checkpoint to {consolidated_path}")
            return consolidated_path
        else:
            raise RuntimeError("Consolidation command succeeded but consolidated.ckpt not found")

    except subprocess.CalledProcessError as e:
        print(f"Error consolidating checkpoint: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def load_checkpoint(ckpt_path, auto_consolidate=True):
    """
    Load FSDP checkpoint from PyTorch Lightning.

    Args:
        ckpt_path: Path to checkpoint (directory or .ckpt file)
        auto_consolidate: If True, automatically consolidate unconsolidated checkpoints

    Returns:
        State dict
    """
    ckpt_path = Path(ckpt_path)

    # If it's a directory and auto_consolidate is enabled, consolidate first
    if ckpt_path.is_dir() and auto_consolidate:
        ckpt_path = consolidate_checkpoint(ckpt_path)

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


if __name__ == "__main__":
    # ckpt_path = "/mnt/weka/models/outputs/cont_tokens/llm_sync_tok/ft/1b_filtered_ce_long/epoch=18-step=95000.ckpt"
    ckpt_path = (
        "/mnt/weka/models/outputs/cont_tokens/llm_sync_tok/ft/1b_filtered_ce_long_gaussian/epoch=5-step=100000.ckpt"
    )
    model = TadaForCausalLM(
        TadaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=128000,
            eos_token_id=128001,
            head_dim=64,
            hidden_size=2048,
            intermediate_size=8192,
            max_position_embeddings=131072,
            model_type="llama",
            num_attention_heads=32,
            num_hidden_layers=16,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            rope_theta=500000.0,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            vocab_size=128256,
            # TADA-specific configs
            shift_acoustic=5,
            head_layers=6,
            head_ffn_ratio=4.0,
            bottleneck_dim=None,
            num_time_classes=256,
        )
    )
    state_dict = load_checkpoint(ckpt_path, auto_consolidate=True)

    for key in list(state_dict.keys()):
        if key.startswith("llm_model."):
            state_dict[key[10:]] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(torch.bfloat16)
    model.save_pretrained(os.path.join(get_weight_dir(), "uploads", "llm"))

    api = HfApi()
    api.upload_folder(
        folder_path=os.path.join(get_weight_dir(), "uploads", "llm"),
        repo_id="HumeAI/tada-1b",
        repo_type="model",
        # revision="dev",
        # create_pr=True,
    )
