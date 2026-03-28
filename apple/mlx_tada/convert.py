import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from .utils import SUPPORTED_LANGUAGES

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    import torch
except ImportError:
    torch = None

__all__ = [
    "convert_tada_model",
    "convert_encoder",
    "convert_decoder",
    "convert_aligner",
    "download_from_hf",
    "resolve_hf",
    "convert_all",
]


def save_as_bfloat16(state: dict[str, np.ndarray], path: str) -> None:
    mlx_state: dict[str, mx.array] = {}

    for k, v in state.items():
        arr = mx.array(v)
        if arr.dtype in (mx.float32, mx.float64):
            arr = arr.astype(mx.bfloat16)
        mlx_state[k] = arr

    mx.save_safetensors(path, mlx_state)
    del mlx_state


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    assert torch is not None, "Weight conversion requires PyTorch. Install with: pip install torch"
    path = Path(path)

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))

    tensors: dict[str, np.ndarray] = {}

    for weight_file in files:
        with safe_open(str(weight_file), framework="pt") as st:
            for key in st.keys():
                t = st.get_tensor(key)
                if t.dtype == torch.bfloat16:
                    t = t.float()
                tensors[key] = t.numpy()

    return tensors


def fold_weight_norm(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    wn_prefixes: set[str] = set()
    pattern = re.compile(r"^(.+)\.parametrizations\.weight\.original[01]$")

    for key in list(state.keys()):
        m = pattern.match(key)
        if m:
            wn_prefixes.add(m.group(1))

    for prefix in wn_prefixes:
        g_key = f"{prefix}.parametrizations.weight.original0"
        v_key = f"{prefix}.parametrizations.weight.original1"

        if g_key not in state or v_key not in state:
            continue

        gain = state.pop(g_key)
        direction = state.pop(v_key)
        axes = tuple(i for i in range(direction.ndim) if gain.shape[i] == 1 or i >= gain.ndim)

        if not axes:
            axes = tuple(range(1, direction.ndim))

        norm = np.sqrt(np.sum(direction**2, axis=axes, keepdims=True) + 1e-12)
        weight = gain * direction / norm
        state[f"{prefix}.weight"] = weight

    return state


def transpose_snake_alpha(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for key in list(state.keys()):
        if key.endswith(".alpha") and state[key].ndim == 3:
            state[key] = np.transpose(state[key], (0, 2, 1))

    return state


def transpose_conv_weights(state: dict[str, np.ndarray], conv_keys: set[str]) -> dict[str, np.ndarray]:
    for key in conv_keys:
        if key in state and state[key].ndim == 3:
            state[key] = np.swapaxes(state[key], 1, 2)

    return state


def rename_adaln_keys(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    renames: dict[str, str] = {}

    for key in list(state.keys()):
        new_key = key.replace(".adaLN_modulation.1.", ".adaLN_modulation_linear.")
        if new_key != key:
            renames[key] = new_key

    for old, new in renames.items():
        state[new] = state.pop(old)

    return state


def rename_timestep_embedder_keys(
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    renames: dict[str, str] = {}

    for key in list(state.keys()):
        new_key = key
        new_key = new_key.replace(".t_embedder.mlp.0.", ".t_embedder.mlp_0.")
        new_key = new_key.replace(".t_embedder.mlp.2.", ".t_embedder.mlp_2.")
        if new_key != key:
            renames[key] = new_key

    for old, new in renames.items():
        state[new] = state.pop(old)

    return state


def collect_encoder_conv_keys(state: dict[str, np.ndarray]) -> set[str]:
    conv_keys: set[str] = set()

    for key in state:
        if not key.endswith(".weight"):
            continue
        if state[key].ndim != 3:
            continue
        conv_keys.add(key)

    return conv_keys


def rename_encoder_keys(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    renames: dict[str, str] = {}
    num_encoder_blocks = 4

    for key in list(state.keys()):
        new_key = key
        new_key = re.sub(r"^wav_encoder\.block\.0\.", "wav_encoder.initial_conv.", new_key)

        for i in range(num_encoder_blocks):
            pt_idx = i + 1

            for ru_idx, ru_name in enumerate(["res1", "res2", "res3"]):
                for sub_idx, sub_name in [
                    (0, "snake1"),
                    (1, "conv1"),
                    (2, "snake2"),
                    (3, "conv2"),
                ]:
                    old_pat = f"wav_encoder.block.{pt_idx}.block.{ru_idx}.block.{sub_idx}."
                    new_pat = f"wav_encoder.blocks.{i}.{ru_name}.{sub_name}."
                    new_key = new_key.replace(old_pat, new_pat)

            old_pat = f"wav_encoder.block.{pt_idx}.block.3."
            new_pat = f"wav_encoder.blocks.{i}.snake."
            new_key = new_key.replace(old_pat, new_pat)

            old_pat = f"wav_encoder.block.{pt_idx}.block.4."
            new_pat = f"wav_encoder.blocks.{i}.conv."
            new_key = new_key.replace(old_pat, new_pat)

        new_key = new_key.replace(f"wav_encoder.block.{num_encoder_blocks + 1}.", "wav_encoder.final_snake.")
        new_key = new_key.replace(f"wav_encoder.block.{num_encoder_blocks + 2}.", "wav_encoder.final_conv.")

        new_key = re.sub(
            r"local_attention_encoder\.layers\.(\d+)\.ffn\.0\.",
            r"local_attention_encoder.layers.\1.linear1.",
            new_key,
        )
        new_key = re.sub(
            r"local_attention_encoder\.layers\.(\d+)\.ffn\.3\.",
            r"local_attention_encoder.layers.\1.linear2.",
            new_key,
        )

        if new_key != key:
            renames[key] = new_key

    for old, new in renames.items():
        state[new] = state.pop(old)
    return state


def rename_decoder_keys(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    renames: dict[str, str] = {}

    for key in list(state.keys()):
        new_key = key
        new_key = new_key.replace("wav_decoder.model.0.", "wav_decoder.initial_conv.")

        for i in range(4):
            pt_idx = i + 1
            new_key = new_key.replace(f"wav_decoder.model.{pt_idx}.block.0.", f"wav_decoder.blocks.{i}.snake.")
            new_key = new_key.replace(
                f"wav_decoder.model.{pt_idx}.block.1.",
                f"wav_decoder.blocks.{i}.conv_transpose.",
            )

            for ru_idx, ru_name in [(2, "res1"), (3, "res2"), (4, "res3")]:
                for sub_idx, sub_name in [
                    (0, "snake1"),
                    (1, "conv1"),
                    (2, "snake2"),
                    (3, "conv2"),
                ]:
                    old_pat = f"wav_decoder.model.{pt_idx}.block.{ru_idx}.block.{sub_idx}."
                    new_pat = f"wav_decoder.blocks.{i}.{ru_name}.{sub_name}."
                    new_key = new_key.replace(old_pat, new_pat)

        new_key = new_key.replace("wav_decoder.model.5.", "wav_decoder.final_snake.")
        new_key = new_key.replace("wav_decoder.model.6.", "wav_decoder.final_conv.")

        new_key = re.sub(
            r"local_attention_decoder\.layers\.(\d+)\.ffn\.0\.",
            r"local_attention_decoder.layers.\1.linear1.",
            new_key,
        )
        new_key = re.sub(
            r"local_attention_decoder\.layers\.(\d+)\.ffn\.3\.",
            r"local_attention_decoder.layers.\1.linear2.",
            new_key,
        )

        if new_key != key:
            renames[key] = new_key

    for old, new in renames.items():
        state[new] = state.pop(old)
    return state


def rename_tada_model_keys(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = rename_adaln_keys(state)
    state = rename_timestep_embedder_keys(state)
    return state


def transpose_conv_transpose_weights(
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:

    for key in list(state.keys()):
        if "conv_transpose." in key and key.endswith(".weight") and state[key].ndim == 3:
            conv_weight = state[key]
            state[key] = np.transpose(conv_weight, (1, 2, 0))

    return state


def remove_rope_keys(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    to_remove = [k for k in state if "rotary_emb" in k]

    for k in to_remove:
        del state[k]

    return state


def remove_encoder_buffers(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    to_remove = [k for k in state if "rope_freqs" in k or "_precomputed_mask" in k]

    for k in to_remove:
        del state[k]

    return state


def handle_encoder_rope(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    removes = []

    for key in list(state.keys()):
        if key.endswith(".rope_freqs"):
            prefix = key[: -len(".rope_freqs")]
            freqs = state[key]
            state[f"{prefix}._rope_cos"] = freqs[..., 0]
            state[f"{prefix}._rope_sin"] = freqs[..., 1]
            removes.append(key)

    for k in removes:
        del state[k]

    return state


def convert_tada_model(
    model_path: str,
    output_path: str,
    config_path: str | None = None,
) -> None:
    print(f"Loading TadaForCausalLM weights from {model_path}")
    state = load_safetensors(model_path)
    print(f"  {len(state)} tensors loaded")
    decoder_keys = [k for k in state if k.startswith("_decoder.") or k.startswith("decoder.")]

    for k in decoder_keys:
        del state[k]
    if decoder_keys:
        print(f"  Stripped {len(decoder_keys)} embedded decoder keys (converted separately)")

    state = remove_rope_keys(state)
    state = rename_tada_model_keys(state)

    if "lm_head.weight" not in state and "model.embed_tokens.weight" in state:
        print("  Tied embeddings detected")

    out_dir = Path(output_path) / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_as_bfloat16(state, str(out_dir / "weights.safetensors"))
    print(f"  Saved {len(state)} tensors to {out_dir / 'weights.safetensors'}")

    if config_path:
        shutil.copy(config_path, out_dir / "config.json")

    print(f"  Model conversion complete -> {out_dir}")


def convert_encoder(
    codec_path: str,
    output_path: str,
    subfolder: str = "encoder",
) -> None:
    enc_path = Path(codec_path) / subfolder if (Path(codec_path) / subfolder).exists() else Path(codec_path)
    print(f"Loading Encoder weights from {enc_path}")
    state = load_safetensors(enc_path)
    print(f"  {len(state)} tensors loaded")
    state = fold_weight_norm(state)
    state = remove_encoder_buffers(state)
    state = handle_encoder_rope(state)
    state = rename_encoder_keys(state)
    conv_keys = collect_encoder_conv_keys(state)
    state = transpose_conv_weights(state, conv_keys)
    state = transpose_snake_alpha(state)
    out_dir = Path(output_path) / "encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_as_bfloat16(state, str(out_dir / "weights.safetensors"))
    print(f"  Saved {len(state)} tensors to {out_dir / 'weights.safetensors'}")
    print(f"  Encoder conversion complete -> {out_dir}")


def convert_decoder(
    codec_path: str,
    output_path: str,
    subfolder: str = "decoder",
) -> None:
    dec_path = Path(codec_path) / subfolder if (Path(codec_path) / subfolder).exists() else Path(codec_path)
    print(f"Loading Decoder weights from {dec_path}")
    state = load_safetensors(dec_path)
    print(f"  {len(state)} tensors loaded")
    state = fold_weight_norm(state)
    state = remove_encoder_buffers(state)
    state = handle_encoder_rope(state)
    state = rename_decoder_keys(state)
    state = transpose_conv_transpose_weights(state)
    conv_keys = set()

    for key in state:
        if key.endswith(".weight") and state[key].ndim == 3 and "conv_transpose" not in key:
            conv_keys.add(key)

    state = transpose_conv_weights(state, conv_keys)
    state = transpose_snake_alpha(state)
    out_dir = Path(output_path) / "decoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_as_bfloat16(state, str(out_dir / "weights.safetensors"))
    print(f"  Saved {len(state)} tensors to {out_dir / 'weights.safetensors'}")
    print(f"  Decoder conversion complete -> {out_dir}")


def rename_aligner_keys(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    renames: dict[str, str] = {}

    for key in list(state.keys()):
        new_key = key
        new_key = new_key.replace("encoder.wav2vec2.", "wav2vec2.")
        new_key = new_key.replace("encoder.lm_head.", "wav2vec2.lm_head.")
        if new_key != key:
            renames[key] = new_key

    for old, new in renames.items():
        state[new] = state.pop(old)

    return state


def collect_aligner_conv_keys(state: dict[str, np.ndarray]) -> set[str]:
    conv_keys: set[str] = set()

    for key in state:
        if not key.endswith(".weight"):
            continue
        if state[key].ndim != 3:
            continue
        if "pos_conv_embed" in key:
            continue
        conv_keys.add(key)

    return conv_keys


def convert_aligner(
    codec_path: str,
    output_path: str,
    subfolder: str = "aligner",
    output_subfolder: str = "aligner",
) -> None:
    aligner_path = Path(codec_path) / subfolder if (Path(codec_path) / subfolder).exists() else Path(codec_path)
    print(f"Loading Aligner weights from {aligner_path}")
    state = load_safetensors(aligner_path)
    print(f"  {len(state)} tensors loaded")
    to_remove = [k for k in state if "masked_spec_embed" in k]

    for k in to_remove:
        del state[k]

    if to_remove:
        print(f"  Removed {len(to_remove)} unused keys (masked_spec_embed)")

    state = fold_weight_norm(state)
    state = rename_aligner_keys(state)
    conv_keys = collect_aligner_conv_keys(state)
    state = transpose_conv_weights(state, conv_keys)
    pos_conv_key = "wav2vec2.encoder.pos_conv_embed.conv.weight"

    if pos_conv_key in state and state[pos_conv_key].ndim == 3:
        pos_conv_weight = state[pos_conv_key]
        state[pos_conv_key] = np.swapaxes(pos_conv_weight, 1, 2)

    out_dir = Path(output_path) / output_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    save_as_bfloat16(state, str(out_dir / "weights.safetensors"))
    print(f"  Saved {len(state)} tensors to {out_dir / 'weights.safetensors'}")
    print(f"  Aligner conversion complete -> {out_dir}")


def download_from_hf(repo_id: str, subfolder: str | None = None, local_dir: str | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

    path = snapshot_download(
        repo_id,
        local_dir=local_dir,
        allow_patterns=([f"{subfolder}/*"] if subfolder else None),
    )
    if subfolder:
        return Path(path) / subfolder

    return Path(path)


def resolve_hf(repo_or_path: str, output: Path, cache_name: str) -> str:
    local_path = Path(repo_or_path)

    if local_path.exists():
        return str(local_path)

    cached = output / "hf_cache" / cache_name

    if cached.exists():
        return str(cached)

    print(f"Downloading {repo_or_path} from HuggingFace...")
    return str(download_from_hf(repo_or_path, local_dir=str(cached)))


def convert_all(model_repo: str, codec_repo: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_path = resolve_hf(model_repo, output, "model")
    codec_path = resolve_hf(codec_repo, output, "codec")
    config_json = Path(model_path) / "config.json"
    convert_tada_model(model_path, str(output), str(config_json) if config_json.exists() else None)
    convert_encoder(codec_path, str(output))
    convert_decoder(codec_path, str(output))
    convert_aligner(codec_path, str(output))
    codec_root = Path(codec_path)

    for language in SUPPORTED_LANGUAGES:
        subfolder = f"aligner-{language}"
        aligner_dir = codec_root / subfolder
        if aligner_dir.exists():
            convert_aligner(codec_path, str(output), subfolder=subfolder, output_subfolder=subfolder)

    print(f"\nAll conversions complete! Weights saved to: {output}")
