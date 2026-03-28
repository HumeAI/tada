import json
import logging
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from .aligner import Aligner
from .audio import load_audio, resample_audio, transcribe_audio
from .config import GenerationOutput, InferenceOptions, Reference, TadaConfig
from .convert import collect_aligner_conv_keys, fold_weight_norm, rename_aligner_keys, transpose_conv_weights
from .decoder import Decoder
from .encoder import Encoder
from .llm import KVCache, LlamaModel, build_rope_cache
from .utils import SUPPORTED_LANGUAGES, SUPPORTED_LANGUAGES_SET, decode_gray_code_to_time, normalize_text
from .vibevoice import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig

log = logging.getLogger("tada")

GREY = "\033[90m"
CYAN = "\033[36m"
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return f"{CYAN}tada{RESET} {GREY}{msg}{RESET}"


def setup_logging():
    """Enable colored debug logging for the TADA model.

    Attaches a StreamHandler with ANSI-colored output to the 'tada' logger.
    Also enabled by setting the DEBUG=1 environment variable when using the CLI.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)


__all__ = [
    "setup_logging",
    "load_weights",
    "prepare_reference",
    "TadaForCausalLM",
]


def load_weights(model: nn.Module, weights_path: Path) -> None:
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()), strict=False)


def prepare_reference(
    audio_path: str,
    audio_text: str | None,
    encoder: nn.Module,
    aligner: nn.Module,
    tokenizer,
) -> Reference:
    if not audio_text:
        audio_text = transcribe_audio(audio_path)

    audio_text = normalize_text(audio_text)
    audio_24k, sr = load_audio(audio_path, target_sr=24000)
    audio_24k_np = audio_24k.reshape(1, -1)
    audio_len = audio_24k_np.shape[1]
    audio_16k = resample_audio(audio_24k, 24000, 16000).reshape(1, -1)
    text_token_ids = tokenizer.encode(audio_text, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    text_tokens_np = np.array([text_token_ids], dtype=np.int64)
    text_tokens_len = len(text_token_ids)
    input_lengths = np.array([math.ceil(audio_len / sr * 50)], dtype=np.int64)
    token_positions_np, token_masks_np = aligner(
        mx.array(audio_16k),
        text_tokens_np,
        input_lengths,
        eos_token_id,
    )
    enc_output = encoder.encode(
        mx.array(audio_24k_np),
        mx.array(token_positions_np),
        mx.array(token_masks_np),
        audio_length=mx.array([audio_len]),
        text=[audio_text],
        text_tokens=mx.array(text_tokens_np),
        text_tokens_len=mx.array([text_tokens_len]),
    )
    mx.eval(enc_output.token_values, enc_output.token_positions, enc_output.token_masks)
    return Reference(
        token_values=np.array(enc_output.token_values),
        token_positions=np.array(enc_output.token_positions),
        token_masks=np.array(enc_output.token_masks) if enc_output.token_masks is not None else None,
        text_tokens=np.array(enc_output.text_tokens) if enc_output.text_tokens is not None else text_tokens_np,
        text_tokens_len=np.array(enc_output.text_tokens_len)
        if enc_output.text_tokens_len is not None
        else np.array([text_tokens_len]),
        audio_len=np.array([audio_len]),
        text=audio_text,
        sample_rate=sr,
    )


class TadaForCausalLM(nn.Module):
    def __init__(self, config: TadaConfig):
        super().__init__()
        self.config = config
        self.num_time_bits = math.ceil(math.log2(config.num_time_classes))
        self.time_dim = 2 * self.num_time_bits
        self.model = LlamaModel(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.acoustic_proj = nn.Linear(config.acoustic_dim, config.hidden_size)
        self.time_start_embed = nn.Embedding(config.num_time_classes, config.hidden_size)
        self.time_end_embed = nn.Embedding(config.num_time_classes, config.hidden_size)
        self.acoustic_mask_emb = nn.Embedding(2, config.hidden_size)
        self.has_bottleneck = config.bottleneck_dim is not None

        if self.has_bottleneck:
            self.bottleneck_proj = nn.Linear(config.hidden_size, config.bottleneck_dim)

        head_hidden = config.bottleneck_dim if config.bottleneck_dim else config.hidden_size
        latent_size = config.acoustic_dim + self.time_dim
        self.prediction_head = VibeVoiceDiffusionHead(
            VibeVoiceDiffusionHeadConfig(
                hidden_size=head_hidden,
                head_layers=config.head_layers,
                head_ffn_ratio=config.head_ffn_ratio,
                latent_size=latent_size,
                rms_norm_eps=1e-5,
            )
        )
        self._rope_cos: mx.array | None = None
        self._rope_sin: mx.array | None = None
        self._rope_len: int = 0

    def ensure_rope(self, seq_len: int) -> None:
        if seq_len <= self._rope_len:
            return

        new_len = max(seq_len, 4096)
        self._rope_cos, self._rope_sin = build_rope_cache(
            new_len, self.config.head_dim, self.config.rope_theta, self.config.rope_scaling
        )
        self._rope_len = new_len

    def get_rope_slice(self, start: int, length: int) -> tuple[mx.array, mx.array]:
        self.ensure_rope(start + length)
        return self._rope_cos[start : start + length], self._rope_sin[start : start + length]

    def lm_head_forward(self, h: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return h @ self.model.embed_tokens.weight.T

        return self.lm_head(h)

    def apply_bottleneck(self, h: mx.array) -> mx.array:
        if self.has_bottleneck:
            return self.bottleneck_proj(h)

        return h

    def forward_one_step(
        self,
        input_ids: mx.array,
        acoustic_features: mx.array,
        acoustic_masks: mx.array,
        time_len_before: mx.array,
        time_len_after: mx.array,
        cache: list[KVCache] | None = None,
    ) -> tuple[mx.array, mx.array]:
        inputs_embeds = (
            self.model.embed_tokens(input_ids)
            + self.acoustic_proj(acoustic_features)
            + self.acoustic_mask_emb(acoustic_masks)
            + self.time_start_embed(time_len_before)
            + self.time_end_embed(time_len_after)
        )
        cache_len = cache[0].seq_len if cache and cache[0].keys is not None else 0
        seq_len = inputs_embeds.shape[1]
        cos, sin = self.get_rope_slice(cache_len, seq_len)

        if seq_len > 1:
            total = cache_len + seq_len
            row_idx = mx.arange(seq_len).reshape(-1, 1) + cache_len
            col_idx = mx.arange(total).reshape(1, -1)
            mask = mx.where(col_idx <= row_idx, mx.array(0.0), mx.array(-1e9))
            mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)
        else:
            mask = None

        hidden = self.model(inputs_embeds, cos, sin, mask, cache)
        logits = self.lm_head_forward(hidden)
        return hidden, logits

    def build_prompt_inputs_embeds(
        self,
        input_ids: mx.array,
        prompt_acoustic_features: mx.array,
        prompt_acoustic_masks: mx.array,
        prompt_time_len_before: mx.array,
        prompt_time_len_after: mx.array,
        prompt_len: int,
    ) -> mx.array:
        B = input_ids.shape[0]
        shift = self.config.shift_acoustic
        token_emb = self.model.embed_tokens(input_ids[:, :prompt_len])
        acoustic_full = mx.zeros((B, prompt_len, self.config.acoustic_dim))
        masks_full = mx.zeros((B, prompt_len), dtype=mx.int32)
        n_ac = min(prompt_len - shift - 1, prompt_acoustic_features.shape[1])
        if n_ac > 0:
            acoustic_full = acoustic_full.at[:, shift + 1 : shift + 1 + n_ac].add(prompt_acoustic_features[:, :n_ac])
            masks_full = masks_full.at[:, shift + 1 : shift + 1 + n_ac].add(prompt_acoustic_masks[:, :n_ac])
        acoustic_emb = self.acoustic_proj(acoustic_full) + self.acoustic_mask_emb(masks_full)
        time_before = mx.zeros((B, prompt_len), dtype=mx.int32)
        time_after = mx.zeros((B, prompt_len), dtype=mx.int32)
        n_t = min(prompt_len - shift - 1, prompt_time_len_before.shape[1] - 1)
        if n_t > 0:
            time_before = time_before.at[:, shift + 1 : shift + 1 + n_t].add(prompt_time_len_before[:, 1 : 1 + n_t])
            time_after = time_after.at[:, shift + 1 : shift + 1 + n_t].add(prompt_time_len_after[:, 1 : 1 + n_t])
        time_emb = self.time_start_embed(time_before) + self.time_end_embed(time_after)
        return token_emb + acoustic_emb + time_emb

    @staticmethod
    def sample_token(
        logits: mx.array,
        input_ids: mx.array,
        opts: InferenceOptions,
        pad_token_id: int,
    ) -> mx.array:
        logits = logits.at[:, pad_token_id].add(mx.array(-1e9))

        if opts.text_do_sample:
            if opts.text_repetition_penalty != 1.0:
                penalty = opts.text_repetition_penalty
                prev_scores = mx.take_along_axis(logits, input_ids, axis=1)
                penalised = mx.where(prev_scores < 0, prev_scores * penalty, prev_scores / penalty)

                for batch_idx in range(logits.shape[0]):
                    for j in range(input_ids.shape[1]):
                        tid = input_ids[batch_idx, j].item()
                        logits = logits.at[batch_idx, tid].add(penalised[batch_idx, j] - prev_scores[batch_idx, j])

            logits = logits / opts.text_temperature

            if opts.text_top_k > 0:
                top_k = min(opts.text_top_k, logits.shape[-1])
                kth = mx.sort(logits, axis=-1)[..., -top_k : -top_k + 1]
                logits = mx.where(logits < kth, mx.array(-1e9), logits)

            if 0.0 < opts.text_top_p < 1.0:
                sorted_idx = mx.argsort(logits, axis=-1)[:, ::-1]
                sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=1)
                cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
                probs_to_remove = cum_probs - mx.softmax(sorted_logits, axis=-1) >= opts.text_top_p
                sorted_logits = mx.where(probs_to_remove, mx.array(-1e9), sorted_logits)
                unsorted_idx = mx.argsort(sorted_idx, axis=-1)
                logits = mx.take_along_axis(sorted_logits, unsorted_idx, axis=1)

            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-12))
            return mx.expand_dims(next_token, -1)
        else:
            return mx.expand_dims(mx.argmax(logits, axis=-1), -1)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        language: str | None = None,
        quantize: int | None = None,
        quantize_group_size: int = 64,
    ) -> "TadaForCausalLM":
        """Load a TADA model from a Hugging Face repo with pre-converted MLX weights.

        Downloads the weights on first use and caches them locally.

        Args:
            repo_id: Hugging Face repo ID (e.g. "HumeAI/mlx-tada-3b").
            language: Language code for the reference aligner. Uses `aligner-{language}`
                when provided, otherwise defaults to `aligner` (English).
            quantize: Bit width for quantization (4 or 8). None for full precision (bf16).
            quantize_group_size: Group size for quantization. Default 64.

        Returns:
            A fully initialized TadaForCausalLM ready for inference.
        """
        from huggingface_hub import snapshot_download

        if language is not None:
            language = language.lower()
            if language not in SUPPORTED_LANGUAGES_SET:
                supported = ", ".join(SUPPORTED_LANGUAGES)
                raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported}")

        weights_dir = snapshot_download(repo_id)

        # HF MLX repos include the default English aligner. For multilingual alignment
        # we mirror the PyTorch setup and fetch language-specific aligner weights from
        # the codec repo.
        if language is None:
            return cls.from_weights(
                weights_dir,
                language=None,
                quantize=quantize,
                quantize_group_size=quantize_group_size,
            )

        model = cls.from_weights(
            weights_dir,
            language=None,
            quantize=quantize,
            quantize_group_size=quantize_group_size,
        )
        codec_snapshot = snapshot_download("HumeAI/tada-codec", allow_patterns=[f"aligner-{language}/*"])
        aligner_dir = Path(codec_snapshot) / f"aligner-{language}"
        converted_weights = aligner_dir / "weights.safetensors"
        raw_weights = aligner_dir / "model.safetensors"

        if not converted_weights.exists() and not raw_weights.exists():
            raise FileNotFoundError(
                f"Missing codec aligner weights in {aligner_dir}. Expected one of: "
                f"{converted_weights}, {raw_weights}"
            )

        aligner = Aligner()
        if converted_weights.exists():
            load_weights(aligner, converted_weights)
        else:
            # Convert HF codec aligner tensors to MLX layout in-memory.
            state: dict[str, np.ndarray] = {}
            for k, v in mx.load(str(raw_weights)).items():
                arr = v.astype(mx.float32) if v.dtype == mx.bfloat16 else v
                state[k] = np.array(arr)
            to_remove = [k for k in state if "masked_spec_embed" in k]

            for k in to_remove:
                del state[k]

            state = fold_weight_norm(state)
            state = rename_aligner_keys(state)
            conv_keys = collect_aligner_conv_keys(state)
            state = transpose_conv_weights(state, conv_keys)
            pos_conv_key = "wav2vec2.encoder.pos_conv_embed.conv.weight"
            if pos_conv_key in state and state[pos_conv_key].ndim == 3:
                state[pos_conv_key] = np.swapaxes(state[pos_conv_key], 1, 2)

            mlx_state = {}
            for k, v in state.items():
                arr = mx.array(v)
                if arr.dtype in (mx.float32, mx.float64):
                    arr = arr.astype(mx.bfloat16)
                mlx_state[k] = arr

            aligner.load_weights(list(mlx_state.items()), strict=False)

        mx.eval(aligner.parameters())
        model._aligner = aligner
        return model

    @classmethod
    def from_weights(
        cls,
        weights_dir: str | Path,
        language: str | None = None,
        quantize: int | None = None,
        quantize_group_size: int = 64,
    ) -> "TadaForCausalLM":
        """Load a TADA model with all components (LLM, encoder, decoder, aligner, tokenizer).

        Args:
            weights_dir: Path to the converted MLX weights directory.
            language: Language code for the reference aligner. Uses `aligner-{language}`
                when provided, otherwise defaults to `aligner` (English).
            quantize: Bit width for quantization (4 or 8). None for full precision (bf16).
                Quantizes the LLM backbone and VibeVoice diffusion head.
            quantize_group_size: Group size for quantization. Default 64.

        Returns:
            A fully initialized TadaForCausalLM ready for inference.
        """
        weights_dir = Path(weights_dir)
        if language is not None:
            language = language.lower()
            if language not in SUPPORTED_LANGUAGES_SET:
                supported = ", ".join(SUPPORTED_LANGUAGES)
                raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported}")

        model_dir = weights_dir / "model"
        config_path = model_dir / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                config = TadaConfig.from_dict(json.load(f))
        else:
            config = TadaConfig()

        model = cls(config)
        load_weights(model, model_dir / "weights.safetensors")

        if quantize is not None:
            nn.quantize(
                model.model,
                group_size=quantize_group_size,
                bits=quantize,
                class_predicate=lambda path, m: isinstance(m, nn.Linear),
            )
            nn.quantize(
                model.prediction_head,
                group_size=quantize_group_size,
                bits=quantize,
                class_predicate=lambda path, m: (
                    isinstance(m, nn.Linear) and m.weight.shape[1] % quantize_group_size == 0
                ),
            )
        mx.eval(model.parameters())
        encoder = Encoder()
        load_weights(encoder, weights_dir / "encoder" / "weights.safetensors")
        mx.eval(encoder.parameters())
        model._encoder = encoder
        decoder = Decoder()
        load_weights(decoder, weights_dir / "decoder" / "weights.safetensors")
        mx.eval(decoder.parameters())
        model._decoder = decoder

        aligner_subdir = f"aligner-{language}" if language else "aligner"
        aligner_weights = weights_dir / aligner_subdir / "weights.safetensors"
        if not aligner_weights.exists():
            if language is None:
                raise FileNotFoundError(f"Missing aligner weights: {aligner_weights}")

            default_aligner = weights_dir / "aligner" / "weights.safetensors"
            raise FileNotFoundError(
                f"Missing aligner weights for language '{language}': {aligner_weights}. "
                f"Expected converted multilingual aligners under '{weights_dir}'. "
                f"If you only have English weights, use language=None (default), which looks for {default_aligner}."
            )

        aligner = Aligner()
        load_weights(aligner, aligner_weights)
        mx.eval(aligner.parameters())
        model._aligner = aligner
        model._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        return model

    def load_reference(self, audio_path: str, audio_text: str | None = None) -> Reference:
        """Encode a reference audio file into a Reference for voice cloning.

        Runs the aligner (forced alignment) and encoder to extract acoustic
        features and token positions from the reference audio.

        Args:
            audio_path: Path to a reference audio file (any format soundfile supports).
            audio_text: Transcript of the reference audio. If None, auto-transcribes
                using mlx-whisper (must be installed).

        Returns:
            A Reference that can be passed to generate() or saved with ref.save().
        """
        return prepare_reference(audio_path, audio_text, self._encoder, self._aligner, self._tokenizer)

    def _build_inputs(
        self,
        text: str,
        reference: Reference,
        num_transition_steps: int,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, int]:
        shift = self.config.shift_acoustic
        bos_id = self.config.bos_token_id
        eot_id = self.config.eot_id
        ref_token_ids = self._tokenizer.encode(reference.text, add_special_tokens=False)
        target_token_ids = self._tokenizer.encode(" " + text, add_special_tokens=False)
        token_positions_mx = mx.array(reference.token_positions)
        pos_padded = mx.concatenate([mx.ones((1, 1), dtype=token_positions_mx.dtype), token_positions_mx], axis=1)
        time_gaps = mx.clip(token_positions_mx - pos_padded[:, :-1], 0, self.config.num_time_classes - 1)
        time_gaps = mx.concatenate([mx.zeros((1, 1), dtype=time_gaps.dtype), time_gaps], axis=1)
        tlb = time_gaps[:, :-1].astype(mx.int32)
        tla = time_gaps[:, 1:].astype(mx.int32)
        paf = mx.array(reference.token_values)
        pam = mx.ones(paf.shape[:2], dtype=mx.int32)
        prefix_text = (
            "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        prefix_ids = self._tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_len = len(prefix_ids)
        prefill_token_ids = [bos_id] + prefix_ids + ref_token_ids
        paf = mx.pad(paf, [(0, 0), (prefix_len, 0), (0, 0)])
        pam = mx.pad(pam, [(0, 0), (prefix_len, 0)])
        tlb = mx.pad(tlb, [(0, 0), (prefix_len, 0)])
        tla = mx.pad(tla, [(0, 0), (prefix_len, 0)])

        if num_transition_steps > 0:
            paf = paf[:, :-num_transition_steps, :]
            pam = pam[:, :-num_transition_steps]
            tlb = tlb[:, :-num_transition_steps]
            tla = tla[:, :-num_transition_steps]

        pam = mx.concatenate([pam[:, 1:], mx.ones_like(pam[:, :1])], axis=1)
        input_ids = mx.array([prefill_token_ids + target_token_ids + [eot_id] * shift], dtype=mx.int32)
        prompt_token_len = paf.shape[1]
        prompt_ids = input_ids[:, :prompt_token_len]
        is_start = prompt_ids == self.config.start_header_id
        is_end = prompt_ids == self.config.end_header_id
        header_depth = mx.cumsum(is_start.astype(mx.int32), axis=1) - mx.cumsum(is_end.astype(mx.int32), axis=1)
        in_header = (header_depth > 0) | is_start | is_end
        is_structural = in_header | (prompt_ids == eot_id) | (prompt_ids == bos_id) | (prompt_ids == 128001)
        masked_ids = mx.where(
            is_structural, prompt_ids, mx.full(prompt_ids.shape, self.config.pad_token_id, dtype=mx.int32)
        )
        input_ids = mx.concatenate([masked_ids, input_ids[:, prompt_token_len:]], axis=1)
        n_ac = min(len(prefill_token_ids) - shift - 1, paf.shape[1])
        n_t = min(len(prefill_token_ids) - shift - 1, tlb.shape[1] - 1)
        n_frames_cap = max(0, tlb.shape[1] - 2)
        n_prefill = min(n_ac, n_t, n_frames_cap) if (n_ac > 0 and n_t > 0) else 0
        prefill_len = min(len(prefill_token_ids), shift + n_prefill + 1) if n_prefill > 0 else 0

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"after masking={self._tokenizer.decode(np.array(input_ids[0]).tolist())}")
            log.debug(f"prefill_len={prefill_len}, prompt_frames={paf.shape[1]}")

        return input_ids, paf, pam, tlb, tla, prefill_len

    def _prefill(
        self,
        input_ids: mx.array,
        paf: mx.array,
        pam: mx.array,
        tlb: mx.array,
        tla: mx.array,
        prefill_len: int,
    ) -> list[KVCache]:
        cache = [KVCache() for _ in range(self.config.num_hidden_layers)]
        if prefill_len > 0:
            embeds = self.build_prompt_inputs_embeds(input_ids, paf, pam, tlb, tla, prefill_len)
            embeds = mx.concatenate([embeds, embeds], axis=0)
            cos, sin = self.get_rope_slice(0, prefill_len)
            row_idx = mx.arange(prefill_len).reshape(-1, 1)
            col_idx = mx.arange(prefill_len).reshape(1, -1)
            cmask = mx.where(col_idx <= row_idx, mx.array(0.0), mx.array(-1e9))
            cmask = mx.expand_dims(mx.expand_dims(cmask, 0), 0)
            hidden = self.model(embeds, cos, sin, cmask, cache)
            mx.eval(hidden)

            for layer_cache in cache:
                mx.eval(layer_cache.keys, layer_cache.values)

        return cache

    def _autoregressive_loop(
        self,
        input_ids: mx.array,
        paf: mx.array,
        pam: mx.array,
        tlb: mx.array,
        tla: mx.array,
        prefill_len: int,
        cache: list[KVCache],
        opts: InferenceOptions,
        num_extra_steps: int = 0,
    ) -> tuple[list[mx.array], list[mx.array], list[mx.array]]:
        shift = self.config.shift_acoustic
        pad_id = self.config.pad_token_id
        B = 1
        n_pf = prefill_len - shift
        acoustic_features_val = (
            mx.expand_dims(paf[:, n_pf - 1], 1) if n_pf > 0 else mx.zeros((B, 1, self.config.acoustic_dim))
        )
        acoustic_masks_val = mx.expand_dims(pam[:, n_pf - 1], 1) if n_pf > 0 else mx.zeros((B, 1), dtype=mx.int32)
        time_before_val = (
            mx.expand_dims(tlb[:, n_pf], 1) if n_pf > 0 and n_pf < tlb.shape[1] else mx.zeros((B, 1), dtype=mx.int32)
        )
        time_after_val = (
            mx.expand_dims(tla[:, n_pf], 1) if n_pf > 0 and n_pf < tla.shape[1] else mx.zeros((B, 1), dtype=mx.int32)
        )
        neg_cond = mx.zeros((B, self.config.hidden_size))
        all_acoustic: list[mx.array] = []
        all_time_before: list[mx.array] = []
        all_token_ids: list[mx.array] = []
        last_time_before = None

        for i in range(n_pf):
            all_acoustic.append(mx.expand_dims(paf[:, i], 1))

        for i in range(n_pf):
            all_time_before.append(mx.expand_dims(tlb[:, i + 1], 1))

        max_steps = input_ids.shape[1] + num_extra_steps
        for step in range(prefill_len, max_steps):
            t0 = time.time()
            input_slice = input_ids[:, step : step + 1]
            is_structural = (
                (input_slice == self.config.start_header_id)
                | (input_slice == self.config.end_header_id)
                | (input_slice == self.config.eot_id)
            )
            neg_slice = mx.where(
                is_structural, input_slice, mx.full(input_slice.shape, pad_id, dtype=input_slice.dtype)
            )
            combined_slice = mx.concatenate([input_slice, neg_slice], axis=0)
            combined_acoustic = mx.concatenate([acoustic_features_val, acoustic_features_val], axis=0)
            combined_masks = mx.concatenate([acoustic_masks_val, acoustic_masks_val], axis=0)
            combined_tb = mx.concatenate([time_before_val, time_before_val], axis=0)
            combined_ta = mx.concatenate([time_after_val, time_after_val], axis=0)
            hidden, logits = self.forward_one_step(
                combined_slice,
                combined_acoustic,
                combined_masks,
                combined_tb,
                combined_ta,
                cache,
            )
            neg_cond = hidden[B : 2 * B]
            hidden = hidden[:B]
            logits = logits[:B]
            noise = mx.random.normal((B, self.config.acoustic_dim + self.time_dim)) * opts.noise_temperature
            speech = self.prediction_head.solve(
                noise,
                hidden,
                neg_cond,
                acoustic_dim=self.config.acoustic_dim,
                num_steps=opts.num_flow_matching_steps,
                acoustic_cfg_scale=opts.acoustic_cfg_scale,
                duration_cfg_scale=opts.duration_cfg_scale,
                cfg_schedule=opts.cfg_schedule,
                time_schedule=opts.time_schedule,
                bottleneck_fn=self.apply_bottleneck,
            )
            time_gray = speech[..., -self.time_dim :]
            pred_tb = mx.expand_dims(
                decode_gray_code_to_time(time_gray[..., : self.num_time_bits], self.num_time_bits), 0
            )
            pred_ta = mx.expand_dims(
                decode_gray_code_to_time(time_gray[..., self.num_time_bits :], self.num_time_bits), 0
            )

            if step >= input_ids.shape[1] - 1:
                next_token = self.sample_token(logits[:, -1, :], input_ids, opts, pad_id)
                input_ids = mx.concatenate([input_ids, next_token.astype(mx.int32)], axis=1)
                all_token_ids.append(next_token)
                if next_token[0, 0].item() in self.config.eos_token_id:
                    break
            else:
                all_token_ids.append(input_ids[:, step + 1 : step + 2])

            if step >= shift:
                if step - shift < paf.shape[1]:
                    acoustic_features_val = mx.expand_dims(paf[:, step - shift], 1)
                    acoustic_masks_val = mx.expand_dims(pam[:, step - shift], 1)
                else:
                    acoustic_features_val = mx.expand_dims(speech[..., : self.config.acoustic_dim], 0)
                    acoustic_masks_val = mx.ones((B, 1), dtype=mx.int32)
                all_acoustic.append(acoustic_features_val)
                if step - shift < tlb.shape[1] - 1:
                    time_before_val = mx.expand_dims(tlb[:, step - shift + 1], 1)
                    time_after_val = mx.expand_dims(tla[:, step - shift + 1], 1)
                else:
                    time_before_val = pred_tb
                    time_after_val = pred_ta
                all_time_before.append(time_before_val)
                last_time_before = time_before_val

            mx.eval(input_ids, acoustic_features_val, time_before_val)

            if log.isEnabledFor(logging.INFO):
                tok_str = self._tokenizer.decode([input_ids[0, -1].item()])
                log.info(f"step {step}: {tok_str!r}  ({(time.time() - t0) * 1000:.0f}ms)")

        if last_time_before is not None:
            all_time_before.append(last_time_before)

        return all_acoustic, all_time_before, all_token_ids

    def _decode_output(
        self,
        all_acoustic: list[mx.array],
        all_time_before: list[mx.array],
        num_prompt_tokens: int,
        num_transition_steps: int,
    ) -> mx.array:
        out_acoustic = (
            mx.concatenate([frame if frame.ndim == 3 else mx.expand_dims(frame, 1) for frame in all_acoustic], axis=1)
            if all_acoustic
            else mx.zeros((1, 0, self.config.acoustic_dim))
        )
        out_time = (
            mx.concatenate(
                [time_step if time_step.ndim == 2 else mx.expand_dims(time_step, 1) for time_step in all_time_before],
                axis=1,
            )
            if all_time_before
            else mx.zeros((1, 0), dtype=mx.int32)
        )
        acoustic_features = out_acoustic * self.config.acoustic_std + self.config.acoustic_mean
        encoded = acoustic_features[:, num_prompt_tokens + num_transition_steps - 1 :, :]
        time_before_out = out_time[:, num_prompt_tokens + num_transition_steps - 1 :]
        log.debug(f"decode: encoded shape={encoded.shape}, time_before_out shape={time_before_out.shape}")
        wav = self._decoder.decode_frames(encoded[0], time_before_out[0])
        lead_frames = int(time_before_out[0, 0].item())
        lead_samples = int(24000 * lead_frames / 50)

        if lead_samples > 0 and lead_samples < wav.shape[0]:
            wav = wav[lead_samples:]

        return wav

    def generate(
        self,
        text: str,
        reference: Reference,
        inference_options: InferenceOptions | None = None,
        num_transition_steps: int = 5,
        num_extra_steps: int = 0,
    ) -> GenerationOutput:
        """Generate speech audio from text, cloning the voice from a reference.

        Args:
            text: The text to speak.
            reference: A Reference from load_reference() or Reference.load().
            inference_options: Generation parameters (temperature, CFG scale,
                flow matching steps, etc.). Uses defaults if None.
            num_transition_steps: Number of reference frames to regenerate at
                the boundary between reference and generated speech. Default 5.

        Returns:
            A GenerationOutput with audio (numpy float32 at 24kHz), duration,
            real-time factor, and token count.
        """
        t_start = time.time()
        opts = inference_options or InferenceOptions()
        text = normalize_text(text)
        input_ids, paf, pam, tlb, tla, prefill_len = self._build_inputs(text, reference, num_transition_steps)
        log.info(f"target text: {text!r}")
        log.info(f"input tokens: {input_ids.shape[1]}, prompt frames: {paf.shape[1]}")
        cache = self._prefill(input_ids, paf, pam, tlb, tla, prefill_len)
        all_acoustic, all_time_before, all_token_ids = self._autoregressive_loop(
            input_ids,
            paf,
            pam,
            tlb,
            tla,
            prefill_len,
            cache,
            opts,
            num_extra_steps=num_extra_steps,
        )
        wav = self._decode_output(all_acoustic, all_time_before, paf.shape[1], num_transition_steps)
        elapsed = time.time() - t_start
        audio_np = np.array(wav, dtype=np.float32)
        duration = len(audio_np) / 24000
        rtf = elapsed / duration if duration > 0 else float("inf")
        log.info(f"generation: {elapsed:.2f}s, audio: {duration:.2f}s, RTF: {rtf:.2f}, tokens: {len(all_token_ids)}")
        return GenerationOutput(audio=audio_np, num_tokens=len(all_token_ids), duration=duration, rtf=rtf)
