import math
from typing import Literal

import torch
from dac.model.dac import Snake1d
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from .encoder import LocalAttentionEncoder, ResidualUnit, WNConv1d


def WNConvTranspose1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.ConvTranspose1d(*args, **kwargs))


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class DACDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def _create_segment_attention_mask(
    text_token_mask: torch.Tensor, version: Literal["v1", "v2", "decoder_block_attention"] = "v1"
) -> torch.Tensor:
    """
    Create an attention mask based on block boundaries marked in text_token_mask.

    Args:
        text_token_mask: (batch_size, seq_len) - binary mask where 1 indicates block boundaries
        version: Type of attention mask to create
            - "v1": Positions can attend to their own block and the next block (except last element)
            - "v2": Complex rules with marked positions having special access
            - "decoder_block_attention": Causal attention within blocks (positions attend to all past positions in same block)

    Returns:
        mask: (batch_size, seq_len, seq_len) - boolean mask where True means masked (cannot attend)
    """
    if version == "v1":
        """
        Decoder block attention with different rules for marked vs non-marked positions:

        - Marked positions (text_token_mask == 1): Only attend causally (j <= i)
        - Non-marked positions: Attend to all past positions (j < i) + current block up to
          and including the next marked position (j >= i in same block)

        Blocks are defined as segments ending at a marked position (inclusive).
        """
        # Compute block IDs so that each block includes positions UP TO the next marked position
        # Subtract text_token_mask so marked positions have the same block ID as preceding positions
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask  # (batch_size, seq_len)

        # Expand for broadcasting
        block_ids_i = block_ids.unsqueeze(2)  # (batch_size, seq_len, 1)
        block_ids_j = block_ids.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Position i can attend to position j if they're in the same block
        same_block = block_ids_i == block_ids_j  # (batch_size, seq_len, seq_len)

        # Create position masks
        batch_size, seq_len = text_token_mask.shape
        positions = torch.arange(seq_len, device=text_token_mask.device)
        pos_i = positions.unsqueeze(1)  # (seq_len, 1)
        pos_j = positions.unsqueeze(0)  # (1, seq_len)

        # Identify marked positions
        is_marked_i = text_token_mask.unsqueeze(2).bool()  # (batch_size, seq_len, 1)

        # For marked positions: only causal attention (j <= i)
        marked_causal = (pos_j <= pos_i).unsqueeze(0) & is_marked_i  # (batch_size, seq_len, seq_len)

        # For non-marked positions: all past (j < i) + current block forward (j >= i in same block)
        past = (pos_j < pos_i).unsqueeze(0)  # (batch_size, seq_len, seq_len)
        current_block_forward = (pos_j >= pos_i) & same_block  # (batch_size, seq_len, seq_len)
        non_marked_attention = (past | current_block_forward) & ~is_marked_i  # (batch_size, seq_len, seq_len)

        # Combine: marked positions use causal, non-marked use past + forward block
        can_attend = marked_causal | non_marked_attention  # (batch_size, seq_len, seq_len)

        # Return inverse (True = masked)
        mask = ~can_attend

        return mask
    elif version == "v2":
        """
        Decoder v2: Each position can attend to the current block and the previous block only.

        Blocks are defined by marked positions (text_token_mask == 1).
        Each marked position is the LAST position of its block (blocks end at marked positions).

        Attention rules:
        - Position i can attend to position j if:
          - block_ids[j] == block_ids[i] (same block), OR
          - block_ids[j] == block_ids[i] - 1 (previous block)
        """
        # Compute block IDs so that each block includes positions UP TO the next marked position
        # Subtract text_token_mask so marked positions have the same block ID as preceding positions
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask  # (batch_size, seq_len)

        # Expand for broadcasting
        block_ids_i = block_ids.unsqueeze(2)  # (batch_size, seq_len, 1)
        block_ids_j = block_ids.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Position i can attend to position j if:
        # - block_ids[j] == block_ids[i] (same block), OR
        # - block_ids[j] == block_ids[i] - 1 (previous block)
        same_block = block_ids_j == block_ids_i
        prev_block = block_ids_j == (block_ids_i - 1)
        can_attend = same_block | prev_block  # (batch_size, seq_len, seq_len)

        # Return inverse (True = masked)
        mask = ~can_attend

        return mask
    else:
        raise ValueError(f"Unknown version: {version}")


class DecoderConfig(PretrainedConfig):
    embed_dim: int = 512
    hidden_dim: int = 1024
    num_attn_layers: int = 6
    num_attn_heads: int = 8
    attn_dim_feedforward: int = 4096
    attn_dropout: float = 0.1
    use_flash_attn: bool = True
    wav_decoder_channels: int = 1536
    strides: list[int] = [4, 4, 5, 6]
    block_attention: Literal["none", "v1", "v2"] = "v2"


class Decoder(PreTrainedModel):
    config_class = DecoderConfig

    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.decoder_proj = nn.Linear(self.config.embed_dim, self.config.hidden_dim)

        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=self.config.hidden_dim,
            num_layers=self.config.num_attn_layers,
            num_heads=self.config.num_attn_heads,
            d_ff=self.config.attn_dim_feedforward,
            dropout=self.config.attn_dropout,
            activation="gelu",
            max_seq_len=8192,
            use_flash_attn=self.config.use_flash_attn,
        )
        self.wav_decoder = DACDecoder(
            input_channel=self.config.hidden_dim,
            channels=self.config.wav_decoder_channels,
            rates=self.config.strides,
        )

    def forward(self, encoded_expanded: torch.Tensor, token_masks: torch.Tensor):
        decoder_input = self.decoder_proj(encoded_expanded)
        # Apply decoder block attention if text_token_mask is provided
        attn_mask = _create_segment_attention_mask(token_masks, version="v2")
        decoded_expanded = self.local_attention_decoder(decoder_input, mask=attn_mask)

        x_rec = self.wav_decoder(decoded_expanded.transpose(1, 2))
        return x_rec

    def generate(self, encoded_expanded: torch.Tensor, **kwargs):
        return self.forward(encoded_expanded, **kwargs)

class StreamingDecoder:
    """Incremental block-by-block decoder with KV-cache for streaming audio generation.

    Uses per-layer KV caching for the transformer (bit-exact with non-streaming)
    and a sliding-window CNN for audio synthesis (near-exact, bounded memory).

    Usage:
        streaming_decoder = StreamingDecoder(decoder)
        streaming_decoder.skip_leading_frames(leading_silence)  # optional
        for token_embedding, time_before in token_stream:
            audio_chunk = streaming_decoder.decode_block(token_embedding, time_before)
            if audio_chunk is not None and audio_chunk.numel() > 0:
                play(audio_chunk)
        final_chunk = streaming_decoder.flush(trailing_frames=last_time_before)
    """

    def __init__(
        self,
        decoder: Decoder,
        min_block_frames: int = 3,
        max_cached_frames: int | None = 500,
        cnn_window_size: int = 100,
        cnn_left_context: int = 20,
        cnn_lookahead: int = 15,
        min_first_emission: int = 50,
    ):
        """
        Args:
            decoder: The pretrained Decoder model to wrap.
            min_block_frames: Minimum frames before decoding a block. Very short
                blocks (1-2 frames) can cause degenerate CNN behavior, so we
                buffer them until we accumulate enough frames.
            max_cached_frames: Maximum frames in the KV-cache. Oldest blocks are
                evicted when exceeded. None = unlimited (bit-exact). Default 500.
            cnn_window_size: Fixed CNN input size in frames. Layout:
                [left_context | emittable | lookahead]. Default 100.
                Use 50 for CPU/edge devices with <32GB RAM.
            cnn_left_context: Left context frames for CNN stability.
                Empirically: 20 → 0.000000 diff. 10 → 0.0003 diff.
            cnn_lookahead: Right context frames held back from CNN output.
                Empirically: 15 → 0.000001 diff (pretrained weights need 15).
            min_first_emission: Minimum accumulated frames before first CNN call.
                Avoids edge artifacts from processing very short sequences.
                Default 50 adds ~1s to TTFA.
        """
        self.decoder = decoder
        self.min_block_frames = min_block_frames
        self.max_cached_frames = max_cached_frames
        self._cnn_window_size = cnn_window_size
        self._cnn_left_context = cnn_left_context
        self._lookahead_frames = cnn_lookahead
        self._min_first_emission = min_first_emission

        # Precompute samples per frame (480 for strides [4,4,5,6])
        self._samples_per_frame = 1
        for s in self.decoder.config.strides:
            self._samples_per_frame *= s

        self._init_state()

    def _init_state(self):
        """Initialize/reset all streaming state."""
        # Per-layer KV cache: list of (cached_k, cached_v) per layer
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._cached_frames: int = 0  # absolute frame count (for RoPE position)
        self._all_token_masks: torch.Tensor | None = None  # accumulated masks for attention
        self._block_frame_counts: list[int] = []  # frames per block (for eviction)

        # Token buffer (accumulate small blocks before decoding)
        self._buffered_tokens: list[torch.Tensor] = []
        self._buffered_times: list[int] = []

        # Sliding window CNN state
        self._all_hidden: torch.Tensor | None = None  # recent transformer outputs (capped)
        self._hidden_offset: int = 0  # absolute frame index of _all_hidden[0]
        self._emitted_frames: int = 0  # absolute frame index up to which audio has been emitted
        self._first_emitted: bool = False  # gates min_first_emission check

    def skip_leading_frames(self, n_frames: int):
        """Skip the first n_frames of audio output (e.g., leading silence).

        Must be called before any decode_block() calls.
        """
        self._emitted_frames = n_frames

    def reset(self):
        """Clear all state for a new utterance."""
        self._init_state()

    def _expand_block(
        self,
        token_embeddings: list[torch.Tensor],
        time_before_values: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand token embeddings into dense frames.

        For each token, inserts (time_before - 1) zero frames before the embedding.

        Returns:
            frames: (1, num_frames, embed_dim)
            masks: (1, num_frames) — 1 at token positions, 0 at zero-fill
        """
        embed_dim = token_embeddings[0].shape[-1]
        parts = []
        mask_parts = []

        for token_emb, t_before in zip(token_embeddings, time_before_values):
            n_zeros = max(0, t_before - 1)
            if n_zeros > 0:
                parts.append(torch.zeros(n_zeros, embed_dim, device=device, dtype=dtype))
                mask_parts.append(torch.zeros(n_zeros, device=device, dtype=torch.long))
            parts.append(token_emb.unsqueeze(0).to(device=device, dtype=dtype))
            mask_parts.append(torch.ones(1, device=device, dtype=torch.long))

        frames = torch.cat(parts, dim=0).unsqueeze(0)
        masks = torch.cat(mask_parts, dim=0).unsqueeze(0)
        return frames, masks

    def _build_kv_cache_mask(
        self, all_token_masks: torch.Tensor, num_new_frames: int
    ) -> torch.Tensor:
        """Build v2 attention mask for KV-cache mode.

        Returns only the rows for new frames: (batch, num_new_frames, total_frames).
        Block ID offsets from eviction cancel out in the v2 equality checks.
        """
        full_mask = _create_segment_attention_mask(all_token_masks, version="v2")
        return full_mask[:, -num_new_frames:, :]

    def _evict_cache(self):
        """Evict oldest blocks from KV-cache if over max_cached_frames.

        Eviction happens at block boundaries to preserve correct block IDs.
        Keeps at least 2 blocks (for v2 attention: current + prev).
        """
        if self.max_cached_frames is None or not self._kv_cache:
            return

        cache_len = self._kv_cache[0][0].shape[2]
        if cache_len <= self.max_cached_frames:
            return

        frames_to_evict = 0
        blocks_to_evict = 0
        remaining = cache_len
        while (blocks_to_evict < len(self._block_frame_counts) - 2
               and remaining - self._block_frame_counts[blocks_to_evict] > self.max_cached_frames):
            frames_to_evict += self._block_frame_counts[blocks_to_evict]
            remaining -= self._block_frame_counts[blocks_to_evict]
            blocks_to_evict += 1

        if frames_to_evict == 0:
            return

        for i, (cached_k, cached_v) in enumerate(self._kv_cache):
            self._kv_cache[i] = (
                cached_k[:, :, frames_to_evict:, :],
                cached_v[:, :, frames_to_evict:, :],
            )

        if self._all_token_masks is not None:
            self._all_token_masks = self._all_token_masks[:, frames_to_evict:]

        self._block_frame_counts = self._block_frame_counts[blocks_to_evict:]
        # Note: _cached_frames stays as absolute position (for RoPE)

    @torch.no_grad()
    def decode_block(
        self,
        token_embedding: torch.Tensor,
        time_before: int,
    ) -> torch.Tensor | None:
        """Decode a single block incrementally. Returns audio chunk or None if buffering."""
        self._buffered_tokens.append(token_embedding)
        self._buffered_times.append(time_before)

        total_frames = sum(max(0, t - 1) for t in self._buffered_times) + len(self._buffered_tokens)
        if total_frames < self.min_block_frames:
            return None

        return self._decode_buffered()

    def _decode_buffered(self) -> torch.Tensor:
        """Decode buffered tokens: transformer (KV-cache) → CNN (sliding window)."""
        device = next(self.decoder.parameters()).device
        decoder_dtype = next(self.decoder.parameters()).dtype

        current_input, current_masks = self._expand_block(
            self._buffered_tokens, self._buffered_times,
            device=device, dtype=decoder_dtype,
        )
        current_frames = current_input.shape[1]
        self._buffered_tokens = []
        self._buffered_times = []

        # Transformer with KV-cache (bit-exact)
        current_hidden = self._transformer_forward_cached(current_input, current_masks)

        # Accumulate hidden states
        if self._all_hidden is not None:
            self._all_hidden = torch.cat([self._all_hidden, current_hidden], dim=1)
        else:
            self._all_hidden = current_hidden

        # Buffer until min_first_emission
        if not self._first_emitted and self._all_hidden.shape[1] < self._min_first_emission + self._lookahead_frames:
            self._block_frame_counts.append(current_frames)
            return torch.zeros(1, 0, device=device, dtype=decoder_dtype).squeeze(0)

        # Run CNN and emit safe audio
        audio = self._cnn_emit(is_flush=False)

        self._block_frame_counts.append(current_frames)
        return audio

    def _cnn_emit(self, is_flush: bool = False) -> torch.Tensor:
        """Run CNN on sliding window and emit the safe audio region.

        Shared by _decode_buffered() and flush(). The only difference is
        flush() disables the lookahead holdback (emits everything).

        Coordinate systems:
        - "absolute": frame index since start of utterance (_emitted_frames)
        - "buffer": index into _all_hidden (0-based, shifts on trim)
        - _hidden_offset converts: absolute = buffer + _hidden_offset
        """
        buf_frames = self._all_hidden.shape[1]
        device = self._all_hidden.device
        dtype = self._all_hidden.dtype
        spf = self._samples_per_frame

        # Determine sliding window position (buffer-relative).
        # Constraint: window must not slide so far that _emitted_frames
        # falls inside the left context zone (would skip unemitted frames).
        max_start_for_emit = max(0, self._emitted_frames - self._hidden_offset - self._cnn_left_context)
        win_buf_start = min(max(0, buf_frames - self._cnn_window_size), max_start_for_emit)

        window = self._all_hidden[:, win_buf_start:, :]
        window_frames = window.shape[1]

        # Run CNN
        audio = self.decoder.wav_decoder(window.transpose(1, 2))

        # Convert to absolute coordinates
        win_abs_start = self._hidden_offset + win_buf_start

        # Safe emit region
        left_ctx = self._cnn_left_context if win_abs_start > 0 else 0
        abs_safe_start = win_abs_start + left_ctx
        lookahead = 0 if is_flush else self._lookahead_frames
        abs_safe_end = win_abs_start + window_frames - lookahead

        # Only emit frames we haven't emitted yet
        emit_start = max(abs_safe_start, self._emitted_frames)
        emit_end = abs_safe_end

        if emit_end <= emit_start:
            return torch.zeros(1, 0, device=device, dtype=dtype).squeeze(0)

        # Map to window-local sample positions
        win_start_sample = (emit_start - win_abs_start) * spf
        win_end_sample = min((emit_end - win_abs_start) * spf, audio.shape[2])

        emit_audio = audio[:, :, win_start_sample:win_end_sample]
        self._emitted_frames = emit_end
        self._first_emitted = True

        # Trim _all_hidden to cap memory
        if buf_frames > self._cnn_window_size:
            trim = buf_frames - self._cnn_window_size
            self._all_hidden = self._all_hidden[:, trim:, :]
            self._hidden_offset += trim

        return emit_audio.squeeze(0)

    def _transformer_forward_cached(
        self, current_input: torch.Tensor, current_masks: torch.Tensor
    ) -> torch.Tensor:
        """Run transformer layers with KV-cache, returning hidden states for new frames."""
        new_frames = current_input.shape[1]

        # Accumulate token masks
        if self._all_token_masks is not None:
            all_masks = torch.cat([self._all_token_masks, current_masks], dim=1)
        else:
            all_masks = current_masks

        # Build attention mask (new frames attend to all cached + new frames)
        attn_mask = self._build_kv_cache_mask(all_masks, new_frames)

        # Project to hidden dim
        decoder_input = self.decoder.decoder_proj(current_input)
        x = self.decoder.local_attention_decoder.input_proj(decoder_input)

        # Iterate through layers with per-layer KV-cache
        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.decoder.local_attention_decoder.layers):
            cached_k, cached_v = self._kv_cache[i] if self._kv_cache else (None, None)

            x, new_k, new_v = layer.forward_with_cache(
                x, cached_k, cached_v,
                position_offset=self._cached_frames,
                mask=attn_mask,
            )

            if cached_k is not None:
                updated_k = torch.cat([cached_k, new_k], dim=2)
                updated_v = torch.cat([cached_v, new_v], dim=2)
            else:
                updated_k, updated_v = new_k, new_v

            new_kv_cache.append((updated_k, updated_v))

        x = self.decoder.local_attention_decoder.final_norm(x)

        # Update state
        self._kv_cache = new_kv_cache
        self._cached_frames += new_frames
        self._all_token_masks = all_masks
        self._evict_cache()

        return x

    @torch.no_grad()
    def flush(self, trailing_frames: int = 0) -> torch.Tensor | None:
        """Flush remaining buffered tokens and trailing silence.

        Emits all remaining audio with no lookahead holdback.
        """
        device = next(self.decoder.parameters()).device
        decoder_dtype = next(self.decoder.parameters()).dtype

        has_buffered = len(self._buffered_tokens) > 0
        has_trailing = trailing_frames > 0

        if not has_buffered and not has_trailing and self._all_hidden is None:
            return None

        audio_parts = []

        # Decode any buffered tokens first
        if has_buffered:
            audio_parts.append(self._decode_buffered())

        # Process trailing silence through transformer
        if has_trailing and self._all_hidden is not None:
            embed_dim = self.decoder.config.embed_dim
            trailing_input = torch.zeros(1, trailing_frames, embed_dim, device=device, dtype=decoder_dtype)
            trailing_masks = torch.zeros(1, trailing_frames, device=device, dtype=torch.long)
            trailing_hidden = self._transformer_forward_cached(trailing_input, trailing_masks)
            self._all_hidden = torch.cat([self._all_hidden, trailing_hidden], dim=1)

        # Emit everything remaining (no lookahead)
        if self._all_hidden is not None:
            audio_parts.append(self._cnn_emit(is_flush=True))

        # Filter empty tensors
        audio_parts = [a for a in audio_parts if a.numel() > 0]

        if not audio_parts:
            return None
        return torch.cat(audio_parts, dim=-1)
