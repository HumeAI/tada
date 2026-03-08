import math
from dataclasses import dataclass
from typing import Literal

import torch
import torchaudio
from dac.nn.layers import Snake1d
from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

from ..utils.text import normalize_text
from .aligner import Aligner, AlignerConfig


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(*args, **kwargs))


class ResidualUnit(torch.nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = torch.nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


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
        # Compute block IDs: each '1' starts a new block
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask  # (batch_size, seq_len)

        # Expand for broadcasting
        block_ids_i = block_ids.unsqueeze(2)  # (batch_size, seq_len, 1)
        block_ids_j = block_ids.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Position i can attend to position j if:
        # - block_ids[j] == block_ids[i] (same block), OR
        # - block_ids[j] == block_ids[i] + 1 (next block)

        same_block = block_ids_j == block_ids_i
        block_ids_j_excluding_last = torch.where(text_token_mask.bool(), -10, block_ids_j[:, 0, :]).unsqueeze(1)
        next_block = block_ids_j_excluding_last == (block_ids_i + 1)
        can_attend = same_block | next_block  # (batch_size, seq_len, seq_len)

        # Return inverse (True = masked)
        mask = ~can_attend

        return mask
    elif version == "v2":
        """
        NEW v2: Block-wise attention with special rules for marked positions.
        Blocks are segments between consecutive marked positions (text_token_mask == 1).

        If p1, p2 are consecutive marked positions, the block is defined as [p1, ..., p2-1],
        meaning everything from p1 to just before p2 is in the same block.

        Attention rules:
        1. Non-marked positions: Can attend to other non-marked positions in their block only.
           They CANNOT attend to the marked position that starts their block.
        2. Marked positions: Can attend to their entire current block (including themselves)
           AND the previous block, but EXCLUDING marked positions in the previous block.

        This creates information flow from previous blocks into current blocks through marked positions,
        while keeping marked positions isolated (they cannot be attended to except by other marked
        positions in the same block).
        """
        # Compute block IDs using cumsum
        # Each marked position (text_token_mask == 1) starts a new block
        block_ids = torch.cumsum(text_token_mask, dim=1)  # (batch_size, seq_len)

        # Expand for broadcasting: (batch, seq_len, 1) and (batch, 1, seq_len)
        block_ids_i = block_ids.unsqueeze(2)  # (batch_size, seq_len, 1)
        block_ids_j = block_ids.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Position i can attend to position j if they have the same block ID
        same_block = block_ids_i == block_ids_j  # (batch_size, seq_len, seq_len)

        # Exclude marked positions from being attended to within their block
        # (except by other marked positions in the same block)
        is_marked_i = text_token_mask.unsqueeze(2).bool()  # (batch, seq_len, 1) - positions attending FROM
        is_marked_j = text_token_mask.unsqueeze(1).bool()  # (batch, 1, seq_len) - positions attending TO

        # Same block attention: exclude marked positions unless you're also marked AND in same block
        same_block_valid = same_block & (~is_marked_j | (is_marked_i & same_block))  # (batch_size, seq_len, seq_len)

        # Positions in previous block: block_ids_j == block_ids_i - 1
        prev_block = block_ids_j == (block_ids_i - 1)  # (batch_size, seq_len, seq_len)

        # Marked positions can also attend to previous block, but exclude marked positions in prev block
        prev_block_valid = prev_block & ~is_marked_j  # (batch_size, seq_len, seq_len)

        # Marked positions can attend to same block OR previous block (both excluding marked positions appropriately)
        can_attend = same_block_valid | (is_marked_i & prev_block_valid)  # (batch_size, seq_len, seq_len)

        # Return inverse (True = masked)
        mask = ~can_attend

        return mask
    else:
        raise ValueError(f"Unknown version: {version}")


class EncoderBlock(torch.nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = torch.nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class WavEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = torch.nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class LocalSelfAttention(torch.nn.Module):
    """Local self-attention with limited receptive field and RoPE (Rotary Position Embeddings)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        causal: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.use_flash_attn = use_flash_attn

        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

        # Precompute RoPE frequencies
        self.register_buffer("rope_freqs", self._compute_rope_freqs(self.head_dim, max_seq_len))

        # Precompute attention mask once for maximum sequence length
        self.max_seq_len = max_seq_len
        self.register_buffer("_precomputed_mask", self._create_full_mask(max_seq_len))

    def _create_full_mask(self, max_seq_len: int) -> torch.Tensor:
        """Precompute the full attention mask for maximum sequence length."""
        # Create position indices
        positions = torch.arange(max_seq_len).unsqueeze(0)  # (1, max_seq_len)
        positions_t = positions.transpose(0, 1)  # (max_seq_len, 1)

        # Compute distances between all positions using broadcasting
        distances = positions - positions_t  # (max_seq_len, max_seq_len)

        mask = distances.abs() >= 0

        return mask

    def _compute_rope_freqs(self, head_dim: int, max_seq_len: int) -> torch.Tensor:
        """Precompute RoPE rotation frequencies."""
        # Compute frequencies for each dimension pair
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Create position indices
        positions = torch.arange(max_seq_len).float()

        # Compute outer product: (max_seq_len, head_dim // 2)
        freqs = torch.outer(positions, inv_freq)

        # Create rotation matrix: (max_seq_len, head_dim)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()

        return torch.stack([freqs_cos, freqs_sin], dim=-1)  # (max_seq_len, head_dim // 2, 2)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        Args:
            x: (batch, num_heads, seq_len, head_dim)
            seq_len: sequence length
        Returns:
            rotated x with same shape
        """
        batch, num_heads, seq_len, head_dim = x.shape

        # Get precomputed frequencies for this sequence length
        freqs = self.rope_freqs[:seq_len]  # (seq_len, head_dim // 2, 2)
        freqs_cos = freqs[..., 0]  # (seq_len, head_dim // 2)
        freqs_sin = freqs[..., 1]  # (seq_len, head_dim // 2)

        # Reshape x into pairs: (batch, num_heads, seq_len, head_dim // 2, 2)
        x_reshaped = x.reshape(batch, num_heads, seq_len, head_dim // 2, 2)

        # Apply rotation: [cos, -sin; sin, cos] @ [x0; x1]
        x0 = x_reshaped[..., 0]  # (batch, num_heads, seq_len, head_dim // 2)
        x1 = x_reshaped[..., 1]  # (batch, num_heads, seq_len, head_dim // 2)

        # Rotate: x_new = [x0 * cos - x1 * sin, x0 * sin + x1 * cos]
        x_rotated_0 = x0 * freqs_cos.unsqueeze(0).unsqueeze(0) - x1 * freqs_sin.unsqueeze(0).unsqueeze(0)
        x_rotated_1 = x0 * freqs_sin.unsqueeze(0).unsqueeze(0) + x1 * freqs_cos.unsqueeze(0).unsqueeze(0)

        # Stack back together
        x_rotated = torch.stack([x_rotated_0, x_rotated_1], dim=-1)

        # Reshape back to original: (batch, num_heads, seq_len, head_dim)
        return x_rotated.reshape(batch, num_heads, seq_len, head_dim)

    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return local attention mask by slicing the precomputed mask."""
        # Simply return the slice of precomputed mask for the current sequence length
        # The mask is already on the correct device as a buffer
        return self._precomputed_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (batch, num_heads, seq_len, seq_len)

        # Apply local attention mask
        if mask is None:
            local_mask = self.create_local_mask(seq_len, x.device)
            # local_mask has shape (seq_len, seq_len), broadcast to all batches and heads
            attn_scores = attn_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            # Handle batch-wise masks: (batch, seq_len, seq_len)
            if mask.dim() == 2:
                # Mask is (seq_len, seq_len), broadcast to all batches and heads
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            elif mask.dim() == 3:
                # Mask is (batch, seq_len, seq_len), broadcast to all heads
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float("-inf"))
            else:
                raise ValueError(f"Mask should have 2 or 3 dimensions, got {mask.dim()}")

        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.layer_norm(x + output)

        return output


class LocalAttentionEncoderLayer(torch.nn.Module):
    """Transformer encoder layer with local self-attention and feed-forward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 8192,
        window_size: int = 1500,  # 30s
        causal: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        # Default FFN dimension is 4x the model dimension (standard Transformer)
        if d_ff is None:
            d_ff = 4 * d_model

        # Local self-attention with RoPE
        self.self_attn = LocalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            causal=causal,
            use_flash_attn=use_flash_attn,
        )

        # Feed-forward network
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU() if activation == "gelu" else torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )

        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Self-attention block (already includes residual + norm)
        x = self.self_attn(x, mask=mask)

        # Feed-forward block with residual connection and layer norm
        x = self.norm(x + self.ffn(x))

        return x


class LocalAttentionEncoder(torch.nn.Module):
    """Stack of local attention encoder layers."""

    def __init__(
        self,
        d_model: int,
        d_input: int | None = None,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 8192,
        causal: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                LocalAttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    max_seq_len=max_seq_len,
                    causal=causal,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = torch.nn.LayerNorm(d_model)

        if d_input is not None and d_input != d_model:
            self.input_proj: torch.nn.Module = torch.nn.Linear(d_input, d_model)
        else:
            self.input_proj: torch.nn.Module = torch.nn.Identity()

    def _forward_window(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for a single window (non-sliding).

        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.final_norm(x)

    def _forward_sliding_window(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        inference_window_size: int | None = None,
        inference_window_stride: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using sliding window inference.

        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        window_size = int(inference_window_size * 50)
        stride = int(inference_window_stride * 50)

        # If sequence fits in one window, use regular forward
        if seq_len <= window_size:
            return self._forward_window(x, mask)

        # Prepare output tensor
        output = torch.zeros(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)

        # Calculate overlap size
        overlap_size = window_size - stride

        # Slide through the sequence
        prev_end_idx = 0
        for start_idx in range(0, seq_len, stride):
            end_idx = min(start_idx + window_size, seq_len)

            # Extract window
            x_window = x[:, start_idx:end_idx, :]

            # Extract mask window if provided
            mask_window = None
            if mask is not None:
                if mask.dim() == 2:
                    mask_window = mask[start_idx:end_idx, start_idx:end_idx]
                elif mask.dim() == 3:
                    mask_window = mask[:, start_idx:end_idx, start_idx:end_idx]

            # Process window
            output_window = self._forward_window(x_window, mask_window)

            if start_idx == 0:
                # First window: use everything
                output[:, start_idx:end_idx, :] = output_window
            else:
                # Calculate overlap region
                overlap_start = start_idx
                overlap_end = min(prev_end_idx, end_idx)
                overlap_length = overlap_end - overlap_start

                if overlap_length > 0:
                    # Use half from each: first half from previous window (already in output),
                    # second half from current window
                    mid_point = overlap_start + overlap_length // 2

                    # Keep first half from previous window (already in output)
                    # Overwrite second half with current window
                    window_offset = mid_point - start_idx
                    output[:, mid_point:overlap_end, :] = output_window[:, window_offset:overlap_length, :]

                # Copy non-overlapping part from current window
                if overlap_end < end_idx:
                    window_offset = overlap_end - start_idx
                    output[:, overlap_end:end_idx, :] = output_window[:, window_offset:, :]

            prev_end_idx = end_idx

            # If we've reached the end, break
            if end_idx >= seq_len:
                break

        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        inference_window_size: int | None = None,
        inference_window_stride: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Use sliding window inference if configured and not in training mode
        if inference_window_size is not None and not self.training:
            return self._forward_sliding_window(x, mask, inference_window_size, inference_window_stride)
        else:
            return self._forward_window(x, mask)


class EncoderConfig(PretrainedConfig):
    hidden_dim: int = 1024
    embed_dim: int = 512
    strides: list[int] = [6, 5, 4, 4]
    num_attn_layers: int = 6
    num_attn_heads: int = 8
    attn_dim_feedforward: int = 4096
    attn_dropout: float = 0.1
    dist_type: Literal["fixed", "gaussian"] = "fixed"
    block_attention: Literal["none", "v1", "v2"] = "v2"
    num_frames_per_second: int = 50
    std: float = 0.5
    acoustic_mean: float = 0.0
    acoustic_std: float = 1.5


@dataclass
class EncoderOutput:
    audio: torch.Tensor
    audio_len: torch.Tensor
    text: list[str]
    token_positions: torch.Tensor
    token_values: torch.Tensor
    sample_rate: int = 24000
    text_tokens: torch.Tensor | None = None
    text_tokens_len: torch.Tensor | None = None
    encoded_expanded: torch.Tensor | None = None
    non_sampled_encoded_expanded: torch.Tensor | None = None
    text_emb_expanded: torch.Tensor | None = None
    token_masks: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    text_emb_reduced: torch.Tensor | None = None

    @classmethod
    def empty(cls, device: torch.device, token_dim: int = 512) -> "EncoderOutput":
        return cls(
            audio=torch.zeros(1, 0, device=device),
            audio_len=torch.zeros(1, device=device),
            text=[""],
            text_tokens=torch.zeros(1, 0, dtype=torch.long, device=device),
            text_tokens_len=torch.zeros(1, dtype=torch.long, device=device),
            token_positions=torch.zeros([1, 0], dtype=torch.long, device=device),
            token_values=torch.zeros([1, 0, token_dim], device=device),
        )


class Encoder(PreTrainedModel):
    config_class = EncoderConfig

    def __init__(
        self,
        config: EncoderConfig,
    ):
        super().__init__(config)
        self.wav_encoder = WavEncoder(d_model=64, strides=config.strides, d_latent=config.hidden_dim)

        self.local_attention_encoder = LocalAttentionEncoder(
            d_model=config.hidden_dim,
            num_layers=config.num_attn_layers,
            num_heads=config.num_attn_heads,
            d_ff=config.attn_dim_feedforward,
            dropout=config.attn_dropout,
            activation="gelu",
            max_seq_len=8192,
        )

        if config.hidden_dim != config.embed_dim:
            self.hidden_linear = torch.nn.Linear(config.hidden_dim, config.embed_dim)
        else:
            self.hidden_linear = torch.nn.Identity()

        self.pos_emb = torch.nn.Embedding(2, config.hidden_dim)

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self.aligner.tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: str = "encoder", language: str | None = None):
        self = super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        aligner_subfolder = f"aligner-{language}" if language else "aligner"
        self._aligner = Aligner.from_pretrained(pretrained_model_name_or_path, subfolder=aligner_subfolder)
        return self

    def to(self, device: str):
        self._aligner.to(device)
        return super().to(device)

    @property
    def aligner(self) -> Aligner:
        if not hasattr(self, "_aligner"):
            self._aligner = Aligner(AlignerConfig())
        return self._aligner

    def get_encoder_outputs(
        self,
        audio: torch.Tensor,
        token_masks: torch.Tensor,
        inference_window_size: float | None = None,
        inference_window_stride: float | None = None,
    ) -> torch.Tensor:
        enc_out = self.wav_encoder(torch.nn.functional.pad(audio.unsqueeze(1), (0, 960), value=0)).transpose(1, 2)
        seq_len = enc_out.shape[1]
        padded_token_masks = torch.nn.functional.pad(token_masks, (0, seq_len - token_masks.shape[1]), value=0)
        # Apply local attention encoder with segment-based attention mask
        enc_out = enc_out + self.pos_emb(padded_token_masks)

        # Create attention mask based on text token boundaries
        attn_mask = _create_segment_attention_mask(
            padded_token_masks, version=self.config.block_attention
        )  # (batch, seq_len, seq_len)
        enc_out = self.local_attention_encoder(
            enc_out,
            mask=attn_mask,
            inference_window_size=inference_window_size,
            inference_window_stride=inference_window_stride,
        )
        enc_out = self.hidden_linear(enc_out)

        return enc_out, padded_token_masks

    def sample(self, x: torch.Tensor, dist_type: str | None = None) -> torch.Tensor:
        if dist_type is None:
            dist_type = self.config.dist_type
        if dist_type == "fixed":
            return x + torch.randn_like(x) * self.config.std
        elif dist_type == "gaussian":
            std = torch.randn(*x.shape[:-1], device=x.device) * self.config.std / 0.8
            return x + std.unsqueeze(-1) * torch.randn_like(x)
        elif dist_type == "mean_std":
            mean = x[..., : x.shape[-1] // 2]
            std = x[..., x.shape[-1] // 2 :]
            y = mean + std * torch.randn_like(mean)
            return torch.cat([y, torch.zeros_like(y)], dim=-1)
        else:
            raise ValueError(f"Invalid distribution type: {dist_type}")

    @property
    def parakeet_ctc_1_1b(self):
        if not hasattr(self, "_parakeet_ctc_1_1b"):
            self._parakeet_ctc_1_1b = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
            for param in self._parakeet_ctc_1_1b.parameters():
                param.requires_grad = False
            self._parakeet_ctc_1_1b.to(next(self.parameters()).device)
        return self._parakeet_ctc_1_1b

    @property
    def parakeet_ctc_1_1b_processor(self):
        if not hasattr(self, "_parakeet_ctc_1_1b_processor"):
            self._parakeet_ctc_1_1b_processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
        return self._parakeet_ctc_1_1b_processor

    def transcribe_audio_parakeet(self, audio_16k: list[torch.Tensor]) -> tuple[list[str]]:
        asr_inputs = self.parakeet_ctc_1_1b_processor(audio_16k, sampling_rate=16000, return_tensors="pt")
        asr_inputs = {k: v.to(next(self.parameters()).device) for k, v in asr_inputs.items()}
        asr_outputs = self.parakeet_ctc_1_1b.generate(**asr_inputs)
        transcripts = self.parakeet_ctc_1_1b_processor.batch_decode(asr_outputs)
        transcripts = [t.strip() for t in transcripts]
        transcripts = [f"{t}." for t in transcripts if t[-1].isalnum()]
        return transcripts

    def forward(
        self,
        audio: torch.Tensor,
        text: list[str] | str | None = None,
        text_tokens: torch.Tensor | None = None,
        text_token_len: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
        token_masks: torch.Tensor | None = None,
        audio_length: torch.Tensor | None = None,
        sample_rate: int = 24000,
        sample: bool = True,
        inference_window_size: float | None = None,
        inference_window_stride: float | None = None,
    ):
        if isinstance(text, str):
            text = [text]
        # self.latent_dropout_layer.training = sample
        if audio_length is None:
            audio_length = torch.tensor([x.shape[-1] for x in audio], device=audio.device)
        if sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, sample_rate, 24000)
            audio_length = audio_length * 24000 / sample_rate
            sample_rate = 24000

        x = audio
        device = x.device

        if text is None and text_tokens is None:
            audio_16k = [
                torchaudio.functional.resample(a[..., : al.long()], sample_rate, 16000)
                for a, al in zip(audio, audio_length)
            ]
            text = self.transcribe_audio_parakeet(audio_16k)

        if text_tokens is None or text_token_len is None:
            text = [normalize_text(t) for t in text]
            text_tokens = [
                self.aligner.tokenizer.encode(t, add_special_tokens=False, return_tensors="pt") for t in text
            ]
            text_token_len = torch.tensor([t.shape[-1] for t in text_tokens], device=device)
            text_tokens = torch.nn.utils.rnn.pad_sequence(
                text_tokens, batch_first=True, padding_value=self.aligner.tokenizer.eos_token_id
            ).to(device)

        if token_positions is None or token_masks is None:
            align_output = self.aligner(
                audio,
                text_tokens=text_tokens,
                audio_length=audio_length,
                inference_window_size=inference_window_size,
                inference_window_stride=inference_window_stride,
            )
            token_positions = align_output.token_positions
            token_masks = align_output.token_masks

        enc_out, token_masks = self.get_encoder_outputs(x, token_masks, inference_window_size, inference_window_stride)

        # encoded = torch.gather(enc_out, 1, all_selected_positions.unsqueeze(-1).expand(-1, -1, enc_out.shape[-1]))
        encoded_expanded = torch.where(token_masks.unsqueeze(-1) == 0, torch.zeros_like(enc_out), enc_out)
        encoded_expanded = encoded_expanded.to(audio.dtype).to(encoded_expanded.dtype)

        non_sampled_encoded_expanded = encoded_expanded

        if self.config.std > 0.0 and sample:
            encoded_expanded = torch.where(
                token_masks.unsqueeze(-1) == 0,
                encoded_expanded,
                self.sample(encoded_expanded, dist_type=self.config.dist_type),
            )

        token_values = torch.gather(
            encoded_expanded,
            1,
            (token_positions - 1).clamp(min=0).unsqueeze(-1).expand(-1, -1, encoded_expanded.shape[-1]),
        )
        token_values = (token_values - self.config.acoustic_mean) / self.config.acoustic_std

        return EncoderOutput(
            audio=audio,
            audio_len=audio_length,
            text=text,
            text_tokens=text_tokens.squeeze(1),
            text_tokens_len=text_token_len,
            encoded_expanded=encoded_expanded,
            non_sampled_encoded_expanded=non_sampled_encoded_expanded,
            token_positions=token_positions,
            token_masks=token_masks,
            token_values=token_values,
        )
