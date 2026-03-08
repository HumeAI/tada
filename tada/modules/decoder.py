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
