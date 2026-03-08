"""
HuggingFace-compatible Acoustic Speaker Verification model.

Maps 512-d acoustic features (continuous tokens) to 192-d L2-normalized speaker embeddings.
Trained via cosine similarity regression against ground truth speaker embeddings.

Usage:
    from tada.modules.acoustic_spkr_verf import AcousticSpkrVerf
    model = AcousticSpkrVerf.from_pretrained("HumeAI/tada-codec", subfolder="spkr-verf")
    emb = model(acoustic_features)       # [B, 512] -> [B, 192], L2-normalized
    sim = model.similarity(emb1, emb2)   # cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class AcousticSpkrVerfConfig(PretrainedConfig):
    model_type = "acoustic_spkr_verf"

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 768,
        embed_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout


class AcousticSpkrVerf(PreTrainedModel):
    """
    Speaker verification model: maps acoustic features to L2-normalized speaker embeddings.

    Input:  [B, 512] acoustic features (continuous tokens)
    Output: [B, 192] L2-normalized speaker embeddings (dot product = cosine similarity)
    """

    config_class = AcousticSpkrVerfConfig
    base_model_prefix = "encoder"

    def __init__(self, config: AcousticSpkrVerfConfig):
        super().__init__(config)
        self.config = config

        layers = []
        for i in range(config.num_layers):
            in_d = config.input_dim if i == 0 else config.hidden_dim
            out_d = config.embed_dim if i == config.num_layers - 1 else config.hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < config.num_layers - 1:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.dropout))
        self.net = nn.Sequential(*layers)

        self.post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 512] -> [..., 192], L2-normalized."""
        emb = self.net(x)
        return F.normalize(emb, dim=-1)

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two L2-normalized embeddings."""
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        return (emb1 * emb2).sum(dim=-1)
