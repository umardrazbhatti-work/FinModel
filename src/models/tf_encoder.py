"""Per-timeframe patch + Transformer encoder with optional RevIN."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .patch_encoder import PatchEmbedding


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al.)."""

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: Tensor, mode: str = "norm") -> Tensor:
        # x: [B, L, C]
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps * 0)
            x = x * self._std + self._mean
            return x
        raise ValueError(f"Unknown RevIN mode: {mode}")


class TFEncoder(nn.Module):
    """
    Small Transformer encoder for one timeframe.

    Input:  [B, seq_len, n_features]
    Output: [B, d_model] pooled representation
    """

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        patch_len: int = 16,
        n_features: int = 5,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN(n_features) if use_revin else None
        self.patch_embed = PatchEmbedding(
            n_features=n_features,
            patch_len=patch_len,
            d_model=d_model,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, seq_len, n_features]
        if self.revin is not None:
            x = self.revin(x, mode="norm")
        patches = self.patch_embed(x)  # [B, n_patches, d_model]
        h = self.encoder(patches)
        h = self.norm(h)
        # mean pool over patches
        return h.mean(dim=1)
