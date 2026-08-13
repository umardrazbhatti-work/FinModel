"""Fusion MLP after gated TF combination."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FusionMLP(nn.Module):
    """Two-layer MLP with residual connection."""

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))
