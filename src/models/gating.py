"""Learnable static timeframe gates."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TFGating(nn.Module):
    """
    Learnable static gates over timeframes.

    Softmax over a parameter vector of size n_tfs (sample-independent).
    """

    def __init__(
        self,
        tf_names: List[str],
        d_model: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.tf_names = list(tf_names)
        self.n_tfs = len(tf_names)
        self.temperature = float(temperature)
        self.logits = nn.Parameter(torch.zeros(self.n_tfs))
        self.d_model = d_model

    def gate_weights(self) -> Tensor:
        """Return softmax gate weights [n_tfs]."""
        return F.softmax(self.logits / max(self.temperature, 1e-6), dim=0)

    def forward(self, tf_reps: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            tf_reps: dict of [B, d_model] tensors, one per TF

        Returns:
            gated_combined: [B, d_model]
            gate_weights: [n_tfs]
        """
        weights = self.gate_weights()  # [n_tfs]
        # stack in declared order
        stacked = torch.stack([tf_reps[name] for name in self.tf_names], dim=1)
        # stacked: [B, n_tfs, d_model]
        w = weights.view(1, self.n_tfs, 1)
        combined = (stacked * w).sum(dim=1)
        return combined, weights
