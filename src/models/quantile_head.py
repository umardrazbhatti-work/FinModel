"""Multi-horizon multi-TF quantile prediction head."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from src.data.multi_tf_dataset import DEFAULT_HORIZONS


class MultiHorizonQuantileHead(nn.Module):
    """
    Predict quantiles for multiple horizons on tradable TFs.

    Separate linear heads per TF.
    """

    def __init__(
        self,
        d_model: int,
        tradable_tfs: List[str] | None = None,
        horizons: Dict[str, List[int]] | None = None,
        quantiles: List[float] | None = None,
    ) -> None:
        super().__init__()
        self.tradable_tfs = list(tradable_tfs or ["30m", "1h", "4h"])
        self.horizons = horizons or {
            k: v for k, v in DEFAULT_HORIZONS.items() if k in self.tradable_tfs
        }
        self.quantiles = list(quantiles or [0.1, 0.5, 0.9])
        self.n_quantiles = len(self.quantiles)

        self.heads = nn.ModuleDict()
        for tf in self.tradable_tfs:
            n_h = len(self.horizons[tf])
            self.heads[tf] = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_h * self.n_quantiles),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Returns:
            {tf: Tensor [B, n_horizons, n_quantiles]}
        """
        out: Dict[str, Tensor] = {}
        b = x.shape[0]
        for tf in self.tradable_tfs:
            n_h = len(self.horizons[tf])
            pred = self.heads[tf](x).view(b, n_h, self.n_quantiles)
            out[tf] = pred
        return out
