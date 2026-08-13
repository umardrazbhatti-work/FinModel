"""Single-timeframe PatchTST-style baseline (primary TF only)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.models.fusion import FusionMLP
from src.models.quantile_head import MultiHorizonQuantileHead
from src.models.tf_encoder import TFEncoder


class SingleTFPatchModel(nn.Module):
    """Baseline: encode only the primary TF, then multi-horizon quantile head."""

    def __init__(self, config: Dict[str, Any], primary_tf: str = "1h") -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        self.primary_tf = primary_tf
        self.tradable_tfs: List[str] = list(
            data_cfg.get("tradable_tfs", ["30m", "1h", "4h"])
        )
        self.horizons = {k: list(v) for k, v in data_cfg.get("horizons", {}).items()}
        self.quantiles = list(data_cfg.get("quantiles", [0.1, 0.5, 0.9]))
        n_features = len(data_cfg.get("feature_cols", ["open", "high", "low", "close", "volume"]))
        d_model = int(model_cfg.get("d_model", 64))

        self.encoder = TFEncoder(
            d_model=d_model,
            n_layers=int(model_cfg.get("n_layers", 3)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            patch_len=int(model_cfg.get("patch_len", 16)),
            n_features=n_features,
            use_revin=bool(model_cfg.get("use_revin", True)),
        )
        self.fusion = FusionMLP(
            d_model=d_model,
            hidden=int(model_cfg.get("dim_feedforward", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        self.head = MultiHorizonQuantileHead(
            d_model=d_model,
            tradable_tfs=self.tradable_tfs,
            horizons=self.horizons,
            quantiles=self.quantiles,
        )
        # dummy gates for API compatibility (one-hot on primary if present)
        n_tfs = len(data_cfg.get("tfs", [primary_tf]))
        self.register_buffer("gate_weights", torch.ones(n_tfs) / n_tfs, persistent=False)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        x = batch["inputs"][self.primary_tf]
        h = self.encoder(x)
        h = self.fusion(h)
        predictions = self.head(h)
        return {
            "predictions": predictions,
            "gate_weights": self.gate_weights,
            "tf_representations": {self.primary_tf: h},
        }
