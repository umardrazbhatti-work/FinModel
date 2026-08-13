"""Full Multi-TF Gated Patch Transformer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .fusion import FusionMLP
from .gating import TFGating
from .quantile_head import MultiHorizonQuantileHead
from .tf_encoder import TFEncoder


class MTPTransformer(nn.Module):
    """
    Multi-TF Gated Patch Transformer.

    Independent patch+encoder per TF → static gates → fusion → quantile heads.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})

        self.tfs: List[str] = list(data_cfg.get("tfs", ["5m", "15m", "30m", "1h", "4h", "1d"]))
        # normalize daily naming
        self.tfs = ["1d" if tf == "daily" else tf for tf in self.tfs]
        self.tradable_tfs: List[str] = list(
            data_cfg.get("tradable_tfs", ["30m", "1h", "4h"])
        )
        self.horizons: Dict[str, List[int]] = {
            k: list(v) for k, v in data_cfg.get("horizons", {}).items()
        }
        self.quantiles: List[float] = list(data_cfg.get("quantiles", [0.1, 0.5, 0.9]))
        self.feature_cols: List[str] = list(
            data_cfg.get("feature_cols", ["open", "high", "low", "close", "volume"])
        )
        self.context_cols: List[str] = list(data_cfg.get("context_cols", []))
        n_features = len(self.feature_cols)
        n_context = len(self.context_cols)

        d_model = int(model_cfg.get("d_model", 64))
        n_layers = int(model_cfg.get("n_layers", 3))
        n_heads = int(model_cfg.get("n_heads", 4))
        dim_ff = int(model_cfg.get("dim_feedforward", 128))
        dropout = float(model_cfg.get("dropout", 0.1))
        patch_len = int(model_cfg.get("patch_len", 16))
        use_revin = bool(model_cfg.get("use_revin", True))
        self.use_context = bool(model_cfg.get("use_context", True)) and n_context > 0
        temperature = float(model_cfg.get("gate_temperature", 1.0))

        self.encoders = nn.ModuleDict(
            {
                tf: TFEncoder(
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    dim_feedforward=dim_ff,
                    dropout=dropout,
                    patch_len=patch_len,
                    n_features=n_features,
                    use_revin=use_revin,
                )
                for tf in self.tfs
            }
        )
        self.gating = TFGating(
            tf_names=self.tfs,
            d_model=d_model,
            temperature=temperature,
        )
        self.fusion = FusionMLP(d_model=d_model, hidden=dim_ff, dropout=dropout)

        if self.use_context:
            self.context_proj = nn.Sequential(
                nn.Linear(n_context, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.context_proj = None

        self.head = MultiHorizonQuantileHead(
            d_model=d_model,
            tradable_tfs=self.tradable_tfs,
            horizons=self.horizons,
            quantiles=self.quantiles,
        )
        self.d_model = d_model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs: Dict[str, Tensor] = batch["inputs"]
        tf_reps: Dict[str, Tensor] = {}
        for tf in self.tfs:
            # map 1d key if batch uses same keys
            key = tf
            if key not in inputs and tf == "1d" and "daily" in inputs:
                key = "daily"
            if key not in inputs:
                raise KeyError(f"Missing TF input: {tf}")
            tf_reps[tf] = self.encoders[tf](inputs[key])

        gated, gate_weights = self.gating(tf_reps)
        fused = self.fusion(gated)

        if self.use_context and self.context_proj is not None:
            ctx = batch.get("context")
            if ctx is not None and ctx.numel() > 0 and ctx.shape[-1] > 0:
                fused = fused + self.context_proj(ctx)

        predictions = self.head(fused)
        return {
            "predictions": predictions,
            "gate_weights": gate_weights,
            "tf_representations": tf_reps,
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
