"""Locked Trade Handler (Module 2): vol forecast → size, never direction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.baselines import SingleTFPatchModel
from src.handler.sizing import (
    SizingConfig,
    SizingResult,
    quantile_index,
    realized_vol_from_closes,
    size_from_vol,
)
from src.utils.config import load_config
from src.utils.io import load_checkpoint


MODULE_ID = 2
MODULE_NAME = "trade_handler"
HANDLER_VERSION = "v1-eurusd-1h-rv"


@dataclass
class VolForecast:
    pair: str
    timestamp: Optional[str]
    tf: str
    horizons_bars: List[int]
    primary_horizon_idx: int
    log_quantiles: Dict[int, Dict[str, float]]
    rv_quantiles: Dict[int, Dict[str, float]]


@dataclass
class HandlerDecision:
    pair: str
    timestamp: Optional[str]
    size_multiplier: float
    stand_aside: bool
    reason: str
    forecast: VolForecast
    handler_version: str = HANDLER_VERSION
    module: int = MODULE_ID
    side: Optional[str] = None  # always None — Signal owns direction
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["side"] = None
        return d


class VolatilityTradeHandler:
    """EURUSD realized-vol model + inverse-vol sizing.

    ``side`` is never populated. A Signal must supply direction later.
    """

    def __init__(
        self,
        model: SingleTFPatchModel,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        data = config["data"]
        ev = config.get("evaluation", {})
        hz = config.get("handler", {})
        self.pair = str(data.get("pair", "EURUSD"))
        self.primary_tf = str(data.get("primary_tf", "1h"))
        self.horizons_bars = list(data.get("horizons", {}).get(self.primary_tf, [4, 12]))
        self.quantiles = list(data.get("quantiles", [0.1, 0.5, 0.9]))
        self.primary_horizon_idx = int(
            hz.get("primary_horizon_idx", ev.get("primary_eval_horizon_idx", 1))
        )
        self.rv_log_transform = bool(data.get("rv_log_transform", True))
        self.sizing_cfg = SizingConfig(
            min_multiplier=float(hz.get("min_multiplier", 0.0)),
            max_multiplier=float(hz.get("max_multiplier", 2.0)),
            max_log_width=float(hz.get("max_log_width", 1.5)),
            max_rv=hz.get("max_rv"),
        )
        self.model = model.to(device)
        self.model.eval()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        checkpoint: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> "VolatilityTradeHandler":
        cfg = load_config(config_path)
        model = SingleTFPatchModel(cfg, primary_tf=str(cfg["data"]["primary_tf"]))
        ckpt = checkpoint or cfg.get("handler", {}).get("checkpoint")
        if ckpt:
            state = load_checkpoint(ckpt, map_location=device)
            model.load_state_dict(state["model"])
        return cls(model, cfg, device=device)

    def _extract_quantiles(self, pred: torch.Tensor, horizon_idx: int) -> Dict[str, float]:
        # pred: [B, H, Q] — use batch 0
        i10 = quantile_index(self.quantiles, 0.1)
        i50 = quantile_index(self.quantiles, 0.5)
        i90 = quantile_index(self.quantiles, 0.9)
        row = pred[0, horizon_idx].detach().cpu().numpy()
        return {
            "q10": float(row[i10]),
            "q50": float(row[i50]),
            "q90": float(row[i90]),
        }

    def _to_rv(self, log_qs: Dict[str, float]) -> Dict[str, float]:
        if not self.rv_log_transform:
            return dict(log_qs)
        return {k: float(np.exp(v)) for k, v in log_qs.items()}

    @torch.no_grad()
    def decide(
        self,
        batch: Dict[str, Any],
        ref_rv: Optional[float] = None,
        closes: Optional[np.ndarray] = None,
    ) -> HandlerDecision:
        """Size one (or the first) sample in ``batch``. Never sets ``side``."""
        local = dict(batch)
        local["inputs"] = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch["inputs"].items()
        }
        if "context" in batch and torch.is_tensor(batch["context"]):
            local["context"] = batch["context"].to(self.device)
        out = self.model(local)
        pred = out["predictions"][self.primary_tf]
        ts = batch.get("timestamp", [None])[0]
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        elif ts is not None:
            ts = str(ts)

        log_q: Dict[int, Dict[str, float]] = {}
        rv_q: Dict[int, Dict[str, float]] = {}
        for h_idx in range(len(self.horizons_bars)):
            lq = self._extract_quantiles(pred, h_idx)
            log_q[h_idx] = lq
            rv_q[h_idx] = self._to_rv(lq)

        primary = log_q[self.primary_horizon_idx]
        if ref_rv is None:
            if closes is None:
                raise ValueError("Provide ref_rv or closes for the reference realized vol")
            ref_rv = realized_vol_from_closes(np.asarray(closes))
        if not np.isfinite(ref_rv):
            sizing = SizingResult(0.0, True, "invalid_ref_rv", float("nan"), float("nan"), float("nan"))
        else:
            sizing = size_from_vol(
                primary["q10"],
                primary["q50"],
                primary["q90"],
                ref_rv=float(ref_rv),
                cfg=self.sizing_cfg,
            )

        forecast = VolForecast(
            pair=self.pair,
            timestamp=ts,
            tf=self.primary_tf,
            horizons_bars=list(self.horizons_bars),
            primary_horizon_idx=self.primary_horizon_idx,
            log_quantiles=log_q,
            rv_quantiles=rv_q,
        )
        return HandlerDecision(
            pair=self.pair,
            timestamp=ts,
            size_multiplier=sizing.size_multiplier,
            stand_aside=sizing.stand_aside,
            reason=sizing.reason,
            forecast=forecast,
            side=None,
            extras={
                "forecast_rv": sizing.forecast_rv,
                "ref_rv": sizing.ref_rv,
                "log_width": sizing.log_width,
            },
        )
