"""Leakage-free multi-timeframe dataset for MTP-Transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# File-name stem for each logical TF (aligned parquet uses "daily" for 1d)
TF_FILE_STEM = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "daily",
    "daily": "daily",
}

TF_DELTA = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "daily": pd.Timedelta(days=1),
}

DEFAULT_LOOKBACK = {
    "5m": 96,
    "15m": 64,
    "30m": 48,
    "1h": 72,
    "4h": 42,
    "1d": 60,
}

DEFAULT_HORIZONS = {
    "30m": [1, 4, 12],
    "1h": [1, 4, 12],
    "4h": [1, 4, 12],
}

DEFAULT_FEATURE_COLS = ["open", "high", "low", "close", "volume"]

DEFAULT_CONTEXT_COLS = [
    "fed_funds_rate",
    "us_2y_yield",
    "us_10y_yield",
    "yield_curve_10y2y",
    "us_10y_real_yield",
    "us_cpi",
    "us_core_cpi",
    "us_core_pce",
    "us_nfp",
    "us_unemployment",
    "us_industrial_production",
    "usd_broad_index",
    "vix",
    "high_impact_count",
    "medium_impact_count",
    "usd_events",
    "eur_events",
    "gbp_events",
    "jpy_events",
]

TRADABLE_TFS = ["30m", "1h", "4h"]


def _normalize_tf(tf: str) -> str:
    if tf == "daily":
        return "1d"
    return tf


def resolve_parquet_path(data_dir: Union[str, Path], pair: str, tf: str) -> Path:
    """Resolve aligned parquet path for a pair/TF, handling 1d/daily naming."""
    data_dir = Path(data_dir)
    stem = TF_FILE_STEM.get(tf, tf)
    path = data_dir / f"{pair}_{stem}_aligned.parquet"
    if not path.exists():
        # fallback alternate naming
        alt = "1d" if stem == "daily" else "daily"
        alt_path = data_dir / f"{pair}_{alt}_aligned.parquet"
        if alt_path.exists():
            return alt_path
        raise FileNotFoundError(f"Aligned parquet not found for {pair} {tf}: {path}")
    return path


class MultiTFDataset(Dataset):
    """
    Leakage-free multi-timeframe dataset for MTP-Transformer.

    At every primary timestamp t:
    - All TF series are strictly truncated to bars with timestamp <= t
    - Targets are future log-returns on 30m / 1h / 4h only
    - Targets are normalized by past-only realized volatility
    - Small moves below cost_threshold are masked
    """

    def __init__(
        self,
        pair: str,
        data_dir: str,
        tfs: Optional[List[str]] = None,
        primary_tf: str = "1h",
        lookback: Optional[Dict[str, int]] = None,
        horizons: Optional[Dict[str, List[int]]] = None,
        quantiles: Optional[List[float]] = None,
        cost_threshold: float = 1e-4,
        feature_cols: Optional[List[str]] = None,
        context_cols: Optional[List[str]] = None,
        mode: str = "train",
        fold_start: Optional[pd.Timestamp] = None,
        fold_end: Optional[pd.Timestamp] = None,
        vol_window: int = 24,
        eps: float = 1e-8,
        feature_mean: Optional[Dict[str, np.ndarray]] = None,
        feature_std: Optional[Dict[str, np.ndarray]] = None,
        context_mean: Optional[np.ndarray] = None,
        context_std: Optional[np.ndarray] = None,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        self.pair = pair
        self.data_dir = Path(data_dir)
        self.tfs = [_normalize_tf(tf) for tf in (tfs or list(DEFAULT_LOOKBACK.keys()))]
        self.primary_tf = _normalize_tf(primary_tf)
        if self.primary_tf not in self.tfs:
            raise ValueError(f"primary_tf {self.primary_tf} must be in tfs {self.tfs}")

        self.lookback = dict(lookback or DEFAULT_LOOKBACK)
        self.lookback = {_normalize_tf(k): int(v) for k, v in self.lookback.items()}
        self.horizons = {
            _normalize_tf(k): list(v)
            for k, v in (horizons or DEFAULT_HORIZONS).items()
        }
        self.quantiles = list(quantiles or [0.1, 0.5, 0.9])
        self.cost_threshold = float(cost_threshold)
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.context_cols = list(context_cols or DEFAULT_CONTEXT_COLS)
        self.mode = mode
        self.vol_window = int(vol_window)
        self.eps = float(eps)
        self.standardize = standardize
        self.tradable_tfs = [tf for tf in TRADABLE_TFS if tf in self.tfs or tf in self.horizons]

        # Arrays keyed by logical TF name (1d not daily)
        self.timestamps: Dict[str, np.ndarray] = {}
        self.features: Dict[str, np.ndarray] = {}
        self.closes: Dict[str, np.ndarray] = {}
        self.context_arr: Optional[np.ndarray] = None
        self.primary_context: Optional[np.ndarray] = None

        self._load_all()
        self._prepare_standardization(
            feature_mean, feature_std, context_mean, context_std
        )
        self.sample_indices = self._build_sample_index(fold_start, fold_end)

    # ------------------------------------------------------------------ load
    def _load_all(self) -> None:
        for tf in self.tfs:
            path = resolve_parquet_path(self.data_dir, self.pair, tf)
            df = pd.read_parquet(path, engine="pyarrow")
            df = self._normalize_frame(df)
            self.timestamps[tf] = df["timestamp"].to_numpy(dtype="datetime64[ns]")
            feat = df[self.feature_cols].to_numpy(dtype=np.float64)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            self.features[tf] = feat.astype(np.float32)
            self.closes[tf] = df["close"].to_numpy(dtype=np.float32)

            if tf == self.primary_tf:
                ctx_cols = [c for c in self.context_cols if c in df.columns]
                if len(ctx_cols) != len(self.context_cols):
                    missing = set(self.context_cols) - set(ctx_cols)
                    # keep only available columns
                    self.context_cols = ctx_cols
                    if missing:
                        pass  # silent: some pairs may lack currency-specific events
                if self.context_cols:
                    ctx = df[self.context_cols].to_numpy(dtype=np.float64)
                    ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
                    self.primary_context = ctx.astype(np.float32)
                else:
                    self.primary_context = np.zeros((len(df), 0), dtype=np.float32)

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" not in df.columns:
            if "time" in df.columns:
                df = df.rename(columns={"time": "timestamp"})
            else:
                raise KeyError("Parquet must contain 'timestamp' or 'time' column")
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = ts
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        df = df.reset_index(drop=True)
        return df

    def _prepare_standardization(
        self,
        feature_mean: Optional[Dict[str, np.ndarray]],
        feature_std: Optional[Dict[str, np.ndarray]],
        context_mean: Optional[np.ndarray],
        context_std: Optional[np.ndarray],
    ) -> None:
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.context_mean = context_mean
        self.context_std = context_std

    def fit_standardization(
        self, indices: Optional[Sequence[int]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Compute per-TF feature mean/std and context mean/std from sample indices.
        Uses only history bars with timestamp <= sample t (no future stats).
        For efficiency, stats are computed over all bars in the train time range.
        """
        if indices is None:
            indices = self.sample_indices
        if len(indices) == 0:
            raise ValueError("Cannot fit standardization on empty index set")

        primary_ts = self.timestamps[self.primary_tf]
        t_min = primary_ts[min(indices)]
        t_max = primary_ts[max(indices)]

        feature_mean: Dict[str, np.ndarray] = {}
        feature_std: Dict[str, np.ndarray] = {}
        for tf in self.tfs:
            ts = self.timestamps[tf]
            # train-period bars only (past and within train fold)
            mask = (ts >= t_min - np.timedelta64(90, "D")) & (ts <= t_max)
            if not np.any(mask):
                mask = ts <= t_max
            vals = self.features[tf][mask]
            mu = vals.mean(axis=0).astype(np.float32)
            sigma = vals.std(axis=0).astype(np.float32)
            sigma = np.where(sigma < self.eps, 1.0, sigma).astype(np.float32)
            feature_mean[tf] = mu
            feature_std[tf] = sigma

        pmask = (primary_ts >= t_min) & (primary_ts <= t_max)
        ctx = self.primary_context[pmask] if self.primary_context is not None else np.zeros((1, 0), dtype=np.float32)
        if ctx.shape[1] == 0:
            c_mean = np.zeros((0,), dtype=np.float32)
            c_std = np.ones((0,), dtype=np.float32)
        else:
            c_mean = ctx.mean(axis=0).astype(np.float32)
            c_std = ctx.std(axis=0).astype(np.float32)
            c_std = np.where(c_std < self.eps, 1.0, c_std).astype(np.float32)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.context_mean = c_mean
        self.context_std = c_std
        return feature_mean, feature_std, c_mean, c_std

    def set_standardization(
        self,
        feature_mean: Dict[str, np.ndarray],
        feature_std: Dict[str, np.ndarray],
        context_mean: np.ndarray,
        context_std: np.ndarray,
    ) -> None:
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.context_mean = context_mean
        self.context_std = context_std

    # ------------------------------------------------------------- indexing
    def _build_sample_index(
        self,
        fold_start: Optional[pd.Timestamp],
        fold_end: Optional[pd.Timestamp],
    ) -> List[int]:
        primary_ts = self.timestamps[self.primary_tf]
        n = len(primary_ts)
        max_h = max(max(hs) for hs in self.horizons.values())
        # need enough primary lookback and enough future for longest primary-aligned horizon
        # future need is governed by 4h * 12 bars etc.; use max over tradable
        future_need = 0
        for tf, hs in self.horizons.items():
            if tf not in self.timestamps:
                continue
            # number of primary bars roughly covering max horizon on that TF
            delta_tf = TF_DELTA[tf]
            delta_p = TF_DELTA[self.primary_tf]
            bars = int(np.ceil((max(hs) * delta_tf) / delta_p)) + 2
            future_need = max(future_need, bars)

        min_primary_lb = self.lookback.get(self.primary_tf, 1)
        start_i = max(min_primary_lb, self.vol_window + 1)
        end_i = n - future_need
        if end_i <= start_i:
            raise ValueError(
                f"Not enough primary bars for lookback/horizons: n={n}, "
                f"start_i={start_i}, end_i={end_i}"
            )

        indices = list(range(start_i, end_i))

        if fold_start is not None:
            fs = pd.Timestamp(fold_start)
            if fs.tzinfo is None:
                fs = fs.tz_localize("UTC")
            else:
                fs = fs.tz_convert("UTC")
            fs64 = np.datetime64(fs.to_datetime64())
            indices = [i for i in indices if primary_ts[i] >= fs64]
        if fold_end is not None:
            fe = pd.Timestamp(fold_end)
            if fe.tzinfo is None:
                fe = fe.tz_localize("UTC")
            else:
                fe = fe.tz_convert("UTC")
            fe64 = np.datetime64(fe.to_datetime64())
            indices = [i for i in indices if primary_ts[i] < fe64]

        # Filter samples that cannot build full targets (missing future)
        valid: List[int] = []
        for i in indices:
            t = primary_ts[i]
            ok = True
            for tf in self.tradable_tfs:
                if tf not in self.timestamps:
                    ok = False
                    break
                # need at least one bar <= t and enough future for max horizon
                end_idx = int(np.searchsorted(self.timestamps[tf], t, side="right") - 1)
                if end_idx < self.vol_window:
                    ok = False
                    break
                max_h_tf = max(self.horizons[tf])
                if end_idx + max_h_tf >= len(self.timestamps[tf]):
                    ok = False
                    break
            if ok:
                valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self.sample_indices)

    def get_primary_timestamps(self) -> np.ndarray:
        return self.timestamps[self.primary_tf]

    def get_sample_timestamp(self, idx: int) -> pd.Timestamp:
        pi = self.sample_indices[idx]
        return pd.Timestamp(self.timestamps[self.primary_tf][pi], tz="UTC")

    # ------------------------------------------------------------- getitem
    def __getitem__(self, idx: int) -> Dict[str, object]:
        primary_i = self.sample_indices[idx]
        t = self.timestamps[self.primary_tf][primary_i]
        t_pd = pd.Timestamp(t, tz="UTC")

        inputs: Dict[str, torch.Tensor] = {}
        for tf in self.tfs:
            hist = self._get_history(tf, t)
            if self.standardize and self.feature_mean is not None:
                hist = (hist - self.feature_mean[tf]) / (self.feature_std[tf] + self.eps)
            inputs[tf] = torch.from_numpy(hist.astype(np.float32))

        context = self._get_context(primary_i)
        if self.standardize and self.context_mean is not None and context.size > 0:
            context = (context - self.context_mean) / (self.context_std + self.eps)
        context_t = torch.from_numpy(context.astype(np.float32))

        targets, masks, raw_returns = self._compute_targets(t)

        return {
            "inputs": inputs,
            "context": context_t,
            "targets": {
                tf: torch.from_numpy(targets[tf]) for tf in self.tradable_tfs
            },
            "target_mask": {
                tf: torch.from_numpy(masks[tf]) for tf in self.tradable_tfs
            },
            "raw_returns": {
                tf: torch.from_numpy(raw_returns[tf]) for tf in self.tradable_tfs
            },
            "timestamp": t_pd,
            "pair": self.pair,
        }

    def _get_history(self, tf: str, t: np.datetime64) -> np.ndarray:
        """Last lookback[tf] bars with timestamp <= t. Left-pad with zeros if short."""
        ts = self.timestamps[tf]
        end_idx = int(np.searchsorted(ts, t, side="right") - 1)
        lb = self.lookback[tf]
        n_feat = len(self.feature_cols)

        if end_idx < 0:
            return np.zeros((lb, n_feat), dtype=np.float32)

        start_idx = end_idx - lb + 1
        if start_idx >= 0:
            hist = self.features[tf][start_idx : end_idx + 1].copy()
        else:
            available = self.features[tf][0 : end_idx + 1]
            pad_len = lb - available.shape[0]
            pad = np.zeros((pad_len, n_feat), dtype=np.float32)
            hist = np.vstack([pad, available])
        return hist.astype(np.float32)

    def _get_context(self, primary_i: int) -> np.ndarray:
        if self.primary_context is None or self.primary_context.shape[1] == 0:
            return np.zeros((0,), dtype=np.float32)
        return self.primary_context[primary_i].copy()

    def _realized_vol(self, tf: str, t: np.datetime64) -> float:
        """Past-only realized vol of log-returns ending at or before t."""
        ts = self.timestamps[tf]
        end_idx = int(np.searchsorted(ts, t, side="right") - 1)
        if end_idx < 1:
            return 1.0
        start_idx = max(0, end_idx - self.vol_window)
        closes = self.closes[tf][start_idx : end_idx + 1].astype(np.float64)
        if len(closes) < 2:
            return 1.0
        # guard zeros
        closes = np.clip(closes, self.eps, None)
        log_rets = np.diff(np.log(closes))
        vol = float(np.std(log_rets))
        if not np.isfinite(vol) or vol < self.eps:
            return float(self.eps)
        return vol

    def _compute_targets(
        self, t: np.datetime64
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        targets: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        raw_returns: Dict[str, np.ndarray] = {}

        for tf in self.tradable_tfs:
            ts = self.timestamps[tf]
            end_idx = int(np.searchsorted(ts, t, side="right") - 1)
            tf_targets: List[float] = []
            tf_masks: List[float] = []
            tf_raw: List[float] = []

            if end_idx < 0:
                n_h = len(self.horizons[tf])
                targets[tf] = np.zeros(n_h, dtype=np.float32)
                masks[tf] = np.zeros(n_h, dtype=np.float32)
                raw_returns[tf] = np.zeros(n_h, dtype=np.float32)
                continue

            current_close = float(self.closes[tf][end_idx])
            vol = self._realized_vol(tf, t)

            for h in self.horizons[tf]:
                fut_idx = end_idx + h
                if fut_idx >= len(self.closes[tf]) or current_close <= 0:
                    tf_targets.append(0.0)
                    tf_masks.append(0.0)
                    tf_raw.append(0.0)
                    continue

                future_close = float(self.closes[tf][fut_idx])
                if future_close <= 0:
                    tf_targets.append(0.0)
                    tf_masks.append(0.0)
                    tf_raw.append(0.0)
                    continue

                raw_ret = float(np.log(future_close / current_close))
                tf_raw.append(raw_ret)

                if abs(raw_ret) < self.cost_threshold:
                    tf_targets.append(0.0)
                    tf_masks.append(0.0)
                else:
                    tf_targets.append(raw_ret / (vol + self.eps))
                    tf_masks.append(1.0)

            targets[tf] = np.asarray(tf_targets, dtype=np.float32)
            masks[tf] = np.asarray(tf_masks, dtype=np.float32)
            raw_returns[tf] = np.asarray(tf_raw, dtype=np.float32)

        return targets, masks, raw_returns

    # ------------------------------------------------------ leakage helpers
    def debug_history_timestamps(self, idx: int, tf: str) -> np.ndarray:
        """Return timestamps of history bars for leakage tests."""
        primary_i = self.sample_indices[idx]
        t = self.timestamps[self.primary_tf][primary_i]
        ts = self.timestamps[tf]
        end_idx = int(np.searchsorted(ts, t, side="right") - 1)
        lb = self.lookback[tf]
        if end_idx < 0:
            return np.array([], dtype="datetime64[ns]")
        start_idx = max(0, end_idx - lb + 1)
        return ts[start_idx : end_idx + 1]

    def debug_prediction_time(self, idx: int) -> np.datetime64:
        primary_i = self.sample_indices[idx]
        return self.timestamps[self.primary_tf][primary_i]
