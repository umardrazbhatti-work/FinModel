"""Controls + locked S-1/S-2 references. Not new Signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.spec import RuleSpec, register
from src.signals.features import ohlc_features
from src.signals.labels import class_from_return, forward_simple_return
from src.signals.logistic import logistic_positions
from src.signals.rules import (
    always_flat,
    always_long,
    coin_flip,
    tod_train_hours,
)


def _logistic_h12_k2(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    cost: float = 0.0001,
    seed: int = 42,
    **_: object,
) -> np.ndarray:
    close = df["close"].to_numpy(dtype=np.float64)
    fwd = forward_simple_return(close, 12)
    y = class_from_return(fwd, cost=cost, k=2.0)
    x = ohlc_features(df)
    return logistic_positions(x, y, train_mask, seed=int(seed))


def register_all() -> None:
    for hold in (1, 12):
        register(
            RuleSpec(
                rule_id=f"always_flat_h{hold}",
                family="protocol",
                hold=hold,
                fn=always_flat,
                control=True,
            )
        )
        register(
            RuleSpec(
                rule_id=f"always_long_h{hold}",
                family="protocol",
                hold=hold,
                fn=always_long,
                control=True,
            )
        )
        register(
            RuleSpec(
                rule_id=f"coin_flip_h{hold}",
                family="protocol",
                hold=hold,
                fn=coin_flip,
                control=True,
            )
        )
    register(
        RuleSpec(
            rule_id="tod_train_hours_h1",
            family="protocol",
            hold=1,
            fn=tod_train_hours,
            needs_train=True,
            note="S-1 best non-control (FAIL)",
        )
    )
    register(
        RuleSpec(
            rule_id="h12_k2_logistic_ohlc",
            family="protocol",
            hold=12,
            fn=_logistic_h12_k2,
            needs_train=True,
            note="Locked S-2 champion. Reference bar.",
        )
    )
