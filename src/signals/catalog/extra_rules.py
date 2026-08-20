"""Seasonality and extra declared fills so the catalog clears 500 named rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import month_utc
from src.signals.catalog.spec import RuleSpec, register


def _month_side(df: pd.DataFrame, month: int, side: float, **_: object) -> np.ndarray:
    m = month_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[m == int(month)] = float(side)
    return pos


def register_all(hold: int = 12) -> None:
    for month in range(1, 13):
        register(
            RuleSpec(
                rule_id=f"month_{month:02d}_long_h{hold}",
                family="time",
                hold=hold,
                fn=_month_side,
                kwargs={"month": month, "side": 1.0},
            )
        )
        register(
            RuleSpec(
                rule_id=f"month_{month:02d}_short_h{hold}",
                family="time",
                hold=hold,
                fn=_month_side,
                kwargs={"month": month, "side": -1.0},
            )
        )
