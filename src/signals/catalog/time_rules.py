"""Time-of-day / session / day-of-week rules. Positions use only the bar timestamp."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import dow_utc, hours_utc
from src.signals.catalog.spec import RuleSpec, register

SESSIONS = {
    "asia": (0, 7),
    "london": (7, 16),
    "ny": (12, 21),
    "overlap": (12, 16),
    "london_open": (7, 11),
    "ny_open": (13, 17),
    "london_close": (14, 16),
    "sydney": (21, 24),
    "asia_late": (4, 7),
    "london_lunch": (11, 13),
}

# 4-hour UTC blocks
BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]


def _in_hours(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    if lo <= hi:
        return (h >= lo) & (h < hi)
    return (h >= lo) | (h < hi)


def _session_side(df: pd.DataFrame, lo: int, hi: int, side: float, **_: object) -> np.ndarray:
    h = hours_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[_in_hours(h, lo, hi)] = float(side)
    return pos


def _session_ls(
    df: pd.DataFrame,
    long_span: Tuple[int, int],
    short_span: Tuple[int, int],
    **_: object,
) -> np.ndarray:
    h = hours_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[_in_hours(h, long_span[0], long_span[1])] = 1.0
    pos[_in_hours(h, short_span[0], short_span[1])] = -1.0
    return pos


def _hour_side(df: pd.DataFrame, hour: int, side: float, **_: object) -> np.ndarray:
    h = hours_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[h == int(hour)] = float(side)
    return pos


def _dow_side(df: pd.DataFrame, dow: int, side: float, **_: object) -> np.ndarray:
    d = dow_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[d == int(dow)] = float(side)
    return pos


def _dow_session(
    df: pd.DataFrame,
    dow: int,
    lo: int,
    hi: int,
    side: float,
    **_: object,
) -> np.ndarray:
    h = hours_utc(df)
    d = dow_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[(d == int(dow)) & _in_hours(h, lo, hi)] = float(side)
    return pos


def register_all(hold: int = 12) -> None:
    for name, (lo, hi) in SESSIONS.items():
        register(
            RuleSpec(
                rule_id=f"sess_{name}_long_h{hold}",
                family="time",
                hold=hold,
                fn=_session_side,
                kwargs={"lo": lo, "hi": hi, "side": 1.0},
            )
        )
        register(
            RuleSpec(
                rule_id=f"sess_{name}_short_h{hold}",
                family="time",
                hold=hold,
                fn=_session_side,
                kwargs={"lo": lo, "hi": hi, "side": -1.0},
            )
        )

    pairs = [
        ("london", "asia"),
        ("ny", "asia"),
        ("overlap", "asia"),
        ("london_open", "asia"),
        ("london", "sydney"),
        ("ny", "london_close"),
    ]
    for long_n, short_n in pairs:
        register(
            RuleSpec(
                rule_id=f"sess_{long_n}_L_{short_n}_S_h{hold}",
                family="time",
                hold=hold,
                fn=_session_ls,
                kwargs={"long_span": SESSIONS[long_n], "short_span": SESSIONS[short_n]},
            )
        )

    for hour in range(24):
        register(
            RuleSpec(
                rule_id=f"hour_{hour:02d}_long_h{hold}",
                family="time",
                hold=hold,
                fn=_hour_side,
                kwargs={"hour": hour, "side": 1.0},
            )
        )
        register(
            RuleSpec(
                rule_id=f"hour_{hour:02d}_short_h{hold}",
                family="time",
                hold=hold,
                fn=_hour_side,
                kwargs={"hour": hour, "side": -1.0},
            )
        )

    for dow in range(5):
        register(
            RuleSpec(
                rule_id=f"dow_{dow}_long_h{hold}",
                family="time",
                hold=hold,
                fn=_dow_side,
                kwargs={"dow": dow, "side": 1.0},
            )
        )
        register(
            RuleSpec(
                rule_id=f"dow_{dow}_short_h{hold}",
                family="time",
                hold=hold,
                fn=_dow_side,
                kwargs={"dow": dow, "side": -1.0},
            )
        )

    for lo, hi in BLOCKS:
        register(
            RuleSpec(
                rule_id=f"block_{lo}_{hi}_long_h{hold}",
                family="time",
                hold=hold,
                fn=_session_side,
                kwargs={"lo": lo, "hi": hi, "side": 1.0},
            )
        )
        register(
            RuleSpec(
                rule_id=f"block_{lo}_{hi}_short_h{hold}",
                family="time",
                hold=hold,
                fn=_session_side,
                kwargs={"lo": lo, "hi": hi, "side": -1.0},
            )
        )

    # Monday London long / Friday NY short — classic TOD overlays
    register(
        RuleSpec(
            rule_id=f"mon_london_long_h{hold}",
            family="time",
            hold=hold,
            fn=_dow_session,
            kwargs={"dow": 0, "lo": 7, "hi": 16, "side": 1.0},
        )
    )
    register(
        RuleSpec(
            rule_id=f"fri_ny_short_h{hold}",
            family="time",
            hold=hold,
            fn=_dow_session,
            kwargs={"dow": 4, "lo": 12, "hi": 21, "side": -1.0},
        )
    )
    register(
        RuleSpec(
            rule_id=f"fri_london_flatish_short_h{hold}",
            family="time",
            hold=hold,
            fn=_dow_session,
            kwargs={"dow": 4, "lo": 7, "hi": 16, "side": -1.0},
        )
    )
