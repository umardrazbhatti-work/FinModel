"""R-multiple trade engine: predefined SL/TP, fixed fractional risk, costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.economic import max_drawdown_from_wealth


@dataclass
class Trade:
    side: int
    entry_i: int
    exit_i: int
    entry_px: float
    exit_px: float
    sl: float
    tp: float
    r: float
    reason: str
    mfe_r: float
    mae_r: float


def _metrics(
    trades: List[Trade],
    *,
    risk_frac: float,
    start_usd: float,
    n_bars: int,
    periods_per_year: float,
) -> Dict[str, Any]:
    empty = {
        "n_trades": 0,
        "win_rate": float("nan"),
        "loss_rate": float("nan"),
        "avg_win_r": float("nan"),
        "avg_loss_r": float("nan"),
        "payoff": float("nan"),
        "expectancy_r": float("nan"),
        "profit_factor": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "end_usd": float(start_usd),
        "profit_usd": 0.0,
        "r_sum": float("nan"),
        "equity": np.array([float(start_usd)], dtype=np.float64),
        "r_list": np.array([], dtype=np.float64),
        "reasons": {},
        "pass_payoff": False,
        "pass_pf": False,
        "pass_sharpe": False,
        "pass_dd": False,
        "pass_e": False,
        "invalid_high_wr_low_payoff": False,
    }
    if not trades:
        return empty

    r = np.array([t.r for t in trades], dtype=np.float64)
    wins = r[r > 0]
    losses = r[r < 0]
    n = int(r.size)
    n_w = int(wins.size)
    n_l = int(losses.size)
    wr = n_w / n
    lr = n_l / n
    avg_w = float(wins.mean()) if n_w else float("nan")
    avg_l = float(-losses.mean()) if n_l else float("nan")
    if n_w and n_l and avg_l > 1e-12:
        payoff = avg_w / avg_l
    elif n_w and not n_l:
        payoff = float("inf")
    else:
        payoff = 0.0
    e = float(r.mean())
    gross_w = float(wins.sum()) if n_w else 0.0
    gross_l = float(-losses.sum()) if n_l else 0.0
    pf = (gross_w / gross_l) if gross_l > 1e-12 else (float("inf") if gross_w > 0 else 0.0)

    wealth = np.empty(n + 1, dtype=np.float64)
    wealth[0] = float(start_usd)
    f = float(risk_frac)
    for i, ri in enumerate(r):
        wealth[i + 1] = wealth[i] * (1.0 + f * float(ri))
        if wealth[i + 1] < 1e-9:
            wealth[i + 1] = 1e-9
    mdd = max_drawdown_from_wealth(wealth)

    years = float(n_bars) / float(periods_per_year) if periods_per_year else float("nan")
    tpy = (n / years) if years and years > 1e-9 else float("nan")
    sig = float(r.std(ddof=0))
    if n >= 2 and sig > 1e-12 and np.isfinite(tpy) and tpy > 0:
        sharpe = float(np.sqrt(tpy) * e / sig)
    else:
        sharpe = float("nan")

    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    high_wr_low_pay = bool(wr >= 0.60 and (not np.isfinite(payoff) or payoff < 1.8))
    return {
        "n_trades": n,
        "win_rate": wr,
        "loss_rate": lr,
        "avg_win_r": avg_w,
        "avg_loss_r": avg_l,
        "payoff": float(payoff) if np.isfinite(payoff) else payoff,
        "expectancy_r": e,
        "profit_factor": float(pf) if np.isfinite(pf) else pf,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "end_usd": float(wealth[-1]),
        "profit_usd": float(wealth[-1] - start_usd),
        "r_sum": float(r.sum()),
        "equity": wealth,
        "r_list": r,
        "reasons": reasons,
        "pass_payoff": bool(np.isfinite(payoff) and payoff >= 1.8),
        "pass_pf": bool(np.isfinite(pf) and pf >= 1.5),
        "pass_sharpe": bool(np.isfinite(sharpe) and sharpe >= 1.2),
        "pass_dd": bool(np.isfinite(mdd) and mdd >= -0.25),
        "pass_e": bool(np.isfinite(e) and e > 0),
        "invalid_high_wr_low_payoff": high_wr_low_pay,
    }


def simulate_r(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    fire: np.ndarray,
    *,
    sl_mult: float,
    rr: float,
    trail: bool,
    max_hold: int,
    spread: float,
    slip: float,
    risk_frac: float = 0.01,
    start_usd: float = 100.0,
    lo: int = 0,
    hi: Optional[int] = None,
    periods_per_year: float = 6048.0,
) -> Dict[str, Any]:
    """
    Signal `fire[i]` is decided at close of bar i; entry is next open.
    Same-bar SL and TP: SL wins (pessimistic). Trailing updates after the bar.
    """
    n = int(len(close))
    if hi is None:
        hi = n
    lo = max(int(lo), 0)
    hi = min(int(hi), n)
    one_way = float(spread) * 0.5 + float(slip)
    trades: List[Trade] = []

    in_trade = False
    side = 0
    entry_i = 0
    entry_px = 0.0
    sl = 0.0
    tp = 0.0
    sl_dist = 0.0
    armed_trail = False
    mfe = 0.0
    mae = 0.0

    def _close_trade(j: int, px: float, reason: str) -> None:
        nonlocal in_trade, mfe, mae
        r_mult = float(side) * (float(px) - entry_px) / sl_dist
        trades.append(
            Trade(
                side=int(side),
                entry_i=int(entry_i),
                exit_i=int(j),
                entry_px=float(entry_px),
                exit_px=float(px),
                sl=float(sl),
                tp=float(tp),
                r=float(r_mult),
                reason=reason,
                mfe_r=float(mfe / sl_dist) if sl_dist > 0 else 0.0,
                mae_r=float(mae / sl_dist) if sl_dist > 0 else 0.0,
            )
        )
        in_trade = False

    just_entered = False
    for i in range(lo, hi):
        if not in_trade:
            sig_i = i - 1
            if sig_i < lo or sig_i < 0:
                continue
            side_f = int(fire[sig_i])
            if side_f == 0:
                continue
            a = float(atr[sig_i])
            if not np.isfinite(a) or a <= 0:
                continue
            sl_dist = float(sl_mult) * a
            if sl_dist < 3.0 * (float(spread) + float(slip)):
                continue
            fill = float(open_[i]) + float(side_f) * one_way
            if not np.isfinite(fill):
                continue
            in_trade = True
            side = side_f
            entry_i = i
            entry_px = fill
            sl = fill - side * sl_dist
            tp = fill + side * float(rr) * sl_dist
            armed_trail = False
            mfe = 0.0
            mae = 0.0
            just_entered = True
        else:
            just_entered = False

        o = float(open_[i])
        if not just_entered:
            if side > 0:
                if o <= sl:
                    _close_trade(i, o - float(slip), "gap_sl")
                    continue
                if o >= tp:
                    _close_trade(i, o - float(slip), "gap_tp")
                    continue
            else:
                if o >= sl:
                    _close_trade(i, o + float(slip), "gap_sl")
                    continue
                if o <= tp:
                    _close_trade(i, o + float(slip), "gap_tp")
                    continue

        hv = float(high[i])
        lv = float(low[i])
        if side > 0:
            mfe = max(mfe, hv - entry_px)
            mae = max(mae, entry_px - lv)
            stop_hit = lv <= sl
            tp_hit = hv >= tp
        else:
            mfe = max(mfe, entry_px - lv)
            mae = max(mae, hv - entry_px)
            stop_hit = hv >= sl
            tp_hit = lv <= tp

        if stop_hit:
            fill = sl - float(side) * float(slip)
            _close_trade(i, fill, "sl")
            continue
        if tp_hit:
            fill = tp - float(side) * float(slip)
            _close_trade(i, fill, "tp")
            continue

        held = i - entry_i
        if held >= int(max_hold):
            fill = float(close[i]) - float(side) * one_way
            _close_trade(i, fill, "time")
            continue

        if trail:
            if (not armed_trail) and mfe >= sl_dist:
                armed_trail = True
                if side > 0:
                    sl = max(sl, entry_px)
                else:
                    sl = min(sl, entry_px)
            if armed_trail:
                if side > 0:
                    sl = max(sl, hv - sl_dist)
                else:
                    sl = min(sl, lv + sl_dist)

    if in_trade and hi - 1 >= entry_i:
        j = hi - 1
        fill = float(close[j]) - float(side) * one_way
        _close_trade(j, fill, "window_end")

    out = _metrics(
        trades,
        risk_frac=risk_frac,
        start_usd=start_usd,
        n_bars=max(hi - lo, 1),
        periods_per_year=periods_per_year,
    )
    out["trades"] = trades
    return out
