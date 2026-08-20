"""R-multiple engine: SL/TP path, pessimistic same-bar, risk, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.r_engine import simulate_r
from src.signals.r_entries import first_fire
from src.signals.r_eval import apply_r_gates


def _ohlc_from_rows(rows):
    n = len(rows)
    o, h, l, c = [], [], [], []
    for r in rows:
        o.append(r[0])
        h.append(r[1])
        l.append(r[2])
        c.append(r[3])
    atr = np.ones(n, dtype=np.float64)
    fire = np.zeros(n, dtype=np.float64)
    return (
        np.array(o),
        np.array(h),
        np.array(l),
        np.array(c),
        atr,
        fire,
    )


def test_tp_then_sl_expectancy():
    # bar0 close signals long; enter bar1 open=10, ATR=1, SL=9 TP=12
    rows = [
        (10, 10.2, 9.8, 10.0),
        (10.0, 12.2, 9.7, 12.1),  # TP
        (10.0, 10.2, 9.8, 10.0),
        (10.0, 10.2, 8.7, 8.8),  # SL
        (10.0, 10.1, 9.9, 10.0),
    ]
    o, h, l, c, atr, fire = _ohlc_from_rows(rows)
    fire[0] = 1.0
    fire[2] = 1.0
    sim = simulate_r(o, h, l, c, atr, fire, sl_mult=1.0, rr=2.0, trail=False, max_hold=20, spread=0.0, slip=0.0)
    assert sim["n_trades"] == 2
    r = sim["r_list"]
    assert r[0] > 1.8
    assert r[1] < 0
    assert sim["payoff"] >= 1.8
    assert sim["expectancy_r"] > 0
    expect = 100.0 * (1.0 + 0.01 * r[0]) * (1.0 + 0.01 * r[1])
    assert abs(sim["end_usd"] - expect) < 1e-6


def test_same_bar_sl_wins():
    rows = [
        (10, 10.2, 9.8, 10.0),
        (10.0, 12.5, 8.5, 10.0),
    ]
    o, h, l, c, atr, fire = _ohlc_from_rows(rows)
    fire[0] = 1.0
    sim = simulate_r(o, h, l, c, atr, fire, sl_mult=1.0, rr=2.0, trail=False, max_hold=20, spread=0.0, slip=0.0)
    assert sim["n_trades"] == 1
    assert sim["trades"][0].reason == "sl"
    assert sim["r_list"][0] < 0


def test_costs_reduce_r():
    rows = [
        (10, 10.2, 9.8, 10.0),
        (10.0, 12.2, 9.7, 12.1),
    ]
    o, h, l, c, atr, fire = _ohlc_from_rows(rows)
    fire[0] = 1.0
    clean = simulate_r(o, h, l, c, atr, fire, sl_mult=1.0, rr=2.0, trail=False, max_hold=20, spread=0.0, slip=0.0)
    taxed = simulate_r(o, h, l, c, atr, fire, sl_mult=1.0, rr=2.0, trail=False, max_hold=20, spread=0.0002, slip=0.0001)
    assert taxed["r_list"][0] < clean["r_list"][0]


def test_first_fire_not_every_bar():
    lvl = np.array([0, 1, 1, 1, 0, -1, -1, 1], dtype=float)
    f = first_fire(lvl)
    assert list(f) == [0, 1, 0, 0, 0, -1, 0, 1]


def test_gate_rejects_high_wr_low_payoff():
    row = {
        "discovery_expectancy_r": 0.05,
        "discovery_frac_pos": 1.0,
        "discovery_payoff": 0.6,
        "discovery_pf": 1.1,
        "discovery_sharpe": 2.0,
        "discovery_max_dd": -0.05,
        "discovery_n_trades": 80,
        "discovery_avg_win_r": 0.4,
        "discovery_avg_loss_r": 0.7,
        "invalid_high_wr_low_payoff": True,
        "unseen_expectancy_r": 0.04,
        "unseen_payoff": 0.6,
        "unseen_pf": 1.1,
        "unseen_sharpe": 2.0,
        "unseen_max_dd": -0.04,
        "unseen_n_trades": 40,
        "unseen_profit_usd": 2.0,
        "unseen_avg_win_r": 0.4,
        "unseen_avg_loss_r": 0.7,
        "unseen_invalid_high_wr": True,
    }
    apply_r_gates(row)
    assert row["survivor"] is False
    assert "high_wr_low_payoff" in row["discovery_reason"]
    assert "payoff<1.8" in row["discovery_reason"]


def test_gate_accepts_low_wr_high_payoff():
    row = {
        "discovery_expectancy_r": 0.25,
        "discovery_frac_pos": 0.67,
        "discovery_payoff": 2.2,
        "discovery_pf": 1.7,
        "discovery_sharpe": 1.4,
        "discovery_max_dd": -0.12,
        "discovery_n_trades": 40,
        "discovery_avg_win_r": 2.1,
        "discovery_avg_loss_r": 0.95,
        "invalid_high_wr_low_payoff": False,
        "unseen_expectancy_r": 0.22,
        "unseen_payoff": 2.0,
        "unseen_pf": 1.6,
        "unseen_sharpe": 1.3,
        "unseen_max_dd": -0.10,
        "unseen_n_trades": 20,
        "unseen_profit_usd": 5.0,
        "unseen_avg_win_r": 2.0,
        "unseen_avg_loss_r": 1.0,
        "unseen_invalid_high_wr": False,
    }
    apply_r_gates(row)
    assert row["survivor"] is True


def test_lookahead_past_trades_stable():
    rng = np.random.default_rng(0)
    n = 80
    close = 10 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + 0.2
    low = close - 0.2
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    atr = np.full(n, 0.15)
    fire = np.zeros(n)
    fire[5] = 1.0
    fire[40] = -1.0
    a = simulate_r(open_, high, low, close, atr, fire, sl_mult=1.5, rr=2.0, trail=True, max_hold=12, spread=0.0001, slip=0.00005)
    close2 = close.copy()
    close2[-1] += 10
    high2 = high.copy()
    high2[-1] += 10
    b = simulate_r(open_, high2, low, close2, atr, fire, sl_mult=1.5, rr=2.0, trail=True, max_hold=12, spread=0.0001, slip=0.00005)
    if a["n_trades"] and b["n_trades"]:
        assert a["trades"][0].entry_px == b["trades"][0].entry_px
        assert a["trades"][0].exit_i == b["trades"][0].exit_i
