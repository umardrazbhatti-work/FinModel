"""Rule factory: catalog size, leakage, $100 account, ledger append-only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.catalog import REGISTRY, build_catalog, get_rule, list_rules, positions_for
from src.signals.factory import apply_discovery_gate, attach_account, score_hold_window
from src.signals.ledger import append_rows, load_ledger, locked_keys
from src.signals.rules import next_bar_simple_return


def _toy_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2018-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="h")
    close = 1.1 + np.cumsum(rng.normal(0, 0.0002, size=n))
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "volume": np.ones(n),
        }
    )


def test_catalog_has_at_least_500_rules():
    build_catalog()
    assert len(REGISTRY) >= 10000, len(REGISTRY)


def test_vol_bases_exist():
    build_catalog()
    from src.signals.catalog.vol_rules import VOL_BASES

    missing = [b for b in VOL_BASES if b not in REGISTRY]
    assert missing == []


def test_donchian_no_lookahead():
    build_catalog()
    df = _toy_df(80)
    spec = get_rule("donchian_24_h12")
    train = np.zeros(len(df), dtype=bool)
    rets = next_bar_simple_return(df["close"].to_numpy())
    a = positions_for(spec, df, train_mask=train, returns=rets, cost=0.0001, seed=0)
    df2 = df.copy()
    df2.loc[len(df2) - 1, "high"] = float(df2["high"].max()) + 1.0
    b = positions_for(spec, df2, train_mask=train, returns=rets, cost=0.0001, seed=0)
    assert np.allclose(a[:-1], b[:-1])


def test_always_flat_account_stays_100():
    df = _toy_df(50)
    pos = np.zeros(len(df))
    r = next_bar_simple_return(df["close"].to_numpy())
    r = np.nan_to_num(r, nan=0.0)
    scored = score_hold_window(pos, r, 0, len(df) - 1, hold=1, cost=0.0001, periods_per_year=6048, start_usd=100.0)
    assert abs(scored["end_usd"] - 100.0) < 1e-9
    assert abs(scored["profit_usd"]) < 1e-9


def test_attach_account_scales_wealth():
    rec = attach_account({"final_wealth": 1.10, "total_return": 0.10, "max_drawdown": -0.05}, 100.0)
    assert abs(rec["end_usd"] - 110.0) < 1e-9
    assert abs(rec["profit_usd"] - 10.0) < 1e-9
    assert abs(rec["profit_pct"] - 10.0) < 1e-9


def test_flat_trick_rejected():
    row = apply_discovery_gate(
        {
            "control": False,
            "discovery_mean_exp": 0.0001,
            "discovery_frac_pos": 1.0,
            "discovery_pct_flat": 0.99,
            "discovery_mean_active": 80,
        },
        al=-0.001,
        cf=-0.001,
        min_exp=0.0,
        min_frac=0.5,
        max_flat=0.95,
        min_active=30,
    )
    assert row["discovery_pass"] is False
    assert "flat_trick" in row["discovery_reason"]


def test_ledger_append_only(tmp_path):
    rows = [
        {"wave": 0, "rule_id": "a", "hold": 12, "discovery_pass": False},
        {"wave": 0, "rule_id": "b", "hold": 12, "discovery_pass": True},
    ]
    n1 = append_rows(tmp_path, rows)
    n2 = append_rows(tmp_path, rows)
    df = load_ledger(tmp_path)
    assert n1 == 2
    assert n2 == 0
    assert len(df) == 2
    assert len(locked_keys(df)) == 2


def test_protocol_and_time_families_register():
    build_catalog()
    proto = list_rules(families=["protocol"])
    time_r = list_rules(families=["time"], hold=12)
    assert any(s.rule_id == "h12_k2_logistic_ohlc" for s in proto)
    assert len(time_r) >= 100


def test_s3_trade_stats_counts():
    from src.signals.s3 import _trade_stats

    pos = np.array([1, 0, -1, 1, 0, 1], dtype=float)
    sized = np.array([0.5, 0, 0.0, 1.0, 0, 0.0], dtype=float)
    st = _trade_stats(pos, sized, 0, 6, hold=1)
    assert st["n_slots"] == 6
    assert st["n_signal"] == 4
    assert st["n_traded"] == 2
    assert st["n_stood_aside"] == 2


def test_public_systems_register_and_no_lookahead_supertrend():
    build_catalog()
    pub = list_rules(families=["public"], hold=12)
    assert any(s.rule_id.startswith("cowabunga") for s in pub)
    assert any(s.rule_id.startswith("supertrend_10_3") for s in pub)
    assert any(s.rule_id.startswith("ichimoku") for s in pub)
    assert len(pub) >= 40
    df = _toy_df(200)
    spec = get_rule("supertrend_10_3p0_h12")
    train = np.zeros(len(df), dtype=bool)
    rets = next_bar_simple_return(df["close"].to_numpy())
    a = positions_for(spec, df, train_mask=train, returns=rets, cost=0.0001, seed=0)
    df2 = df.copy()
    df2.loc[len(df2) - 1, "high"] = float(df2["high"].max()) + 1.0
    b = positions_for(spec, df2, train_mask=train, returns=rets, cost=0.0001, seed=0)
    assert np.allclose(a[:-1], b[:-1])
