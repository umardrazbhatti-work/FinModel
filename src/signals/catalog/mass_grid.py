"""Dense published-indicator grids so the catalog clears 10_000 named strategies."""

from __future__ import annotations

from src.signals.catalog.breakout_rules import _donchian
from src.signals.catalog.helpers import dow_utc, hours_utc
from src.signals.catalog.momentum_rules import _macd, _persist, _roc
from src.signals.catalog.mr_rules import _bb_fade, _consec_reversal, _rsi_fade, _zscore_fade
from src.signals.catalog.public_systems import _cci_fade, _cci_trend, _stoch_fade, _supertrend_pos, _willr_fade
from src.signals.catalog.spec import REGISTRY, RuleSpec, register
from src.signals.catalog.time_rules import _dow_side, _hour_side, _session_side
from src.signals.catalog.trend_rules import _dual_ma, _vs_ma

SMA_W = list(range(5, 201))
EMA_W = list(range(5, 121))
DUAL = [
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 24, 26, 30, 32, 34,
    36, 40, 42, 45, 48, 50, 55, 60, 65, 72, 80, 89, 96, 100, 120, 144, 168, 200,
]
RSI_P = list(range(2, 31))
RSI_LV = [(20, 80), (25, 75), (30, 70), (35, 65), (40, 60)]
DON = list(range(5, 121, 1))
ST_ATR = list(range(6, 21))
ST_M = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
SESS = {
    "asia": (0, 7),
    "london": (7, 16),
    "ny": (12, 21),
    "overlap": (12, 16),
    "london_open": (7, 11),
}


def _dow_hour_side(df, dow: int, hour: int, side: float, **_: object):
    import numpy as np

    h = hours_utc(df)
    d = dow_utc(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[(d == int(dow)) & (h == int(hour))] = float(side)
    return pos


def _gated(base_fn, base_kwargs, hour=None, sess=None):
    def _fn(df, **kw):
        pos = base_fn(df, **base_kwargs)
        if hour is not None:
            h = hours_utc(df)
            pos = pos.copy()
            pos[h != int(hour)] = 0.0
        if sess is not None:
            lo, hi = sess
            h = hours_utc(df)
            pos = pos.copy()
            pos[~((h >= lo) & (h < hi))] = 0.0
        return pos

    return _fn


def register_all() -> None:
    holds = (12, 24)

    for hold in holds:
        for w in SMA_W:
            register(RuleSpec(rule_id=f"g_sma_{w}_h{hold}", family="mass", hold=hold, fn=_vs_ma, kwargs={"window": w, "kind": "sma", "mode": "both"}))
        for w in EMA_W:
            register(RuleSpec(rule_id=f"g_ema_{w}_h{hold}", family="mass", hold=hold, fn=_vs_ma, kwargs={"window": w, "kind": "ema", "mode": "both"}))
        for w in range(5, 81):
            register(RuleSpec(rule_id=f"g_sma_{w}_L_h{hold}", family="mass", hold=hold, fn=_vs_ma, kwargs={"window": w, "kind": "sma", "mode": "long"}))
            register(RuleSpec(rule_id=f"g_sma_{w}_S_h{hold}", family="mass", hold=hold, fn=_vs_ma, kwargs={"window": w, "kind": "sma", "mode": "short"}))
        for i, f in enumerate(DUAL):
            for s in DUAL[i + 1 :]:
                register(RuleSpec(rule_id=f"g_dsma_{f}_{s}_h{hold}", family="mass", hold=hold, fn=_dual_ma, kwargs={"fast": f, "slow": s, "kind": "sma"}))
        for i, f in enumerate(DUAL[:18]):
            for s in DUAL[i + 1 : 22]:
                if s <= f:
                    continue
                register(RuleSpec(rule_id=f"g_dema_{f}_{s}_h{hold}", family="mass", hold=hold, fn=_dual_ma, kwargs={"fast": f, "slow": s, "kind": "ema"}))
        for p in RSI_P:
            for lo, hi in RSI_LV:
                register(RuleSpec(rule_id=f"g_rsi{p}_{int(lo)}_{int(hi)}_h{hold}", family="mass", hold=hold, fn=_rsi_fade, kwargs={"period": p, "lo": lo, "hi": hi, "mode": "both"}))
        for p in range(5, 22):
            for lo, hi in ((20, 80), (30, 70)):
                register(RuleSpec(rule_id=f"g_stoch_{p}_{int(lo)}_{int(hi)}_h{hold}", family="mass", hold=hold, fn=_stoch_fade, kwargs={"period": p, "smooth": 3, "lo": float(lo), "hi": float(hi)}))
        for lb in DON:
            register(RuleSpec(rule_id=f"g_don_{lb}_h{hold}", family="mass", hold=hold, fn=_donchian, kwargs={"lookback": lb, "use": "hl", "mode": "both"}))
        for aw in ST_ATR:
            for m in ST_M:
                tag = str(m).replace(".", "p")
                register(RuleSpec(rule_id=f"g_st_{aw}_{tag}_h{hold}", family="mass", hold=hold, fn=_supertrend_pos, kwargs={"atr_window": aw, "mult": m}))
        for w in range(10, 42, 2):
            for k in (1.5, 2.0, 2.5):
                tag = str(k).replace(".", "p")
                register(RuleSpec(rule_id=f"g_bb{w}_{tag}_h{hold}", family="mass", hold=hold, fn=_bb_fade, kwargs={"window": w, "k": k}))
        for w in range(8, 81, 4):
            for t in (1.0, 1.5, 2.0, 2.5):
                tag = str(t).replace(".", "p")
                register(RuleSpec(rule_id=f"g_z{w}_{tag}_h{hold}", family="mass", hold=hold, fn=_zscore_fade, kwargs={"window": w, "thresh": t}))
        for p in range(6, 31):
            for t in (100.0, 200.0):
                register(RuleSpec(rule_id=f"g_cci{p}_{int(t)}_fade_h{hold}", family="mass", hold=hold, fn=_cci_fade, kwargs={"period": p, "thresh": t}))
                register(RuleSpec(rule_id=f"g_cci{p}_{int(t)}_tr_h{hold}", family="mass", hold=hold, fn=_cci_trend, kwargs={"period": p, "thresh": t}))
        for p in range(7, 29):
            for lo, hi in ((-80.0, -20.0), (-70.0, -30.0)):
                register(RuleSpec(rule_id=f"g_wr{p}_{int(abs(lo))}_{int(abs(hi))}_h{hold}", family="mass", hold=hold, fn=_willr_fade, kwargs={"period": p, "lo": lo, "hi": hi}))
        for w in range(2, 49):
            for k in (0, 1, 2):
                register(RuleSpec(rule_id=f"g_roc{w}_k{k}_h{hold}", family="mass", hold=hold, fn=_roc, kwargs={"window": w, "k": float(k)}))
        for hour in range(24):
            register(RuleSpec(rule_id=f"g_hr{hour:02d}_L_h{hold}", family="mass", hold=hold, fn=_hour_side, kwargs={"hour": hour, "side": 1.0}))
            register(RuleSpec(rule_id=f"g_hr{hour:02d}_S_h{hold}", family="mass", hold=hold, fn=_hour_side, kwargs={"hour": hour, "side": -1.0}))
        for n in range(2, 9):
            register(RuleSpec(rule_id=f"g_consec{n}_h{hold}", family="mass", hold=hold, fn=_consec_reversal, kwargs={"n_bars": n}))

    # Session x SMA and hour x RSI (one hold=12 to push over 10k without cloning everything)
    hold = 12
    for name, span in SESS.items():
        register(
            RuleSpec(
                rule_id=f"g_sess_{name}_rsi14_h{hold}",
                family="mass",
                hold=hold,
                fn=_gated(_rsi_fade, {"period": 14, "lo": 30.0, "hi": 70.0, "mode": "both"}, sess=span),
            )
        )
        for w in (8, 12, 20, 24, 48, 72, 100):
            register(
                RuleSpec(
                    rule_id=f"g_sess_{name}_sma{w}_h{hold}",
                    family="mass",
                    hold=hold,
                    fn=_gated(_vs_ma, {"window": w, "kind": "sma", "mode": "both"}, sess=span),
                )
            )
    for hour in range(24):
        for p in (6, 7, 9, 14, 21):
            register(
                RuleSpec(
                    rule_id=f"g_hr{hour:02d}_rsi{p}_h{hold}",
                    family="mass",
                    hold=hold,
                    fn=_gated(_rsi_fade, {"period": p, "lo": 30.0, "hi": 70.0, "mode": "both"}, hour=hour),
                )
            )
    for hold in (12, 24):
        for hour in range(24):
            for w in (12, 24, 48, 72):
                register(
                    RuleSpec(
                        rule_id=f"g_hr{hour:02d}_sma{w}_h{hold}",
                        family="mass",
                        hold=hold,
                        fn=_gated(_vs_ma, {"window": w, "kind": "sma", "mode": "both"}, hour=hour),
                    )
                )
            for lb in (12, 24, 48):
                register(
                    RuleSpec(
                        rule_id=f"g_hr{hour:02d}_don{lb}_h{hold}",
                        family="mass",
                        hold=hold,
                        fn=_gated(_donchian, {"lookback": lb, "use": "hl", "mode": "both"}, hour=hour),
                    )
                )
        for lb in range(5, 101):
            register(
                RuleSpec(
                    rule_id=f"g_donc_{lb}_h{hold}",
                    family="mass",
                    hold=hold,
                    fn=_donchian,
                    kwargs={"lookback": lb, "use": "close", "mode": "both"},
                )
            )
        for fast, slow, sig in (
            (5, 21, 5),
            (8, 21, 5),
            (8, 24, 9),
            (12, 26, 9),
            (12, 48, 9),
            (16, 48, 12),
            (19, 39, 9),
            (24, 52, 9),
        ):
            register(
                RuleSpec(
                    rule_id=f"g_macd_{fast}_{slow}_{sig}_h{hold}",
                    family="mass",
                    hold=hold,
                    fn=_macd,
                    kwargs={"fast": fast, "slow": slow, "signal": sig},
                )
            )
        for k in (0, 1, 2, 3):
            register(
                RuleSpec(
                    rule_id=f"g_persist_k{k}_h{hold}",
                    family="mass",
                    hold=hold,
                    fn=_persist,
                    kwargs={"horizon": hold, "k": float(k)},
                )
            )

    # Extra holds + windows to clear 10_000 named strategies
    for hold in (6, 8, 10, 16, 18, 20, 36, 48):
        for w in SMA_W:
            register(RuleSpec(rule_id=f"g_sma_{w}_h{hold}", family="mass", hold=hold, fn=_vs_ma, kwargs={"window": w, "kind": "sma", "mode": "both"}))
        for p in RSI_P:
            for lo, hi in RSI_LV:
                register(RuleSpec(rule_id=f"g_rsi{p}_{int(lo)}_{int(hi)}_h{hold}", family="mass", hold=hold, fn=_rsi_fade, kwargs={"period": p, "lo": lo, "hi": hi, "mode": "both"}))
        for lb in range(5, 97, 2):
            register(RuleSpec(rule_id=f"g_don_{lb}_h{hold}", family="mass", hold=hold, fn=_donchian, kwargs={"lookback": lb, "use": "hl", "mode": "both"}))
        for aw in (7, 10, 14, 20):
            for m in (2.0, 3.0, 3.5):
                tag = str(m).replace(".", "p")
                register(RuleSpec(rule_id=f"g_st_{aw}_{tag}_h{hold}", family="mass", hold=hold, fn=_supertrend_pos, kwargs={"atr_window": aw, "mult": m}))
    for hold in (12, 24):
        for w in range(201, 321):
            register(RuleSpec(rule_id=f"g_sma_{w}_h{hold}", family="mass", hold=hold, fn=_vs_ma, kwargs={"window": w, "kind": "sma", "mode": "both"}))
        for d in range(5):
            for hour in range(24):
                register(
                    RuleSpec(
                        rule_id=f"g_dow{d}_hr{hour:02d}_L_h{hold}",
                        family="mass",
                        hold=hold,
                        fn=_dow_hour_side,
                        kwargs={"dow": d, "hour": hour, "side": 1.0},
                    )
                )
                register(
                    RuleSpec(
                        rule_id=f"g_dow{d}_hr{hour:02d}_S_h{hold}",
                        family="mass",
                        hold=hold,
                        fn=_dow_hour_side,
                        kwargs={"dow": d, "hour": hour, "side": -1.0},
                    )
                )
