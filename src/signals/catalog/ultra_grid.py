"""Ultra grid: published indicators x holds x sessions x hours. Prefix u_."""

from __future__ import annotations

from src.signals.catalog.breakout_rules import _donchian
from src.signals.catalog.helpers import dow_utc, hours_utc, month_utc
from src.signals.catalog.momentum_rules import _macd, _roc
from src.signals.catalog.mr_rules import _bb_fade, _rsi_fade, _zscore_fade
from src.signals.catalog.public_systems import (
    _cci_fade,
    _stoch_fade,
    _supertrend_pos,
    _willr_fade,
)
from src.signals.catalog.spec import REGISTRY, RuleSpec, register
from src.signals.catalog.time_rules import _hour_side, _session_side
from src.signals.catalog.trend_rules import _dual_ma, _vs_ma

HOLDS = (4, 6, 8, 10, 12, 16, 20, 24, 36, 48)
SESS = {
    "asia": (0, 7),
    "lon": (7, 16),
    "ny": (12, 21),
    "ov": (12, 16),
    "lopen": (7, 11),
    "kzL": (7, 10),
    "kzN": (13, 16),
}
RSI_LV = ((20, 80), (25, 75), (30, 70), (35, 65), (40, 60))
DUAL_F = (5, 8, 10, 12, 15, 20, 21, 24, 30, 34)
DUAL_S = (20, 24, 30, 34, 48, 50, 55, 72, 100, 120, 168)


def _and_sess(fn, kwargs, lo, hi):
    def _f(df, **kw):
        pos = fn(df, **kwargs)
        h = hours_utc(df)
        out = pos.copy()
        out[~((h >= lo) & (h < hi))] = 0.0
        return out

    return _f


def _and_hour(fn, kwargs, hour):
    def _f(df, **kw):
        pos = fn(df, **kwargs)
        h = hours_utc(df)
        out = pos.copy()
        out[h != int(hour)] = 0.0
        return out

    return _f


def _dow_hour(df, dow, hour, side, **_):
    import numpy as np

    pos = np.zeros(len(df), dtype=np.float64)
    pos[(dow_utc(df) == int(dow)) & (hours_utc(df) == int(hour))] = float(side)
    return pos


def _and_month_rsi(df, month, period, lo, hi, **_):
    pos = _rsi_fade(df, period=int(period), lo=float(lo), hi=float(hi), mode="both")
    out = pos.copy()
    out[month_utc(df) != int(month)] = 0.0
    return out


def register_all() -> None:
    # --- core indicators x holds ---
    for hold in HOLDS:
        for w in range(5, 181, 2):
            register(
                RuleSpec(
                    rule_id=f"u_sma_{w}_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_vs_ma,
                    kwargs={"window": w, "kind": "sma", "mode": "both"},
                )
            )
        for w in range(5, 121, 3):
            register(
                RuleSpec(
                    rule_id=f"u_ema_{w}_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_vs_ma,
                    kwargs={"window": w, "kind": "ema", "mode": "both"},
                )
            )
        for p in range(3, 29, 2):
            for lo, hi in RSI_LV:
                register(
                    RuleSpec(
                        rule_id=f"u_rsi{p}_{int(lo)}_{int(hi)}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_rsi_fade,
                        kwargs={"period": p, "lo": float(lo), "hi": float(hi), "mode": "both"},
                    )
                )
        for lb in range(6, 97, 3):
            register(
                RuleSpec(
                    rule_id=f"u_don_{lb}_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_donchian,
                    kwargs={"lookback": lb, "use": "hl", "mode": "both"},
                )
            )
        for aw in (7, 10, 14, 20):
            for m in (2.0, 2.5, 3.0, 3.5):
                tag = str(m).replace(".", "p")
                register(
                    RuleSpec(
                        rule_id=f"u_st_{aw}_{tag}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_supertrend_pos,
                        kwargs={"atr_window": aw, "mult": m},
                    )
                )
        for w in range(10, 51, 5):
            for k in (1.5, 2.0, 2.5):
                tag = str(k).replace(".", "p")
                register(
                    RuleSpec(
                        rule_id=f"u_bb{w}_{tag}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_bb_fade,
                        kwargs={"window": w, "k": k},
                    )
                )
        for w in range(12, 73, 6):
            for t in (1.0, 1.5, 2.0):
                tag = str(t).replace(".", "p")
                register(
                    RuleSpec(
                        rule_id=f"u_z{w}_{tag}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_zscore_fade,
                        kwargs={"window": w, "thresh": t},
                    )
                )
        for p in range(6, 25, 2):
            register(
                RuleSpec(
                    rule_id=f"u_cci{p}_100_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_cci_fade,
                    kwargs={"period": p, "thresh": 100.0},
                )
            )
        for p in range(7, 22, 2):
            register(
                RuleSpec(
                    rule_id=f"u_wr{p}_80_20_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_willr_fade,
                    kwargs={"period": p, "lo": -80.0, "hi": -20.0},
                )
            )
            register(
                RuleSpec(
                    rule_id=f"u_wr{p}_70_30_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_willr_fade,
                    kwargs={"period": p, "lo": -70.0, "hi": -30.0},
                )
            )
        for p in (5, 8, 10, 14, 21):
            for lo, hi in ((20, 80), (30, 70)):
                register(
                    RuleSpec(
                        rule_id=f"u_stoch{p}_{lo}_{hi}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_stoch_fade,
                        kwargs={"period": p, "smooth": 3, "lo": float(lo), "hi": float(hi)},
                    )
                )
        for w in (4, 8, 12, 24, 36):
            for k in (0, 1, 2):
                register(
                    RuleSpec(
                        rule_id=f"u_roc{w}_k{k}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_roc,
                        kwargs={"window": w, "k": float(k)},
                    )
                )
        for hour in range(24):
            register(
                RuleSpec(
                    rule_id=f"u_hr{hour:02d}_L_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_hour_side,
                    kwargs={"hour": hour, "side": 1.0},
                )
            )
            register(
                RuleSpec(
                    rule_id=f"u_hr{hour:02d}_S_h{hold}",
                    family="ultra",
                    hold=hold,
                    fn=_hour_side,
                    kwargs={"hour": hour, "side": -1.0},
                )
            )
        for f in DUAL_F:
            for s in DUAL_S:
                if s <= f:
                    continue
                register(
                    RuleSpec(
                        rule_id=f"u_dsma_{f}_{s}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_dual_ma,
                        kwargs={"fast": f, "slow": s, "kind": "sma"},
                    )
                )

    # --- session x rsi/sma (subset of holds to add volume) ---
    for hold in (8, 12, 16, 24):
        for sname, span in SESS.items():
            for w in (8, 12, 20, 24, 48, 72, 100, 150):
                register(
                    RuleSpec(
                        rule_id=f"u_sess_{sname}_sma{w}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_sess(_vs_ma, {"window": w, "kind": "sma", "mode": "both"}, span[0], span[1]),
                    )
                )
            for p in (5, 7, 9, 14, 21):
                for lo, hi in RSI_LV:
                    register(
                        RuleSpec(
                            rule_id=f"u_sess_{sname}_rsi{p}_{int(lo)}_{int(hi)}_h{hold}",
                            family="ultra",
                            hold=hold,
                            fn=_and_sess(
                                _rsi_fade,
                                {"period": p, "lo": float(lo), "hi": float(hi), "mode": "both"},
                                span[0],
                                span[1],
                            ),
                        )
                    )

    # --- hour x rsi / sma / don ---
    for hold in (8, 12, 16, 24):
        for hour in range(24):
            for p in (6, 7, 9, 14, 21):
                register(
                    RuleSpec(
                        rule_id=f"u_hr{hour:02d}_rsi{p}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_hour(
                            _rsi_fade,
                            {"period": p, "lo": 30.0, "hi": 70.0, "mode": "both"},
                            hour,
                        ),
                    )
                )
            for w in (12, 24, 48, 72):
                register(
                    RuleSpec(
                        rule_id=f"u_hr{hour:02d}_sma{w}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_hour(_vs_ma, {"window": w, "kind": "sma", "mode": "both"}, hour),
                    )
                )
            for lb in (12, 24, 48):
                register(
                    RuleSpec(
                        rule_id=f"u_hr{hour:02d}_don{lb}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_hour(_donchian, {"lookback": lb, "use": "hl", "mode": "both"}, hour),
                    )
                )

    # --- dow x hour x side x hold ---
    for hold in (8, 12, 16, 24):
        for d in range(5):
            for hour in range(0, 24, 2):
                for side, tag in ((1.0, "L"), (-1.0, "S")):
                    register(
                        RuleSpec(
                            rule_id=f"u_dow{d}_hr{hour:02d}_{tag}_h{hold}",
                            family="ultra",
                            hold=hold,
                            fn=_dow_hour,
                            kwargs={"dow": d, "hour": hour, "side": side},
                        )
                    )

    # --- session x stoch / wr ---
    for hold in (12, 16, 24):
        for sname, span in SESS.items():
            for p in (7, 14, 21):
                register(
                    RuleSpec(
                        rule_id=f"u_sess_{sname}_wr{p}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_sess(_willr_fade, {"period": p, "lo": -80.0, "hi": -20.0}, span[0], span[1]),
                    )
                )
                register(
                    RuleSpec(
                        rule_id=f"u_sess_{sname}_stoch{p}_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_sess(
                            _stoch_fade,
                            {"period": p, "smooth": 3, "lo": 20.0, "hi": 80.0},
                            span[0],
                            span[1],
                        ),
                    )
                )

    # --- large crosses to reach 100k named strategies ---
    for hold in (6, 8, 10, 12, 16, 20, 24, 36):
        for hour in range(24):
            for p in range(4, 22, 2):
                for lo, hi in ((20, 80), (30, 70), (40, 60)):
                    register(
                        RuleSpec(
                            rule_id=f"u_hr{hour:02d}_rsi{p}_{lo}_{hi}_h{hold}",
                            family="ultra",
                            hold=hold,
                            fn=_and_hour(
                                _rsi_fade,
                                {"period": p, "lo": float(lo), "hi": float(hi), "mode": "both"},
                                hour,
                            ),
                        )
                    )
        for sname, span in SESS.items():
            for p in range(4, 22, 2):
                for lo, hi in ((30, 70), (40, 60)):
                    register(
                        RuleSpec(
                            rule_id=f"u_sess_{sname}_rsi{p}_{lo}_{hi}_x_h{hold}",
                            family="ultra",
                            hold=hold,
                            fn=_and_sess(
                                _rsi_fade,
                                {"period": p, "lo": float(lo), "hi": float(hi), "mode": "both"},
                                span[0],
                                span[1],
                            ),
                        )
                    )
            for w in range(8, 97, 4):
                register(
                    RuleSpec(
                        rule_id=f"u_sess_{sname}_sma{w}_x_h{hold}",
                        family="ultra",
                        hold=hold,
                        fn=_and_sess(_vs_ma, {"window": w, "kind": "sma", "mode": "both"}, span[0], span[1]),
                    )
                )
        for month in range(1, 13):
            for p in (5, 7, 9, 14, 21):
                for lo, hi in ((30, 70), (40, 60)):
                    register(
                        RuleSpec(
                            rule_id=f"u_mon{month:02d}_rsi{p}_{lo}_{hi}_h{hold}",
                            family="ultra",
                            hold=hold,
                            fn=_and_month_rsi,
                            kwargs={"month": month, "period": p, "lo": float(lo), "hi": float(hi)},
                        )
                    )

    extra_lv = ((15, 85), (18, 82), (22, 78), (28, 72), (32, 68), (38, 62), (42, 58), (45, 55))
    for hold in HOLDS:
        for hour in range(24):
            for p in range(2, 32):
                for lo, hi in extra_lv:
                    if len(REGISTRY) >= 100000:
                        return
                    rid = f"u2_hr{hour:02d}_rsi{p}_{int(lo)}_{int(hi)}_h{hold}"
                    if rid in REGISTRY:
                        continue
                    register(
                        RuleSpec(
                            rule_id=rid,
                            family="ultra",
                            hold=hold,
                            fn=_and_hour(
                                _rsi_fade,
                                {"period": p, "lo": float(lo), "hi": float(hi), "mode": "both"},
                                hour,
                            ),
                        )
                    )
    more_lv = ((12, 88), (16, 84), (24, 76), (26, 74), (33, 67), (36, 64), (43, 57), (48, 52))
    for hold in HOLDS:
        for hour in range(24):
            for p in range(2, 32):
                for lo, hi in more_lv:
                    if len(REGISTRY) >= 100000:
                        return
                    rid = f"u3_hr{hour:02d}_rsi{p}_{int(lo)}_{int(hi)}_h{hold}"
                    register(
                        RuleSpec(
                            rule_id=rid,
                            family="ultra",
                            hold=hold,
                            fn=_and_hour(
                                _rsi_fade,
                                {"period": p, "lo": float(lo), "hi": float(hi), "mode": "both"},
                                hour,
                            ),
                        )
                    )
