"""Small next-bar (H=1) control set. S-1 already failed this hold; do not expand."""

from __future__ import annotations

from src.signals.catalog.breakout_rules import _donchian
from src.signals.catalog.mr_rules import _rsi_fade
from src.signals.catalog.spec import RuleSpec, register
from src.signals.catalog.time_rules import _session_side
from src.signals.catalog.trend_rules import _vs_ma
from src.signals.rules import breakout, session_london, sma_drift


def register_all() -> None:
    hold = 1
    register(
        RuleSpec(
            rule_id="s1_session_london_h1",
            family="h1",
            hold=hold,
            fn=session_london,
        )
    )
    register(
        RuleSpec(
            rule_id="s1_sma_drift_24_h1",
            family="h1",
            hold=hold,
            fn=sma_drift,
            kwargs={"window": 24},
        )
    )
    register(
        RuleSpec(
            rule_id="s1_breakout_24_h1",
            family="h1",
            hold=hold,
            fn=breakout,
            kwargs={"lookback": 24},
        )
    )
    register(
        RuleSpec(
            rule_id="sess_london_long_h1",
            family="h1",
            hold=hold,
            fn=_session_side,
            kwargs={"lo": 7, "hi": 16, "side": 1.0},
        )
    )
    register(
        RuleSpec(
            rule_id="sma_24_h1",
            family="h1",
            hold=hold,
            fn=_vs_ma,
            kwargs={"window": 24, "kind": "sma", "mode": "both"},
        )
    )
    register(
        RuleSpec(
            rule_id="donchian_24_h1",
            family="h1",
            hold=hold,
            fn=_donchian,
            kwargs={"lookback": 24, "use": "hl", "mode": "both"},
        )
    )
    register(
        RuleSpec(
            rule_id="rsi14_30_70_h1",
            family="h1",
            hold=hold,
            fn=_rsi_fade,
            kwargs={"period": 14, "lo": 30.0, "hi": 70.0, "mode": "both"},
        )
    )
