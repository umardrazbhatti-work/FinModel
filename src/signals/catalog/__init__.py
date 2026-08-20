"""Rule catalog. Import build_catalog() before listing rules."""

from __future__ import annotations

from src.signals.catalog.spec import REGISTRY, RuleSpec, get_rule, list_rules, positions_for, register

_BUILT = False


def build_catalog() -> None:
    global _BUILT
    if _BUILT:
        return
    from src.signals.catalog import (
        breakout_rules,
        combo_systems,
        controls,
        extra_rules,
        mass_grid,
        h1_control,
        momentum_rules,
        mr_rules,
        price_action,
        public_systems,
        smc,
        session_systems,
        time_rules,
        trend_rules,
        ultra_grid,
        vol_rules,
    )

    controls.register_all()
    time_rules.register_all(hold=12)
    extra_rules.register_all(hold=12)
    trend_rules.register_all(hold=12)
    breakout_rules.register_all(hold=12)
    mr_rules.register_all(hold=12)
    momentum_rules.register_all(hold=12)
    vol_rules.register_all(hold=12)
    h1_control.register_all()
    public_systems.register_all(hold=12)
    session_systems.register_all(hold=12)
    price_action.register_all(hold=12)
    combo_systems.register_all(hold=12)
    smc.register_all(hold=12)
    smc.register_all(hold=24)
    mass_grid.register_all()
    ultra_grid.register_all()
    _BUILT = True


def catalog_size() -> int:
    build_catalog()
    return len(REGISTRY)


__all__ = [
    "REGISTRY",
    "RuleSpec",
    "build_catalog",
    "catalog_size",
    "get_rule",
    "list_rules",
    "positions_for",
    "register",
]
