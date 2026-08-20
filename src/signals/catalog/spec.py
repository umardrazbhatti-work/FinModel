"""Named rule specs for the Module 1 factory. S-1 RULE_SPECS stays frozen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

PosFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family: str
    hold: int
    fn: PosFn
    control: bool = False
    needs_train: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


REGISTRY: Dict[str, RuleSpec] = {}


def register(spec: RuleSpec) -> RuleSpec:
    if spec.rule_id in REGISTRY:
        raise ValueError(f"duplicate rule_id: {spec.rule_id}")
    REGISTRY[spec.rule_id] = spec
    return spec


def get_rule(rule_id: str) -> RuleSpec:
    return REGISTRY[rule_id]


def list_rules(
    families: Optional[List[str]] = None,
    hold: Optional[int] = None,
    ids: Optional[List[str]] = None,
) -> List[RuleSpec]:
    out = list(REGISTRY.values())
    if ids is not None:
        want = set(ids)
        out = [s for s in out if s.rule_id in want]
    if families is not None:
        fam = set(families)
        out = [s for s in out if s.family in fam]
    if hold is not None:
        out = [s for s in out if int(s.hold) == int(hold)]
    return sorted(out, key=lambda s: (s.family, s.hold, s.rule_id))


def positions_for(
    spec: RuleSpec,
    df: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    returns: np.ndarray,
    cost: float,
    seed: int,
) -> np.ndarray:
    kwargs = dict(spec.kwargs or {})
    if spec.needs_train:
        return spec.fn(
            df,
            train_mask=train_mask,
            returns=returns,
            cost=cost,
            seed=seed,
            **kwargs,
        )
    return spec.fn(df, seed=seed, cost=cost, **kwargs)
