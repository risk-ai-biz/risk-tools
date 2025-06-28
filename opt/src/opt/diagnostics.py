from __future__ import annotations

from typing import List

from .config import ProblemConfig
from .constraints import (
    ConstraintSpec,
    InstrumentBoundConstraint,
    InstrumentTurnoverConstraint,
    GrossExposureConstraint,
    NetExposureConstraint,
    FactorBoundConstraint,
)


def check_constraints(cfg: ProblemConfig) -> List[str]:
    """Return a list of obvious consistency issues in ``cfg``."""
    errors: List[str] = []
    for c in cfg.constraints:
        if isinstance(c, InstrumentBoundConstraint) and c.lower > c.upper:
            errors.append(f"Instrument bounds for {c.idx} are inconsistent")
        if isinstance(c, NetExposureConstraint) and c.lower > c.upper:
            errors.append("Net exposure lower bound exceeds upper bound")
        if isinstance(c, InstrumentTurnoverConstraint) and c.limit < 0:
            errors.append("Turnover limit cannot be negative")
        if isinstance(c, GrossExposureConstraint) and c.limit < 0:
            errors.append("Gross exposure limit cannot be negative")
        if isinstance(c, FactorBoundConstraint) and c.max_abs < 0:
            errors.append("Factor bound max_abs cannot be negative")
        if isinstance(c, ConstraintSpec) and getattr(c, "label", "") == "":
            errors.append(f"Constraint {c.__class__.__name__} has no label")
    return errors
