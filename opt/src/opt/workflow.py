from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import ProblemConfig, UtilityConfig
from .constraints import (
    FactorBoundConstraint,
    GrossExposureConstraint,
    InstrumentBoundConstraint,
    InstrumentTurnoverConstraint,
    NetExposureConstraint,
)
from .instruments import InstrumentMap
from .portfolio import Portfolio
from .risk import FactorRiskModel


class ProblemBuilder:
    """Convenience helper for constructing :class:`ProblemConfig`."""

    def __init__(
        self,
        risk_model: FactorRiskModel,
        instrument_map: InstrumentMap,
        alpha_dec: NDArray[np.floating],
    ) -> None:
        self.risk_model = risk_model
        self.instrument_map = instrument_map
        self.alpha_dec = alpha_dec
        self.alpha_exp = np.zeros(instrument_map.shape[0])
        self.start_dec = np.zeros(instrument_map.shape[1])
        self.cost_model: Optional[Any] = None
        self.constraints: list = []
        self.risk_aversion_sys = 1.0
        self.risk_aversion_spec = 1.0

    def with_cost_model(self, model: Any) -> "ProblemBuilder":
        self.cost_model = model
        return self

    def with_start_from_portfolio(self, p: Portfolio) -> "ProblemBuilder":
        self.start_dec = p.to_weights()
        return self

    def add_bounds(
        self, idx: Sequence[int], lower: float = -np.inf, upper: float = np.inf
    ) -> "ProblemBuilder":
        self.constraints.append(
            InstrumentBoundConstraint(idx=list(idx), lower=lower, upper=upper)
        )
        return self

    def add_turnover_limit(self, limit: float) -> "ProblemBuilder":
        self.constraints.append(
            InstrumentTurnoverConstraint(start_dec=self.start_dec, limit=limit)
        )
        return self

    def add_gross_limit(self, limit: float) -> "ProblemBuilder":
        self.constraints.append(GrossExposureConstraint(limit=limit))
        return self

    def add_net_bounds(self, lower: float = -np.inf, upper: float = np.inf) -> "ProblemBuilder":
        self.constraints.append(NetExposureConstraint(lower=lower, upper=upper))
        return self

    def add_group_limits(self, groups: Sequence[str], max_abs: float) -> "ProblemBuilder":
        unique = sorted(set(groups))
        B = np.zeros((len(groups), len(unique)))
        for i, g in enumerate(groups):
            B[i, unique.index(g)] = 1.0
        self.constraints.append(FactorBoundConstraint(B=B, max_abs=max_abs))
        return self

    def build(self) -> ProblemConfig:
        util = UtilityConfig(
            risk_aversion_sys=self.risk_aversion_sys,
            risk_aversion_spec=self.risk_aversion_spec,
            cost_model=self.cost_model,
        )
        return ProblemConfig(
            risk_model=self.risk_model,
            instrument_map=self.instrument_map,
            alpha_dec=self.alpha_dec,
            alpha_exp=self.alpha_exp,
            start_dec=self.start_dec,
            utility=util,
            constraints=list(self.constraints),
        )


def make_long_only_problem(
    risk_model: FactorRiskModel,
    instrument_map: InstrumentMap,
    alpha_dec: NDArray[np.floating],
    groups: Sequence[str],
) -> ProblemConfig:
    builder = ProblemBuilder(risk_model, instrument_map, alpha_dec)
    n = instrument_map.shape[1]
    builder.add_bounds(range(n), lower=0.0, upper=0.6)
    builder.add_gross_limit(3.0)
    builder.add_net_bounds(0.9, 1.1)
    builder.add_group_limits(groups, max_abs=2.0)
    return builder.build()


def make_long_short_problem(
    risk_model: FactorRiskModel,
    instrument_map: InstrumentMap,
    alpha_dec: NDArray[np.floating],
    start_dec: NDArray[np.floating],
    groups: Sequence[str],
) -> ProblemConfig:
    builder = ProblemBuilder(risk_model, instrument_map, alpha_dec)
    builder.start_dec = start_dec
    builder.add_bounds(range(instrument_map.shape[1]), lower=-0.5, upper=0.5)
    builder.add_turnover_limit(0.5)
    builder.add_gross_limit(3.0)
    builder.add_net_bounds(-0.2, 0.2)
    builder.add_group_limits(groups, max_abs=3.0)
    return builder.build()
