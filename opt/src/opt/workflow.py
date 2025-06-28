from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import ProblemConfig, UtilityConfig
from .constraints import (
    FactorBoundConstraint,
    GrossExposureConstraint,
    InstrumentBoundConstraint,
    TurnoverConstraint,
    NetExposureConstraint,
)
from .core import QuantityType
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
        *,
        quantity_type: QuantityType = QuantityType.WEIGHT,
    ) -> None:
        self.risk_model = risk_model
        self.instrument_map = instrument_map
        self.alpha_dec = alpha_dec
        self.alpha_exp = np.zeros(instrument_map.shape[0])
        self.start_dec = np.zeros(instrument_map.shape[1])
        self.quantity_type = quantity_type
        self.cost_model: Optional[Any] = None
        self.constraints: list = []
        self.risk_aversion_sys = 1.0
        self.risk_aversion_spec = 1.0

    def with_cost_model(self, model: Any) -> "ProblemBuilder":
        self.cost_model = model
        return self

    def with_start_from_portfolio(self, p: Portfolio) -> "ProblemBuilder":
        if self.quantity_type is QuantityType.NOTIONAL:
            self.start_dec = p.to_notional()
        else:
            self.start_dec = p.to_weights()
        return self

    def add_cash(self, name: str = "CASH") -> "ProblemBuilder":
        n_risk = self.instrument_map.shape[0]
        E_cash = np.zeros((n_risk, 1))
        self.instrument_map = InstrumentMap(
            E=np.hstack([self.instrument_map.E, E_cash]),
            names_risk=list(self.instrument_map.names_risk),
            names_decision=self.instrument_map.names_decision + [name],
        )
        self.alpha_dec = np.concatenate([self.alpha_dec, [0.0]])
        self.start_dec = np.concatenate([self.start_dec, [0.0]])
        return self

    def add_bounds(
        self, idx: Sequence[int], lower: float = -np.inf, upper: float = np.inf
    ) -> "ProblemBuilder":
        self.constraints.append(
            InstrumentBoundConstraint(idx=list(idx), lower=lower, upper=upper, label="bounds")
        )
        return self

    def add_turnover_limit(self, limit: float) -> "ProblemBuilder":
        self.constraints.append(
            TurnoverConstraint(start_dec=self.start_dec, limit=limit, label="turnover")
        )
        return self

    def add_gross_limit(self, limit: float) -> "ProblemBuilder":
        self.constraints.append(GrossExposureConstraint(limit=limit, label="gross"))
        return self

    def add_net_bounds(self, lower: float = -np.inf, upper: float = np.inf) -> "ProblemBuilder":
        self.constraints.append(NetExposureConstraint(lower=lower, upper=upper, label="net"))
        return self

    def add_group_limits(self, groups: Sequence[str], max_abs: float) -> "ProblemBuilder":
        unique = sorted(set(groups))
        B = np.zeros((len(groups), len(unique)))
        for i, g in enumerate(groups):
            B[i, unique.index(g)] = 1.0
        self.constraints.append(FactorBoundConstraint(B=B, max_abs=max_abs, label="group"))
        return self

    def build(self) -> ProblemConfig:
        util = UtilityConfig(
            risk_aversion_sys=self.risk_aversion_sys,
            risk_aversion_spec=self.risk_aversion_spec,
            cost_model=self.cost_model,
        )
        cfg = ProblemConfig(
            risk_model=self.risk_model,
            instrument_map=self.instrument_map,
            alpha_dec=self.alpha_dec,
            alpha_exp=self.alpha_exp,
            start_dec=self.start_dec,
            utility=util,
            constraints=list(self.constraints),
        )
        from .diagnostics import check_constraints

        errs = check_constraints(cfg)
        if errs:
            raise ValueError("Invalid constraint configuration: " + "; ".join(errs))
        return cfg


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
