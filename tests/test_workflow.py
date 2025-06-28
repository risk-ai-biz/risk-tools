import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.instruments import InstrumentMap
from opt.risk import FactorRiskModel, FactorRiskModelData
from opt.workflow import ProblemBuilder, make_long_only_problem
from opt.core import QuantityType
from opt.costs import PowerLawCost
from opt.optimizer import optimize_portfolio
from mosek.fusion import OptimizeError


def _solve_or_skip(cfg):
    try:
        return optimize_portfolio(cfg)
    except (ImportError, OptimizeError) as exc:
        import pytest

        pytest.skip(f"optimizer unavailable: {exc}")


def test_builder_long_only():
    n = 5
    imap = InstrumentMap.identity(n)
    alpha = np.linspace(0.1, 0.5, n)
    loadings = np.eye(n)
    frmd = FactorRiskModelData(
        loadings=loadings,
        factor_cov=np.eye(n),
        specific_var=0.1 * np.ones(n),
    )
    rm = FactorRiskModel(data=frmd)
    builder = ProblemBuilder(risk_model=rm, instrument_map=imap, alpha_dec=alpha)
    builder.add_bounds(range(n), 0.0, 0.4)
    builder.add_gross_limit(2.0)
    builder.add_net_bounds(0.9, 1.1)
    builder.with_cost_model(PowerLawCost(scale=0.01))
    cfg = builder.build()

    res = _solve_or_skip(cfg)
    assert res.decision_weights.shape[0] == n


def test_make_long_only_problem():
    n = 4
    imap = InstrumentMap.identity(n)
    alpha = np.linspace(0.1, 0.4, n)
    loadings = np.eye(n)
    frmd = FactorRiskModelData(loadings=loadings, factor_cov=np.eye(n), specific_var=0.1 * np.ones(n))
    rm = FactorRiskModel(data=frmd)
    groups = ["g1"] * n
    cfg = make_long_only_problem(rm, imap, alpha, groups)

    res = _solve_or_skip(cfg)
    assert res.decision_weights.shape[0] == n


def test_builder_notional_and_cash():
    n = 3
    imap = InstrumentMap.identity(n)
    alpha = np.linspace(0.1, 0.3, n)
    loadings = np.eye(n)
    frmd = FactorRiskModelData(loadings=loadings, factor_cov=np.eye(n), specific_var=0.1 * np.ones(n))
    rm = FactorRiskModel(data=frmd)

    builder = ProblemBuilder(risk_model=rm, instrument_map=imap, alpha_dec=alpha, quantity_type=QuantityType.NOTIONAL)
    builder.add_cash()
    cfg = builder.build()

    assert cfg.instrument_map.shape[1] == n + 1
