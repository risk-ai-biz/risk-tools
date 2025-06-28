import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.instruments import InstrumentMap
from opt.risk import FactorRiskModel, FactorRiskModelData
from opt.config import ProblemConfig, UtilityConfig
from opt.costs import PowerLawCost
from opt.constraints import (
    InstrumentBoundConstraint,
    InstrumentTurnoverConstraint,
    GrossExposureConstraint,
    NetExposureConstraint,
    FactorBoundConstraint,
)
from opt.synthetic import load_synthetics_csv, apply_synthetics, SyntheticInstrument
from opt.optimizer import optimize_portfolio
from mosek.fusion import OptimizeError


def base_config_10():
    imap = InstrumentMap.identity(10)
    alpha = np.linspace(0.1, 1.0, 10)
    loadings = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 1],
            [1, 1, 1],
        ]
    )
    frmd = FactorRiskModelData(
        loadings=loadings,
        factor_cov=np.eye(3),
        specific_var=np.linspace(0.1, 1.0, 10),
    )
    cfg = ProblemConfig(
        risk_model=FactorRiskModel(data=frmd),
        instrument_map=imap,
        alpha_dec=alpha,
        start_dec=np.zeros(10),
        utility=UtilityConfig(cost_model=PowerLawCost(scale=0.01)),
    )
    return cfg


def load_all_synthetics():
    here = os.path.dirname(__file__)
    s1 = load_synthetics_csv(os.path.join(here, "synthetics_a.csv"))
    s2 = load_synthetics_csv(os.path.join(here, "synthetics_b.csv"))
    costs = {
        "FUTA": 0.015,
        "ETFB": 0.012,
        "PAIRC": 0.02,
        "INDEXD": 0.03,
    }
    all_syn = s1 + s2
    for syn in all_syn:
        syn.cost_model = PowerLawCost(scale=costs[syn.name])
    return all_syn


def test_load_and_apply_multiple_synthetics():
    cfg = base_config_10()
    syns = load_all_synthetics()
    new_cfg = apply_synthetics(cfg, syns)

    assert new_cfg.instrument_map.shape[1] == 14
    assert new_cfg.instrument_map.names_decision[-4:] == [
        "FUTA",
        "ETFB",
        "PAIRC",
        "INDEXD",
    ]
    # Check exposures for first synthetic
    np.testing.assert_allclose(new_cfg.instrument_map.E[:, 10], [0.5, 0.5] + [0] * 8)
    # Check alpha overlays applied
    assert np.isclose(new_cfg.alpha_dec[10], 0.16)
    assert np.isclose(new_cfg.alpha_dec[11], 0.36)
    assert len(new_cfg.utility.cost_model.models) == 14


def make_long_only_problem():
    cfg = apply_synthetics(base_config_10(), load_all_synthetics())
    n_dec = cfg.instrument_map.shape[1]
    cfg.constraints = [
        InstrumentBoundConstraint(idx=list(range(n_dec)), lower=0.0, upper=0.6),
        GrossExposureConstraint(limit=3.0),
        NetExposureConstraint(lower=0.9, upper=1.1),
        FactorBoundConstraint(B=np.ones((cfg.instrument_map.shape[0], 1)), max_abs=2.0),
    ]
    return cfg


def make_long_short_problem():
    cfg = apply_synthetics(base_config_10(), load_all_synthetics())
    n_dec = cfg.instrument_map.shape[1]
    start = np.zeros(n_dec)
    cfg.start_dec = start
    cfg.constraints = [
        InstrumentBoundConstraint(idx=list(range(n_dec)), lower=-0.5, upper=0.5),
        InstrumentTurnoverConstraint(start_dec=start, limit=0.5),
        GrossExposureConstraint(limit=3.0),
        NetExposureConstraint(lower=-0.2, upper=0.2),
        FactorBoundConstraint(B=np.ones((cfg.instrument_map.shape[0], 1)), max_abs=3.0),
    ]
    return cfg


def test_problem_long_only_config():
    cfg = make_long_only_problem()
    assert len(cfg.constraints) == 4
    assert isinstance(cfg.constraints[0], InstrumentBoundConstraint)
    assert isinstance(cfg.constraints[1], GrossExposureConstraint)
    assert isinstance(cfg.constraints[2], NetExposureConstraint)
    assert isinstance(cfg.constraints[3], FactorBoundConstraint)


def test_problem_long_short_config():
    cfg = make_long_short_problem()
    assert len(cfg.constraints) == 5
    assert isinstance(cfg.constraints[0], InstrumentBoundConstraint)
    assert isinstance(cfg.constraints[1], InstrumentTurnoverConstraint)
    assert isinstance(cfg.constraints[2], GrossExposureConstraint)
    assert isinstance(cfg.constraints[3], NetExposureConstraint)
    assert isinstance(cfg.constraints[4], FactorBoundConstraint)


def _solve_or_skip(cfg):
    try:
        return optimize_portfolio(cfg)
    except (ImportError, OptimizeError) as exc:
        import pytest

        pytest.skip(f"optimizer unavailable: {exc}")


def _check_long_only_constraints(cfg, res):
    w = res.decision_weights
    e = res.exposure_weights

    assert np.all(w >= -1e-8)
    assert np.all(w <= 0.6 + 1e-8)
    assert np.sum(np.abs(e)) <= 3.0 + 1e-6
    assert 0.9 - 1e-6 <= np.sum(e) <= 1.1 + 1e-6
    assert np.abs(np.sum(e)) <= 2.0 + 1e-6

    rm = cfg.risk_model
    np.testing.assert_allclose(res.risk_sys, rm.systematic_risk(e), rtol=1e-6)
    np.testing.assert_allclose(res.risk_spec, rm.specific_risk(e), rtol=1e-6)


def _check_long_short_constraints(cfg, res):
    w = res.decision_weights
    e = res.exposure_weights

    assert np.all(w >= -0.5 - 1e-8)
    assert np.all(w <= 0.5 + 1e-8)
    turnover = np.sum(np.abs(w - cfg.start_dec))
    assert turnover <= 0.5 + 1e-6
    assert np.sum(np.abs(e)) <= 3.0 + 1e-6
    assert -0.2 - 1e-6 <= np.sum(e) <= 0.2 + 1e-6
    assert np.abs(np.sum(e)) <= 3.0 + 1e-6

    rm = cfg.risk_model
    np.testing.assert_allclose(res.risk_sys, rm.systematic_risk(e), rtol=1e-6)
    np.testing.assert_allclose(res.risk_spec, rm.specific_risk(e), rtol=1e-6)


def test_optimize_long_only():
    cfg = make_long_only_problem()
    res = _solve_or_skip(cfg)
    _check_long_only_constraints(cfg, res)


def test_optimize_long_short():
    cfg = make_long_short_problem()
    res = _solve_or_skip(cfg)
    _check_long_short_constraints(cfg, res)

