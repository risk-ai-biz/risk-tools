import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.portfolio_core import (
    InstrumentMap,
    PowerLawCost,
    UtilityConfig,
    ProblemConfig,
    FactorRiskModel,
    FactorRiskModelData,
)
from opt.synthetic import SyntheticInstrument, extend_instrument_map, PerInstrumentCostModel

# resolve forward references in ProblemConfig
ProblemConfig.model_rebuild()


def basic_risk_model(n: int) -> FactorRiskModel:
    data = FactorRiskModelData(
        loadings=np.eye(n),
        factor_cov=np.eye(n),
        specific_var=np.ones(n) * 0.1,
    )
    return FactorRiskModel(data=data)


def test_extend_instrument_map_basic():
    base_map = InstrumentMap(
        E=np.eye(2),
        names_risk=["r1", "r2"],
        names_decision=["A", "B"],
    )
    base_alpha = np.array([0.01, 0.02])
    base_cm = PowerLawCost(scale=0.1)
    syn = SyntheticInstrument(
        name="Syn",
        weights={"A": 0.5, "B": 0.5},
        cost_model=PowerLawCost(scale=0.2),
        alpha_overlay=0.001,
    )

    imap, alpha, cms = extend_instrument_map(base_map, base_alpha, [syn], base_cost_model=base_cm)

    assert imap.names_decision == ["A", "B", "Syn"]
    assert np.allclose(alpha, [0.01, 0.02, 0.016])
    assert isinstance(cms[0], PowerLawCost) and cms[0].scale == 0.1
    assert isinstance(cms[2], PowerLawCost) and cms[2].scale == 0.2


def test_problem_config_integration():
    imap = InstrumentMap(
        E=np.eye(2),
        names_risk=["r1", "r2"],
        names_decision=["A", "B"],
    )
    alpha = np.array([0.01, 0.02])
    base_cm = PowerLawCost(scale=0.1)
    syn = SyntheticInstrument(
        name="Syn",
        weights={"A": 0.5, "B": 0.5},
        cost_model=PowerLawCost(scale=0.2),
        alpha_overlay=0.001,
    )

    cfg = ProblemConfig(
        risk_model=basic_risk_model(2),
        instrument_map=imap,
        alpha_dec=alpha,
        utility=UtilityConfig(cost_model=base_cm),
        synthetic_instruments=[syn],
    )

    assert cfg.instrument_map.names_decision == ["A", "B", "Syn"]
    assert np.allclose(cfg.alpha_dec, [0.01, 0.02, 0.016])
    assert cfg.start_dec.shape[0] == 3
    assert isinstance(cfg.utility.cost_model, PerInstrumentCostModel)
    assert len(cfg.utility.cost_model.models) == 3
    assert isinstance(cfg.utility.cost_model.models[0], PowerLawCost)
    assert isinstance(cfg.utility.cost_model.models[2], PowerLawCost)
