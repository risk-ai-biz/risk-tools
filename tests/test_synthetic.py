import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.portfolio_core import (
    InstrumentMap,
    FactorRiskModel, FactorRiskModelData,
    ProblemConfig, UtilityConfig,
    PowerLawCost,
    apply_synthetics,
)
from opt.synthetic import SyntheticInstrument


def base_config():
    imap = InstrumentMap.identity(2)
    alpha = np.array([0.1, 0.2])
    frmd = FactorRiskModelData(
        loadings=np.eye(2),
        factor_cov=np.eye(2),
        specific_var=np.array([0.1, 0.1]),
    )
    cfg = ProblemConfig(
        risk_model=FactorRiskModel(data=frmd),
        instrument_map=imap,
        alpha_dec=alpha,
        utility=UtilityConfig(cost_model=PowerLawCost(scale=0.01)),
    )
    return cfg


def test_extend_and_apply_synthetic():
    cfg = base_config()
    syn = SyntheticInstrument(
        name="SYN",
        weights={"A0": 0.5, "A1": 0.5},
        cost_model=PowerLawCost(scale=0.02),
        alpha_overlay=0.01,
    )
    new_cfg = apply_synthetics(cfg, [syn])

    # instrument names extended
    assert new_cfg.instrument_map.names_decision == ["A0", "A1", "SYN"]
    # synthetic exposures are averages of the two columns
    np.testing.assert_allclose(new_cfg.instrument_map.E[:, 2], [0.5, 0.5])
    # alpha with overlay
    assert np.isclose(new_cfg.alpha_dec[2], 0.16)
    # cost model wrapper
    assert len(new_cfg.utility.cost_model.models) == 3
