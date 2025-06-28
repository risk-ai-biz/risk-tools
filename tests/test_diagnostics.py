import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.instruments import InstrumentMap
from opt.risk import FactorRiskModel, FactorRiskModelData
from opt.workflow import ProblemBuilder
from opt.diagnostics import check_constraints


def test_check_constraints_reports_unlabelled():
    n = 2
    imap = InstrumentMap.identity(n)
    alpha = np.array([0.1, 0.2])
    frmd = FactorRiskModelData(loadings=np.eye(n), factor_cov=np.eye(n), specific_var=0.1 * np.ones(n))
    rm = FactorRiskModel(data=frmd)
    builder = ProblemBuilder(risk_model=rm, instrument_map=imap, alpha_dec=alpha)
    builder.add_bounds(range(n), 0.0, 0.5)
    cfg = builder.build()
    errors = check_constraints(cfg)
    assert not errors  # bounds labelled by builder
