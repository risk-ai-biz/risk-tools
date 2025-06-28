import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.result import PortfolioResult


class DummyExpr:
    def __init__(self, val):
        self.val = val

    def level(self):
        return self.val


def test_result_instantiation_from_expr():
    dec = DummyExpr([0.1, 0.2])
    exp = DummyExpr([0.3, 0.4])
    res = PortfolioResult(
        decision_weights=dec,
        exposure_weights=exp,
        obj_value=DummyExpr(1.0),
        risk_sys=DummyExpr(0.5),
        risk_spec=DummyExpr(0.2),
        cost=DummyExpr(0.01),
    )
    assert isinstance(res.decision_weights, np.ndarray)
    np.testing.assert_allclose(res.decision_weights, [0.1, 0.2])
    assert isinstance(res.obj_value, float) and np.isclose(res.obj_value, 1.0)
