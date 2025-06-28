import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "opt", "src"))

from opt.result import PortfolioResult
from opt.utils import value_of


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


class FakeWorkStack:
    def __init__(self):
        self.data = []

    def allocf64(self, n):
        self.data = []

    def pushf64(self, x):
        self.data.append(float(x))

    def popf64(self):
        return self.data.pop()


class FakeExpr:
    def __init__(self, val):
        self.arr = np.asarray(val, dtype=float)

    def getSize(self):
        return self.arr.size

    def getShape(self):
        return self.arr.shape

    def eval(self, rs, ws, xs):
        for v in reversed(self.arr.ravel()):
            rs.pushf64(v)


def test_value_of_mosek_like_expr(monkeypatch):
    import types

    fake_mod = types.SimpleNamespace(Expr=FakeExpr, WorkStack=FakeWorkStack)
    monkeypatch.setitem(sys.modules, "mosek", types.ModuleType("mosek"))
    monkeypatch.setitem(sys.modules, "mosek.fusion", fake_mod)

    expr = FakeExpr([[1.0, 2.0], [3.0, 4.0]])
    val = value_of(expr)
    np.testing.assert_allclose(val, [[1.0, 2.0], [3.0, 4.0]])
