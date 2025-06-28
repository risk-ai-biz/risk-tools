import enum
import types
import sys
import pytest

from opt.optimizer import ConvexFactorOptimizer

class DummyStatus(enum.Enum):
    Optimal = 0
    NearOptimal = 1
    Unknown = 2

class DummyError(Exception):
    pass

fake_mod = types.SimpleNamespace(SolutionStatus=DummyStatus, OptimizeError=DummyError)


def test_check_status_raises_on_bad_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "mosek", types.ModuleType("mosek"))
    monkeypatch.setitem(sys.modules, "mosek.fusion", fake_mod)
    with pytest.raises(DummyError):
        ConvexFactorOptimizer._check_status(DummyStatus.Unknown)


def test_check_status_ok(monkeypatch):
    monkeypatch.setitem(sys.modules, "mosek", types.ModuleType("mosek"))
    monkeypatch.setitem(sys.modules, "mosek.fusion", fake_mod)
    # should not raise
    ConvexFactorOptimizer._check_status(DummyStatus.Optimal)
