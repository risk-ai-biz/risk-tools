from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .core import OptBaseModel
from .utils import abs_expr

try:
    import mosek.fusion as mf
    import mosek.fusion.pythonic  # noqa: F401 – enable operator overloads
except ImportError:  # pragma: no cover – unit-test builds without MOSEK
    mf = None  # type: ignore


class ConstraintSpec(OptBaseModel):
    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        ...


# --- Instrument-side constraints -----------------------------------------
class InstrumentBoundConstraint(ConstraintSpec):
    idx: Sequence[int]
    lower: float = -np.inf
    upper: float = np.inf

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        if np.isfinite(self.lower):
            M.constraint(w_dec[self.idx] >= self.lower)
        if np.isfinite(self.upper):
            M.constraint(w_dec[self.idx] <= self.upper)


class InstrumentTurnoverConstraint(ConstraintSpec):
    start_dec: NDArray[np.floating]
    limit: float

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        delta = w_dec - self.start_dec
        u = abs_expr(M, delta, "turnover_abs")
        M.constraint(mf.Expr.sum(u) <= self.limit)


# --- Exposure-side constraints -------------------------------------------
class GrossExposureConstraint(ConstraintSpec):
    limit: float

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        u = abs_expr(M, w_exp, "gross_abs")
        M.constraint(mf.Expr.sum(u) <= self.limit)


class NetExposureConstraint(ConstraintSpec):
    lower: float = -np.inf
    upper: float = np.inf

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        s = mf.Expr.sum(w_exp)
        if np.isfinite(self.lower):
            M.constraint(s >= self.lower)
        if np.isfinite(self.upper):
            M.constraint(s <= self.upper)


class FactorBoundConstraint(ConstraintSpec):
    B: NDArray[np.floating]
    max_abs: float

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        fexp = self.B.T @ w_exp
        M.constraint(fexp <= self.max_abs)
        M.constraint(fexp >= -self.max_abs)
