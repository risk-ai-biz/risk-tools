import enum
from typing import List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, root_validator, validator

try:
    import mosek.fusion as mf
    import mosek.fusion.pythonic  # noqa: F401 – enable operator overloads
except ImportError:  # pragma: no cover – unit‑test builds without MOSEK
    mf = None  # type: ignore

# ---------------------------------------------------------------------------
# 1. Basic enums & helpers
# ---------------------------------------------------------------------------

class QuantityType(str, enum.Enum):
    WEIGHT = "weight"
    NOTIONAL = "notional"

# ---------------------------------------------------------------------------
# 2. Portfolio container (unchanged)
# ---------------------------------------------------------------------------

class Portfolio(BaseModel):
    positions: NDArray[np.floating]
    quantity_type: QuantityType = QuantityType.WEIGHT
    nav: float = 1.0
    cash: Optional[float] = None

    def to_weights(self) -> NDArray[np.floating]:
        return (
            self.positions
            if self.quantity_type is QuantityType.WEIGHT
            else self.positions / self.nav
        )

    def to_notional(self) -> NDArray[np.floating]:
        return (
            self.positions
            if self.quantity_type is QuantityType.NOTIONAL
            else self.positions * self.nav
        )

# ---------------------------------------------------------------------------
# 3. Instrument mapping
# ---------------------------------------------------------------------------

class InstrumentMap(BaseModel):
    """Dense matrix **E** mapping decision variables → risk exposures."""

    E: NDArray[np.floating]
    names_risk: List[str]
    names_decision: List[str]

    @validator("E")
    def _2d(cls, v):  # noqa: N805
        if v.ndim != 2:
            raise ValueError("Instrument matrix must be 2‑D")
        return v

    @property
    def shape(self) -> Tuple[int, int]:
        return self.E.shape

    @classmethod
    def identity(cls, n: int) -> "InstrumentMap":
        labels = [f"A{i}" for i in range(n)]
        return cls(E=np.eye(n), names_risk=labels, names_decision=labels)

# ---------------------------------------------------------------------------
# 4. Risk‑model layer
# ---------------------------------------------------------------------------

class FactorRiskModelData(BaseModel):
    loadings: NDArray[np.floating]
    factor_cov: NDArray[np.floating]
    specific_var: NDArray[np.floating]

class FactorRiskModel(BaseModel):
    data: FactorRiskModelData

    def systematic_risk(self, w_exp: NDArray[np.floating]) -> float:
        B, F = self.data.loadings, self.data.factor_cov
        return float(w_exp @ (B @ F @ B.T) @ w_exp)

    def specific_risk(self, w_exp: NDArray[np.floating]) -> float:
        return float((w_exp ** 2) @ self.data.specific_var)

    def total_risk(self, w_exp: NDArray[np.floating]) -> float:
        return self.systematic_risk(w_exp) + self.specific_risk(w_exp)

# ---------------------------------------------------------------------------
# 5. Transaction‑cost models
# ---------------------------------------------------------------------------

class TransactionCostModel(Protocol):
    def cost_soc(self, M: "mf.Model", delta_dec: "mf.Expr") -> "mf.Expr": ...  # noqa: F821

class PowerLawCost(BaseModel):
    exponent: float = Field(1.5, gt=1, lt=2)
    scale: float = 1e-4

    def cost_soc(self, M: "mf.Model", delta_dec: "mf.Expr") -> "mf.Expr":  # noqa: F821
        beta = abs_expr(M, delta_dec, "beta_cost")
        return self.scale * mf.sum(beta)

# ---------------------------------------------------------------------------
# 6. Constraint helpers & specs
# ---------------------------------------------------------------------------

def abs_expr(M: "mf.Model", x: "mf.Expr", name: str) -> "mf.Var":  # noqa: F821
    """Return non‑negative *u* with u ≥ |x| element‑wise."""
    u = M.variable(name, x.shape[0])
    M.constraint(x <= u)
    M.constraint(-x <= u)
    return u

class ConstraintSpec(BaseModel):
    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        ...

# --- Instrument‑side constraints -----------------------------------------

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
        M.constraint(mf.sum(u) <= self.limit)

# --- Exposure‑side constraints -------------------------------------------

class GrossExposureConstraint(ConstraintSpec):
    limit: float

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        u = abs_expr(M, w_exp, "gross_abs")
        M.constraint(mf.sum(u) <= self.limit)

class NetExposureConstraint(ConstraintSpec):
    lower: float = -np.inf
    upper: float = np.inf

    def apply(self, M: "mf.Model", w_dec: "mf.Expr", w_exp: "mf.Expr") -> None:  # noqa: F821
        s = mf.sum(w_exp)
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

# ---------------------------------------------------------------------------
# 7. Utility & ProblemConfig
# ---------------------------------------------------------------------------

class UtilityConfig(BaseModel):
    risk_aversion_sys: float = 1.0
    risk_aversion_spec: float = 1.0
    cost_model: Optional[TransactionCostModel] = None

class ProblemConfig(BaseModel):
    risk_model: FactorRiskModel
    instrument_map: InstrumentMap
    alpha_dec: NDArray[np.floating]
    alpha_exp: Optional[NDArray[np.floating]] = None  # overlay
    start_dec: NDArray[np.floating] = Field(default_factory=lambda: np.array([]))
    utility: UtilityConfig = UtilityConfig()
    constraints: List[ConstraintSpec] = Field(default_factory=list)

    @root_validator
    def _defaults(cls, v):  # noqa: N805
        E = v["instrument_map"].E
        n_dec = E.shape[1]
        if v["start_dec"].size == 0:
            v["start_dec"] = np.zeros(n_dec)
        if v["alpha_exp"] is None:
            v["alpha_exp"] = np.zeros(E.shape[0])
        return v

# ---------------------------------------------------------------------------
# 8. Result wrapper
# ---------------------------------------------------------------------------

class PortfolioResult(BaseModel):
    decision_weights: NDArray[np.floating]
    exposure_weights: NDArray[np.floating]
    obj_value: float
    risk_sys: float
    risk_spec: float
    cost: float

# ---------------------------------------------------------------------------
# 9. Optimiser
# ---------------------------------------------------------------------------

class ConvexFactorOptimizer:
    """Conic optimiser with synthetic‑instrument mapping."""

    def solve(self, cfg: ProblemConfig) -> PortfolioResult:  # noqa: C901
        if mf is None:
            raise ImportError("MOSEK not available in this environment.")

        E = cfg.instrument_map.E
        n_dec = E.shape[1]

        with mf.Model("synthetic_portfolio") as M:
            # Decision vars --------------------------------------------------
            w_dec = M.variable("w_dec", n_dec)
            w_exp = E @ w_dec  # computed expression (risk exposures)

            # Risk cones -----------------------------------------------------
            rm = cfg.risk_model
            B, F = rm.data.loadings, rm.data.factor_cov
            A_sys = B @ np.linalg.cholesky(F)  # (n_risk, n_factors)
            D_sqrt = np.sqrt(rm.data.specific_var)
            A_spec = np.diag(D_sqrt)

            t_sys = M.variable("t_sys")
            M.constraint(mf.vstack(0.5, t_sys, A_sys @ w_exp) == mf.Domain.inRotatedQCone())
            t_spec = M.variable("t_spec")
            M.constraint(mf.vstack(0.5, t_spec, A_spec @ w_exp) == mf.Domain.inRotatedQCone())

            # Transaction cost ---------------------------------------------
            if cfg.utility.cost_model is not None:
                delta_dec = w_dec - cfg.start_dec
                cost_expr = cfg.utility.cost_model.cost_soc(M, delta_dec)
            else:
                cost_expr = 0.0

            # Plug‑in constraints ------------------------------------------
            for c in cfg.constraints:
                c.apply(M, w_dec, w_exp)

            # Objective -----------------------------------------------------
            reward = cfg.alpha_dec @ w_dec + cfg.alpha_exp @ w_exp  # type: ignore[arg-type]
            penalty = (
                cfg.utility.risk_aversion_sys * t_sys
                + cfg.utility.risk_aversion_spec * t_spec
                + cost_expr
            )
            M.maximize(reward - penalty)
            M.setLogHandler(None)
            M.solve()

            return PortfolioResult(
                decision_weights=w_dec.level(),
                exposure_weights=(E @ w_dec).level(),
                obj_value=M.primalObjValue(),
                risk_sys=t_sys.level()[0],
                risk_spec=t_spec.level()[0],
                cost=float(cost_expr.level()[0]) if cfg.utility.cost_model else 0.0,
            )

# ---------------------------------------------------------------------------
# 10. Convenience helper
# ---------------------------------------------------------------------------

def optimize_portfolio(cfg: ProblemConfig) -> PortfolioResult:
    """Solve *cfg* with the default optimiser instance."""
    return ConvexFactorOptimizer().solve(cfg)
