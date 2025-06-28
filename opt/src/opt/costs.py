from typing import Any, Protocol

from pydantic import Field

from .core import OptBaseModel
from .utils import abs_expr


class TransactionCostModel(Protocol):
    def cost_soc(self, M: "mf.Model", delta_dec: "mf.Expr") -> "mf.Expr": ...  # noqa: F821


class PowerLawCost(OptBaseModel):
    exponent: float = Field(1.5, gt=1, lt=2)
    scale: float = 1e-4

    def cost_soc(self, M: "mf.Model", delta_dec: "mf.Expr") -> "mf.Expr":  # noqa: F821
        import mosek.fusion as mf

        import uuid

        beta = abs_expr(M, delta_dec, f"beta_cost_{uuid.uuid4().hex}")
        return self.scale * mf.Expr.sum(beta)
