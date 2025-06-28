import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from .core import OptBaseModel


class PortfolioResult(OptBaseModel):
    decision_weights: NDArray[np.floating]
    exposure_weights: NDArray[np.floating]
    obj_value: float
    risk_sys: float
    risk_spec: float
    cost: float

    @staticmethod
    def _as_numpy(v: object) -> object:
        """Return numbers/arrays from MOSEK expressions or raw values."""
        if hasattr(v, "level"):
            v = v.level()
        return np.asarray(v, dtype=float) if np.ndim(v) else float(v)

    @model_validator(mode="before")
    def _coerce(cls, values: dict) -> dict:  # noqa: D401,N805 - pydantic API
        """Coerce MOSEK expression types into plain numpy types."""
        out = dict(values)
        for k in ("decision_weights", "exposure_weights"):
            out[k] = cls._as_numpy(out[k])
        for k in ("obj_value", "risk_sys", "risk_spec", "cost"):
            out[k] = float(cls._as_numpy(out[k]))
        return out
