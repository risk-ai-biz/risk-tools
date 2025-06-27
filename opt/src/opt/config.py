from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from .core import OptBaseModel
from .constraints import ConstraintSpec
from .instruments import InstrumentMap
from .risk import FactorRiskModel


class UtilityConfig(OptBaseModel):
    risk_aversion_sys: float = 1.0
    risk_aversion_spec: float = 1.0
    cost_model: Optional[Any] = None


class ProblemConfig(OptBaseModel):
    risk_model: FactorRiskModel
    instrument_map: InstrumentMap
    alpha_dec: NDArray[np.floating]
    alpha_exp: Optional[NDArray[np.floating]] = None
    start_dec: NDArray[np.floating] = Field(default_factory=lambda: np.array([]))
    utility: UtilityConfig = UtilityConfig()
    constraints: List[ConstraintSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _defaults(cls, v: "ProblemConfig") -> "ProblemConfig":
        E = v.instrument_map.E
        n_dec = E.shape[1]
        if v.start_dec.size == 0:
            v.start_dec = np.zeros(n_dec)
        if v.alpha_exp is None:
            v.alpha_exp = np.zeros(E.shape[0])
        return v
