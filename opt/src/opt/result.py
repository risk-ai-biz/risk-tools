import numpy as np
from numpy.typing import NDArray

from .core import OptBaseModel


class PortfolioResult(OptBaseModel):
    decision_weights: NDArray[np.floating]
    exposure_weights: NDArray[np.floating]
    obj_value: float
    risk_sys: float
    risk_spec: float
    cost: float
