import numpy as np
from numpy.typing import NDArray

from .core import OptBaseModel


class FactorRiskModelData(OptBaseModel):
    loadings: NDArray[np.floating]
    factor_cov: NDArray[np.floating]
    specific_var: NDArray[np.floating]


class FactorRiskModel(OptBaseModel):
    data: FactorRiskModelData

    def systematic_risk(self, w_exp: NDArray[np.floating]) -> float:
        B, F = self.data.loadings, self.data.factor_cov
        return float(w_exp @ (B @ F @ B.T) @ w_exp)

    def specific_risk(self, w_exp: NDArray[np.floating]) -> float:
        return float((w_exp ** 2) @ self.data.specific_var)

    def total_risk(self, w_exp: NDArray[np.floating]) -> float:
        return self.systematic_risk(w_exp) + self.specific_risk(w_exp)
