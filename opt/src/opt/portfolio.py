from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .core import OptBaseModel, QuantityType


class Portfolio(OptBaseModel):
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
