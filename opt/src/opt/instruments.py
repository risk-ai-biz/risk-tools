from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from .core import OptBaseModel


class InstrumentMap(OptBaseModel):
    """Dense matrix **E** mapping decision variables → risk exposures."""

    E: NDArray[np.floating]
    names_risk: List[str]
    names_decision: List[str]

    @field_validator("E")
    def _2d(cls, v):  # noqa: N805
        if v.ndim != 2:
            raise ValueError("Instrument matrix must be 2-D")
        return v

    @property
    def shape(self) -> Tuple[int, int]:
        return self.E.shape

    @classmethod
    def identity(cls, n: int) -> "InstrumentMap":
        labels = [f"A{i}" for i in range(n)]
        return cls(E=np.eye(n), names_risk=labels, names_decision=labels)
