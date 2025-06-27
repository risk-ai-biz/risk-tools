import enum
from pydantic import BaseModel, ConfigDict

class QuantityType(str, enum.Enum):
    WEIGHT = "weight"
    NOTIONAL = "notional"


class OptBaseModel(BaseModel):
    """Base model with numpy compatibility."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
