from typing import Any

from pydantic import Field

from ._base import BaseConfig


class ModelConfig(BaseConfig):
    """
    Validation method configuration schema.
    """

    name: str = Field(..., description="Name of the model.")
    model_class: str = Field(..., description="Full class path of the model.")
    init_params: dict[str, Any] | None = Field(
        default=None, description="Parameters for model initialization."
    )
    fit_params: dict[str, Any] | None = Field(
        default=None, description="Parameters for model fitting."
    )
    predict_params: dict[str, Any] | None = Field(
        default=None, description="Parameters for model prediction."
    )
