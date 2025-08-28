from typing import Literal

from pydantic import Field

from ._base import BaseConfig

CrossValidationStrategy = Literal["sliding_window", "expanding_window"]


class ValidationMethodConfig(BaseConfig):
    """
    Validation method configuration schema.
    """

    strategy: CrossValidationStrategy = Field(
        default="sliding_window", description="Validation strategy."
    )
    n_splits: int = Field(5, description="Number of splits for cross-validation.")
    validation_duration: str = Field(
        ..., description="Duration of the validation period."
    )
    gap_duration: str | None = Field(
        ..., description="Duration of the gap between train and validation periods."
    )
    step_duration: str = Field(
        ..., description="Duration of the step for sliding window."
    )
    train_duration: str | None = Field(
        default=None, description="Duration of the training period."
    )
    train_start_date: str | None = Field(
        default=None, description="End date for the test period."
    )
    test_end_date: str | None = Field(
        default=None, description="End date for the test period."
    )
