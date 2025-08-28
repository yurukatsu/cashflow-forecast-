from typing import Any

from pydantic import Field

from ._base import BaseConfig


class OptunaConfig(BaseConfig):
    """
    Optuna hyperparameter optimization configuration.
    """
    
    enabled: bool = Field(
        default=False, description="Whether to enable Optuna optimization."
    )
    n_trials: int = Field(
        default=100, description="Number of trials for optimization."
    )
    optimize_metric: str = Field(
        default="rmse_val", description="Metric to optimize (minimize)."
    )
    use_pruning: bool = Field(
        default=True, description="Whether to use pruning for early stopping."
    )
    n_startup_trials: int = Field(
        default=5, description="Number of startup trials before pruning."
    )
    n_warmup_steps: int = Field(
        default=1, description="Number of warmup steps for pruning."
    )
    search_space: dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameter search space definition."
    )


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
    feature_importance_params: dict[str, Any] | None = Field(
        default=None, description="Parameters for feature importance extraction."
    )
    optuna_config: OptunaConfig | None = Field(
        default=None, description="Optuna hyperparameter optimization configuration."
    )
