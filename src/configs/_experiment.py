from pathlib import Path

from pydantic import Field

from ._base import BaseConfig, load_yaml
from ._data import DataConfig
from ._model import ModelConfig
from ._validation_method import ValidationMethodConfig


class ExperimentConfig(BaseConfig):
    """
    Experiment configuration schema.
    """

    name: str = Field(..., description="Name of the experiment.")
    model: ModelConfig = Field(..., description="Configuration for the model.")
    data: DataConfig = Field(..., description="Configuration for the data.")
    validation_method: ValidationMethodConfig = Field(
        ..., description="Configuration for the validation method."
    )

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "ExperimentConfig":
        data = load_yaml(file_path)
        return cls(
            name=data["name"],
            model=ModelConfig.from_yaml(data["config_path"]["model"]),
            data=DataConfig.from_yaml(data["config_path"]["data"]),
            validation_method=ValidationMethodConfig.from_yaml(
                data["config_path"]["validation_method"]
            ),
        )
