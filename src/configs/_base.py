from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file from the specified path and return its contents as a dictionary.

    Args:
        path (Path):
            The path to the YAML file.

    Returns:
        dict[str, Any]:
            The contents of the YAML file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseConfig(BaseModel):
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        """
        Load configuration from a YAML file.
        """
        config_data = load_yaml(path)
        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """
        Load configuration from a dictionary.
        """
        return cls(**data)
