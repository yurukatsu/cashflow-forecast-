from pathlib import Path

from pydantic import Field

from ._base import BaseConfig, load_yaml


class PreprocessConfig(BaseConfig):
    """
    Preprocessing configuration schema.
    """

    filter_company_ids: list[str] | None = Field(
        default=None,
        description="List of company IDs to filter. If not specified, all companies are included.",
    )


class ColumnConfig(BaseConfig):
    """
    Data column configuration schema.
    """

    target: str = Field(..., description="Name of the target column.")
    date: str = Field(..., description="Name of the date column.")
    company_id: str = Field(..., description="Name of the company ID column.")
    company_name: str = Field(..., description="Name of the company name column.")
    bool_columns: list[str] | None = Field(
        default=None, description="List of boolean columns."
    )
    str_columns: list[str] | None = Field(
        default=None, description="List of string columns."
    )
    int_columns: list[str] | None = Field(
        default=None, description="List of integer columns."
    )
    ignore_columns: list[str] | None = Field(
        default=None, description="List of columns to ignore."
    )


class DataConfig(BaseConfig):
    """
    Data configuration schema.
    """

    name: str = Field(..., description="Name of the dataset.")
    train_path: str = Field(..., description="Path to the training data file.")
    test_path: str = Field(..., description="Path to the testing data file.")
    column: ColumnConfig = Field(..., description="Configuration for the data column.")
    preprocess: PreprocessConfig = Field(
        ..., description="Configuration for preprocessing."
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        """
        Load configuration from a YAML file.
        """
        data = load_yaml(path)
        return cls(
            name=data["name"],
            train_path=data["train_path"],
            test_path=data["test_path"],
            column=ColumnConfig.from_dict(data["column"]),
            preprocess=PreprocessConfig.from_dict(data["preprocess"]),
        )
