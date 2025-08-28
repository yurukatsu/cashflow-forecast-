import luigi
import polars as pl
from gokart.parameter import TaskInstanceParameter

from src.configs import ExperimentConfig
from src.tasks import LoadConfigTask, MlflowTask
from src.utils.preprocess import (
    coerce_booleans,
    convert_bool_to_numeric,
    impute_booleans,
    remove_bool_columns,
)


class IngestDataTask(MlflowTask):
    """
    Task for ingesting training data.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    is_train = luigi.BoolParameter()

    def requires(self):
        return self.load_config_task

    def _run(
        self, data_path: str, col_company_id: str, filter_company_ids: list[str] | None
    ) -> pl.DataFrame:
        df = pl.read_csv(data_path, try_parse_dates=True)
        if filter_company_ids:
            df = df.filter(pl.col(col_company_id).is_in(filter_company_ids))
        return df

    @property
    def parameters(self) -> dict:
        config: ExperimentConfig = self.load()
        return {
            "data_path": config.data.train_path
            if self.is_train
            else config.data.test_path,
            "col_company_id": config.data.column.company_id,
            "filter_company_ids": config.data.preprocess.filter_company_ids,
        }


class DropIgnoredColumnsTask(MlflowTask):
    """
    Task for dropping ignored columns from the DataFrame.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    ingest_data_task = TaskInstanceParameter(expected_type=IngestDataTask)

    def requires(self):
        return {"config": self.load_config_task, "data": self.ingest_data_task}

    def _run(self, ignore_columns: list[str]) -> pl.DataFrame:
        data: pl.DataFrame = self.load("data")
        if ignore_columns:
            data = data.drop(ignore_columns)
        return data

    @property
    def parameters(self) -> dict:
        config: ExperimentConfig = self.load("config")
        return {
            "ignore_columns": config.data.column.ignore_columns,
        }


class CoerceBooleansTask(MlflowTask):
    """
    Task for coercing boolean columns in the DataFrame.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    drop_ignored_columns_task = TaskInstanceParameter(
        expected_type=DropIgnoredColumnsTask
    )

    def requires(self):
        return {"config": self.load_config_task, "data": self.drop_ignored_columns_task}

    def _run(self, bool_columns: list[str]) -> pl.DataFrame:
        data: pl.DataFrame = self.load("data")
        if bool_columns:
            data = coerce_booleans(data, bool_columns)
        return data

    @property
    def parameters(self) -> dict:
        config: ExperimentConfig = self.load("config")
        return {
            "bool_columns": config.data.column.bool_columns,
        }


class ImputeBooleansTask(MlflowTask):
    """
    Task for imputing boolean columns in the DataFrame.
    """

    coerce_booleans_task = TaskInstanceParameter(expected_type=CoerceBooleansTask)

    def requires(self):
        return self.coerce_booleans_task

    def _run(self) -> pl.DataFrame:
        data: pl.DataFrame = self.load()
        return impute_booleans(data)

    @property
    def parameters(self) -> dict:
        return {}


class ConvertBoolToNumericTask(MlflowTask):
    """
    Task for converting boolean columns to numeric format.
    """

    coerce_booleans_task = TaskInstanceParameter(expected_type=CoerceBooleansTask)

    def requires(self):
        return self.coerce_booleans_task

    def _run(self) -> pl.DataFrame:
        data: pl.DataFrame = self.load()
        return convert_bool_to_numeric(data)

    @property
    def parameters(self) -> dict:
        return {}


class RemoveBoolColumnsTask(MlflowTask):
    """
    Task for removing boolean columns from the DataFrame.
    """

    convert_bool_to_numeric_task = TaskInstanceParameter(
        expected_type=ConvertBoolToNumericTask
    )

    def requires(self):
        return self.convert_bool_to_numeric_task

    def _run(self) -> pl.DataFrame:
        data: pl.DataFrame = self.load()
        return remove_bool_columns(data)

    @property
    def parameters(self) -> dict:
        return {}


class CoerceIntColumnsTask(MlflowTask):
    """
    Task for coercing integer columns in the DataFrame.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    remove_bool_columns_task = TaskInstanceParameter(
        expected_type=RemoveBoolColumnsTask
    )

    def requires(self):
        return {"config": self.load_config_task, "data": self.remove_bool_columns_task}

    def _run(self, int_columns: list[str]) -> pl.DataFrame:
        data: pl.DataFrame = self.load("data")
        if int_columns:
            data = data.with_columns([pl.col(c).cast(pl.Int64) for c in int_columns])
        return data

    @property
    def parameters(self) -> dict:
        config: ExperimentConfig = self.load("config")
        return {
            "int_columns": config.data.column.int_columns,
        }


class PreprocessPipelineBeforeSplit(MlflowTask):
    """
    Preprocessing pipeline for the ML workflow.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    is_train = luigi.BoolParameter()

    def requires(self):
        ingest_data_task = IngestDataTask(
            load_config_task=self.load_config_task,
            is_train=self.is_train,
        )

        drop_ignored_columns_task = DropIgnoredColumnsTask(
            load_config_task=self.load_config_task,
            ingest_data_task=ingest_data_task,
        )

        coerce_booleans_task = CoerceBooleansTask(
            load_config_task=self.load_config_task,
            drop_ignored_columns_task=drop_ignored_columns_task,
        )

        convert_bool_to_numeric_task = ConvertBoolToNumericTask(
            coerce_booleans_task=coerce_booleans_task,
        )

        remove_bool_columns_task = RemoveBoolColumnsTask(
            convert_bool_to_numeric_task=convert_bool_to_numeric_task,
        )

        coerce_int_columns_task = CoerceIntColumnsTask(
            load_config_task=self.load_config_task,
            remove_bool_columns_task=remove_bool_columns_task,
        )

        return coerce_int_columns_task

    def _run(self) -> pl.DataFrame:
        return self.load()

    @property
    def parameters(self) -> dict:
        return {}
