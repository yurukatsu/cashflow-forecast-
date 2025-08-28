import luigi
import polars as pl
from gokart.parameter import TaskInstanceParameter

from src.configs import ExperimentConfig, ValidationMethodConfig
from src.tasks import LoadConfigTask, MlflowTask
from src.utils.preprocess import convert_date_to_numeric, remove_date_columns
from src.utils.splitters import TimeSeriesDataSplitter

from ._preprocess_before_split import PreprocessPipelineBeforeSplit


class TrainValidationSplitTask(MlflowTask):
    """
    Task for splitting the DataFrame into training and validation sets.
    """

    is_train = luigi.BoolParameter()
    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    preprocess_pipeline_before_split = TaskInstanceParameter(
        expected_type=PreprocessPipelineBeforeSplit
    )

    def requires(self):
        return {
            "config": self.load_config_task,
            "data": self.preprocess_pipeline_before_split,
        }

    def _run(
        self, validation_method_config: ValidationMethodConfig, date_column_name: str
    ):
        data = self.load("data")

        if not self.is_train:
            return data
        splitter = TimeSeriesDataSplitter(
            validation_method_config,
            date_column_name=date_column_name,
        )

        return list(splitter.split(data))

    @property
    def parameters(self) -> dict:
        config: ExperimentConfig = self.load("config")
        return {
            "validation_method_config": config.validation_method,
            "date_column_name": config.data.column.date,
        }


class ConvertDateToNumericTask(MlflowTask):
    """
    Task for converting date columns to numeric format.
    """

    is_train = luigi.BoolParameter()
    train_validation_split_task = TaskInstanceParameter(
        expected_type=TrainValidationSplitTask
    )

    def requires(self):
        return self.train_validation_split_task

    def _run(self):
        if not self.is_train:
            data = self.load()
            return convert_date_to_numeric(data)

        folds = []
        for train, val in self.load():
            train = convert_date_to_numeric(train)
            val = convert_date_to_numeric(val)
            folds.append((train, val))
        return folds

    @property
    def parameters(self) -> dict:
        return {}


class RemoveUnusedColumnsTask(MlflowTask):
    """
    Task for removing unused columns from the DataFrame.
    """

    is_train = luigi.BoolParameter()
    convert_date_to_numeric_task = TaskInstanceParameter(
        expected_type=ConvertDateToNumericTask
    )

    def requires(self):
        return self.convert_date_to_numeric_task

    def _run(self):
        if not self.is_train:
            data = self.load()
            return remove_date_columns(data)
        folds = []
        for train, val in self.load():
            train = remove_date_columns(train)
            val = remove_date_columns(val)
            folds.append((train, val))
        return folds

    @property
    def parameters(self) -> dict:
        return {}


class PreprocessPipeline(MlflowTask):
    """
    Preprocessing pipeline for the ML workflow.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    is_train = luigi.BoolParameter()

    def requires(self):
        preprocess_pipeline_before_split = PreprocessPipelineBeforeSplit(
            load_config_task=self.load_config_task, is_train=self.is_train
        )

        train_validation_split_task = TrainValidationSplitTask(
            is_train=self.is_train,
            load_config_task=self.load_config_task,
            preprocess_pipeline_before_split=preprocess_pipeline_before_split,
        )

        convert_date_to_numeric_task = ConvertDateToNumericTask(
            is_train=self.is_train,
            train_validation_split_task=train_validation_split_task,
        )

        remove_unused_columns_task = RemoveUnusedColumnsTask(
            is_train=self.is_train,
            convert_date_to_numeric_task=convert_date_to_numeric_task,
        )

        return remove_unused_columns_task

    def _run(self) -> pl.DataFrame:
        return self.load()

    @property
    def parameters(self) -> dict:
        return {}
