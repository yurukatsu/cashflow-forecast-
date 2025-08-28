import datetime
import tempfile
from pathlib import Path

import luigi
import mlflow
import gokart
import polars as pl

from src.configs import ExperimentConfig
from src.tasks import LoadConfigTask, MlflowTask, TrainTask
from src.tasks.preprocess import PreprocessPipeline


class ExperimentPipeline(MlflowTask):
    experiment_config_path = luigi.Parameter()

    def requires(self):
        load_config_task = LoadConfigTask(
            experiment_config_path=self.experiment_config_path
        )
        train_preprocess_pipeline = PreprocessPipeline(
            load_config_task=load_config_task,
            is_train=True,
        )
        test_preprocess_pipeline = PreprocessPipeline(
            load_config_task=load_config_task,
            is_train=False,
        )

        return {
            "config": load_config_task,
            "train": TrainTask(
                load_config_task=load_config_task,
                train_preprocess_pipeline=train_preprocess_pipeline,
                test_preprocess_pipeline=test_preprocess_pipeline,
            ),
        }

    def _run(self) -> pl.DataFrame:
        task_tree = gokart.make_task_info_as_tree_str(
            self,
            details=True,
            abbr=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            task_tree_path = out_dir / "task_tree.txt"
            with open(task_tree_path, "w") as f:
                f.write(task_tree)

            mlflow.log_artifact(str(task_tree_path), artifact_path="tasks")
        return self.load("train")

    @property
    def parameters(self) -> dict:
        return {}

    @property
    def mlflow_run_name(self) -> str:
        """
        Run name for MLflow.
        """
        # Check if config file exists before loading
        config_target = self._get_input_targets("config")
        if not config_target.exists():
            # Return a default name if config doesn't exist yet
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"model-{now}"

        config: ExperimentConfig = self.load("config")
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{config.model.name}-{config.data.name}-{now}"
