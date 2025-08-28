import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

import mlflow
from gokart.conflict_prevention_lock.task_lock import make_task_lock_params
from gokart.file_processor import FileProcessor, make_file_processor
from gokart.target import (
    SingleFileTarget,
    TargetOnKart,
    _make_file_system_target,
    make_target,
)
from gokart.task import TaskOnKart
from mlflow.entities import Run
from mlflow.protos.service_pb2 import ACTIVE_ONLY
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME


def from_uri(uri: str) -> Path:
    """
    Return a new path from the given 'file' URI.
    """
    if not uri.startswith("file:"):
        msg = f"URI does not start with 'file:': {uri!r}"
        raise ValueError(msg)
    path = uri[5:]
    if path[:3] == "///":
        # Remove empty authority
        path = path[2:]
    elif path[:12] == "//localhost/":
        # Remove 'localhost' authority
        path = path[11:]
    if path[:3] == "///" or (path[:1] == "/" and path[2:3] in ":|"):
        # Remove slash before DOS device/UNC path
        path = path[1:]
    if path[1:2] == "|":
        # Replace bar with colon in DOS drive
        path = path[:1] + ":" + path[2:]
    from urllib.parse import unquote_to_bytes

    path = Path(os.fsdecode(unquote_to_bytes(path)))
    if not path.is_absolute():
        msg = f"URI is not absolute: {uri!r}"
        raise ValueError(msg)
    return path


class MlflowTask(TaskOnKart):
    """
    Abstract class for MLflow tasks.
    """

    metrics: dict[str, float] = {}

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the MLflow task.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        Parameters for the MLflow task. The parameters will be passed to _run.
        """
        raise NotImplementedError

    @property
    def mlflow_experiment_name(self) -> str:
        """
        Experiment name for MLflow.
        """
        return f"{type(self).__name__} - {self.__module__.replace('.', '/')}"

    @property
    def mlflow_run_name(self) -> str:
        """
        Run name for MLflow.
        """
        return self.make_unique_id()

    def run(self) -> None:
        """
        Run the MLflow task.
        """
        mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)

        with mlflow.start_run():
            mlflow.set_tag(MLFLOW_RUN_NAME, self.mlflow_run_name)
            result = self._run(**self.parameters)
            self.dump(result)
            self._log_parameters()
            self._log_artifacts()
            self._log_metrics()

    def search_exist_mlflow_run(self) -> Run | None:
        """
        Search for an existing MLflow run.
        """
        experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)

        if experiment is None:
            return None

        result = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f'tags.{MLFLOW_RUN_NAME}="{self.mlflow_run_name}"',
            run_view_type=ACTIVE_ONLY,
            output_format="list",
            max_results=1,
        )

        if len(result) > 0:
            return Run.from_proto(result[0].to_proto())
        return None

    def make_target(
        self,
        relative_file_path: str | None = None,
        use_unique_id: bool = True,
        processor: FileProcessor | None = None,
    ) -> TargetOnKart:
        """
        Make target with mlflow run.

        :param relative_file_path: The relative file path to save the target. If None, it will be set to "<module_path>/<class_name>.pkl".
        :type relative_file_path: str | None
        :param use_unique_id: Whether to use a unique ID for the target.
        :type use_unique_id: bool
        :param processor: The file processor to use for the target.
        :type processor: FileProcessor | None

        :return: The target for the MLflow run.
        :rtype: TargetOnKart
        """
        maybe_mlf_run = self.search_exist_mlflow_run()
        formatted_relative_file_path = str(
            relative_file_path
            if relative_file_path is not None
            else Path(self.__module__.replace(".", "/")) / f"{type(self).__name__}.pkl"
        )
        unique_id = self.make_unique_id() if use_unique_id else None

        task_lock_params = make_task_lock_params(
            file_path=formatted_relative_file_path,
            unique_id=unique_id,
            redis_host=self.redis_host,
            redis_port=self.redis_port,
            redis_timeout=self.redis_timeout,
            raise_task_lock_exception_on_collision=False,
        )

        if maybe_mlf_run is None or maybe_mlf_run.info.artifact_uri is None:
            # Use gokart's workspace directory instead of temporary directory
            file_path = str(
                Path(self.workspace_directory) / formatted_relative_file_path
            )

            return make_target(
                file_path=file_path,
                unique_id=unique_id,
                processor=processor,
                task_lock_params=task_lock_params,
                store_index_in_feather=self.store_index_in_feather,
            )

        artifact_dir = from_uri(maybe_mlf_run.info.artifact_uri)
        file_path = str(artifact_dir / formatted_relative_file_path)

        processor = processor or make_file_processor(
            file_path, store_index_in_feather=self.store_index_in_feather
        )
        file_system_target = _make_file_system_target(
            file_path,
            processor=processor,
            store_index_in_feather=self.store_index_in_feather,
        )

        return SingleFileTarget(
            target=file_system_target,
            processor=processor,
            task_lock_params=task_lock_params,
        )

    def _log_parameters(self) -> None:
        """
        Log parameters.
        """
        params = {
            "gokart_requires": self.requires(),
            **self.parameters,
        }
        mlflow.log_params(params)

    def _log_artifacts(self) -> None:
        """
        Log artifacts.
        """
        output_path = self.output().path()  # type: ignore
        if Path(output_path).exists():
            mlflow.log_artifact(output_path)

    def _log_metrics(self) -> None:
        """
        Log metrics.
        """
        if self.metrics:
            mlflow.log_metrics(self.metrics)
