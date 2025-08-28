import logging

import luigi

from src.configs import ExperimentConfig
from src.tasks import MlflowTask

logger = logging.getLogger(__name__)


class LoadConfigTask(MlflowTask):
    experiment_config_path = luigi.Parameter()

    def _run(self, experiment_config_path: str):
        logger.info(f"Loading experiment config from {experiment_config_path}")
        config = ExperimentConfig.from_yaml(experiment_config_path)
        logger.info(f"Loaded experiment config: {config}")
        return config

    @property
    def parameters(self) -> dict:
        return {"experiment_config_path": self.experiment_config_path}
