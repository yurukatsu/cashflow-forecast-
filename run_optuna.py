import logging
import shutil
from pathlib import Path

import click
import gokart

from src.pipeline_optuna import OptunaExperimentPipeline


@click.command()
@click.option(
    "--experiment-config-path",
    default="configs/experiments/xgboost_optuna.yml",
    help="Path to experiment configuration file with Optuna settings",
    type=click.Path(exists=True),
)
@click.option(
    "--show-tree",
    is_flag=True,
    default=True,
    help="Show task dependency tree before execution",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set the logging level",
)
@click.option(
    "--scheduler-url",
    default="http://0.0.0.0:18082",
    help="URL of the Luigi scheduler",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Force re-run by clearing cache directories (mlruns and resources)",
)
def main(experiment_config_path, show_tree, log_level, scheduler_url, force_rerun):
    """Run the Optuna optimization pipeline with the specified configuration."""

    if force_rerun:
        click.echo("Force re-run enabled. Clearing cache directories...")

        mlruns_path = Path("mlruns")
        if mlruns_path.exists():
            for item in mlruns_path.iterdir():
                if item.is_dir() and item.name != "0" and item.name != "models":
                    click.echo(f"  Removing {item}")
                    shutil.rmtree(item)

        resources_path = Path("resources")
        if resources_path.exists():
            click.echo(f"  Removing {resources_path}")
            shutil.rmtree(resources_path)

        click.echo("Cache cleared.\n")

    pipeline = OptunaExperimentPipeline(experiment_config_path=experiment_config_path)

    if show_tree:
        click.echo("Task dependency tree:")
        click.echo(gokart.make_task_info_as_tree_str(pipeline))

    log_level_int = getattr(logging, log_level)

    click.echo(f"Running Optuna pipeline with config: {experiment_config_path}")
    click.echo("This will perform hyperparameter optimization using Optuna.")
    gokart.build(
        task=pipeline,
        log_level=log_level_int,
        return_value=False,
        reset_register=True,
        scheduler_url=scheduler_url,
    )


if __name__ == "__main__":
    main()