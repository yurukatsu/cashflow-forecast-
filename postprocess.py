import mlflow
import polars as pl
import pandas as pd
import pickle

from src.configs import ExperimentConfig
from src.utils.preprocess import (
    coerce_booleans,
    convert_bool_to_numeric,
    remove_bool_columns,
    convert_date_to_numeric,
    remove_date_columns,
)
from src.tasks._base import from_uri


def get_parent_run_names(
    experiment_name: str = "TrainTask - src/tasks/_train",
) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    run_ids = runs["tags.mlflow.parentRunId"].dropna().unique()
    print(run_ids)
    run_names = runs[runs["run_id"].isin(run_ids)]["tags.mlflow.runName"].to_list()
    return run_names


def get_runs(
    experiment_name: str = "TrainTask - src/tasks/_train",
    parent_run_name: str = "CatBoostModel-20250901-235328",
) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    parent_run_id = runs[runs["tags.mlflow.runName"] == parent_run_name][
        "run_id"
    ].values[0]
    runs = runs[runs["tags.mlflow.parentRunId"] == parent_run_id]
    return runs


def get_data() -> pl.DataFrame:
    df_train = pl.read_csv("data/raw/train.csv", try_parse_dates=True)
    df_test = pl.read_csv("data/raw/test.csv", try_parse_dates=True)

    return df_train, df_test


def preprocess(data: pl.DataFrame, config: ExperimentConfig):
    data = data.drop(config.data.column.ignore_columns)
    data = coerce_booleans(data, config.data.column.bool_columns)
    data = convert_bool_to_numeric(data)
    data = remove_bool_columns(data)
    data = data.with_columns(
        [pl.col(c).cast(pl.Int64) for c in config.data.column.int_columns]
    )
    data = convert_date_to_numeric(data)
    data = remove_date_columns(data)
    return data


if __name__ == "__main__":
    from pathlib import Path

    config = ExperimentConfig.from_yaml("configs/experiments/catboost06.yml")

    df_train, df_test = get_data()
    df_train_processed = preprocess(df_train, config)
    df_test_processed = preprocess(df_test, config)

    X_train = df_train_processed.drop(config.data.column.target)
    y_train = df_train_processed[config.data.column.target]
    X_test = df_test_processed.drop(config.data.column.target)
    y_test = df_test_processed[config.data.column.target]

    experiment_name: str = "TrainTask - src/tasks/_train"

    def inference(parent_run_name: str):
        save_path = Path(f"./data/processed/{parent_run_name}")
        if save_path.exists():
            print("Already processed:", parent_run_name)
            return
        save_path.mkdir(parents=True, exist_ok=True)

        runs = get_runs(experiment_name, parent_run_name)
        for i, run in runs.iterrows():
            run_name = run["tags.mlflow.runName"]

            if run_name == "ensemble":
                continue

            artifact_uri = run["artifact_uri"]
            artifact_path = from_uri(artifact_uri)
            model_path = artifact_path / "models/model_class.pkl"

            if parent_run_name.startswith("CatBoost"):
                with model_path.open("rb") as f:
                    from src.models.catboost import CatBoostModel

                    model: CatBoostModel = pickle.load(f)

            if parent_run_name.startswith("LightGBM"):
                with model_path.open("rb") as f:
                    from src.models.lgbm import LightGBMModel

                    model: LightGBMModel = pickle.load(f)

            if parent_run_name.startswith("XGBoost"):
                with model_path.open("rb") as f:
                    from src.models.xgboost import XGBoostModel

                    model: XGBoostModel = pickle.load(f)

            y_train_pred = model.predict(X_train)
            result_train = pl.concat(
                [
                    df_train.select(
                        [
                            config.data.column.date,
                            config.data.column.company_id,
                            config.data.column.company_name,
                            config.data.column.target,
                        ]
                    ),
                    y_train_pred,
                ],
                how="horizontal",
            ).with_columns(pl.lit(False).alias("is_test"))

            y_test_pred = model.predict(X_test)
            result_test = pl.concat(
                [
                    df_test.select(
                        [
                            config.data.column.date,
                            config.data.column.company_id,
                            config.data.column.company_name,
                            config.data.column.target,
                        ]
                    ),
                    y_test_pred,
                ],
                how="horizontal",
            ).with_columns(pl.lit(True).alias("is_test"))
            result = pl.concat([result_train, result_test]).with_columns(
                [
                    pl.lit(parent_run_name).alias("model"),
                    pl.lit(run_name).alias("fold"),
                ]
            )
            result.write_csv(save_path / f"{run_name}.csv")

    for parent_run_name in get_parent_run_names(experiment_name):
        try:
            inference(parent_run_name)
        except Exception as e:
            print("Error processing:", parent_run_name)
            print(e)
