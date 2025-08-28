import datetime
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import mlflow
import numpy as np
import plotly.graph_objects as go
import polars as pl
from gokart.parameter import TaskInstanceParameter

from src.configs import ExperimentConfig
from src.models.base import BaseRegressionModel
from src.tasks import LoadConfigTask, MlflowTask
from src.tasks.preprocess import IngestDataTask, PreprocessPipeline
from src.utils.figure import plot_target_prediction_by_company

logger = logging.getLogger(__name__)


class Output:
    def __init__(self, df: pl.DataFrame, config: ExperimentConfig):
        self.df = df
        self.config = config

    def to_figure(self) -> go.Figure:
        return plot_target_prediction_by_company(
            self.df,
            date_col=self.config.data.column.date,
            company_col=self.config.data.column.company_name,
            target_col=self.config.data.column.target,
            pred_col="prediction",
        )

    def save_files(
        self,
        out_dir: Path,
        basename: str = "prediction",
        save_html: bool = True,
        save_png: bool = True,
        width: int = 1000,
        height: int = 800,
        scale: float = 2.0,
    ) -> dict:
        """
        図をファイル出力する。戻り値は {"html": Path|None, "png": Path|None}
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = self.to_figure()
        paths = {"html": None, "png": None}

        if save_html:
            html_path = out_dir / f"{basename}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
            paths["html"] = html_path

        if save_png:
            png_path = out_dir / f"{basename}.png"
            try:
                fig.write_image(str(png_path), width=width, height=height, scale=scale)
                paths["png"] = png_path
            except Exception as e:
                print(
                    f"[WARN] PNG 書き出しに失敗: {e}\n"
                    f"       `pip install -U kaleido` を検討してください。"
                )
                paths["png"] = None

        return paths

    def save_csv(self, out_dir: Path, basename: str = "prediction") -> Path:
        """
        DataFrame を CSV で保存して Path を返す。
        （UTF-8、ヘッダあり。Excel で開く場合はBOM付与が必要なら別途対応）
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{basename}.csv"
        # polars は UTF-8 で書き出し。float はそのまま
        self.df.write_csv(str(csv_path))
        return csv_path

    def log_to_mlflow(
        self,
        artifact_subdir: str | None = "figures",
        basename: str = "prediction",
        log_html: bool = True,
        log_png: bool = True,
        log_csv: bool = True,
        csv_subdir: str | None = "tables",
    ) -> dict:
        """
        MLflow に HTML と PNG を記録する。
        - HTML はインタラクティブ表示用（UIではリンク表示）
        - PNG は UI 上にサムネイル表示（MLflow 2.3+ で `log_image` 推奨）
        戻り値は {"html": bool, "png": bool}
        """
        recorded = {"html": False, "png": False, "csv": False}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self.save_files(
                Path(tmpdir), basename, save_html=log_html, save_png=log_png
            )
            # HTML は artifact として保存
            if log_html and out["html"] is not None:
                mlflow.log_artifact(str(out["html"]), artifact_path=artifact_subdir)
                recorded["html"] = True

            # PNG は可能ならサムネイル表示されるAPIで
            if log_png and out["png"] is not None:
                try:
                    # MLflow 2.3+ （なければ log_artifact にフォールバック）
                    mlflow.log_image(
                        str(out["png"]),
                        artifact_file=f"{artifact_subdir}/{basename}.png",
                    )
                except Exception:
                    mlflow.log_artifact(str(out["png"]), artifact_path=artifact_subdir)
                recorded["png"] = True

            if log_csv:
                csv_path = self.save_csv(Path(tmpdir), basename=basename)
                mlflow.log_artifact(
                    str(csv_path),
                    artifact_path=csv_subdir if csv_subdir else artifact_subdir,
                )
                recorded["csv"] = True

        return recorded


class TrainTask(MlflowTask):
    """
    Generic task for training regression models and logging metrics to MLflow.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    train_preprocess_pipeline = TaskInstanceParameter(expected_type=PreprocessPipeline)
    test_preprocess_pipeline = TaskInstanceParameter(expected_type=PreprocessPipeline)

    def requires(self):
        ingest_data_task = IngestDataTask(
            load_config_task=self.load_config_task,
            is_train=False,
        )
        return {
            "config": self.load_config_task,
            "train_val_pairs": self.train_preprocess_pipeline,
            "test": self.test_preprocess_pipeline,
            "test_raw": ingest_data_task,
        }

    @staticmethod
    def _get_model_class(model_class: str) -> Type[BaseRegressionModel]:
        """
        Dynamically import and return the model class.
        """
        module_path, class_name = model_class.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _prepare_data(self, df, target_column: str):
        """
        Prepare features and target from dataframe.

        Args:
            df: Input dataframe
            target_column: Name of the target column

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        X = df.drop(target_column)
        y = df.select(target_column)
        return X, y

    def _create_output(self, pred: pl.DataFrame) -> Output:
        """
        Create output dataframe from predictions.

        Args:
            pred: DataFrame containing predictions

        Returns:
            DataFrame formatted for output
        """
        config: ExperimentConfig = self.load("config")
        test_raw: pl.DataFrame = self.load("test_raw")
        test_raw = test_raw.select(
            [
                config.data.column.date,
                config.data.column.company_id,
                config.data.column.company_name,
                config.data.column.target,
            ]
        )
        output = pl.concat([test_raw, pred], how="horizontal")
        return Output(output, config)

    def _train_single_fold(
        self,
        model: BaseRegressionModel,
        train_df,
        val_df,
        target_column: str,
        fit_params: dict,
        predict_params: dict,
    ) -> Tuple[dict, dict]:
        """
        Train model on a single fold and evaluate.

        Args:
            model: Model instance to train
            train_df: Training data
            val_df: Validation data
            target_column: Name of target column
            fit_params: Parameters for model.fit()
            predict_params: Parameters for model.predict()

        Returns:
            Tuple of (validation_metrics, test_metrics)
        """
        X_train, y_train = self._prepare_data(train_df, target_column)
        X_val, y_val = self._prepare_data(val_df, target_column)

        # Train model
        if model.val_is_required:
            model.fit(X_train, y_train, X_val, y_val, **fit_params)
        else:
            model.fit(X_train, y_train, **fit_params)

        # Evaluate on validation set
        y_pred = model.predict(X_val, **predict_params)
        val_metrics = model.evaluate(y_val, y_pred, suffix="_val")

        return val_metrics

    def _evaluate_on_test(
        self,
        model: BaseRegressionModel,
        X_test,
        y_test,
        predict_params: dict,
        return_predictions: bool = False,
    ) -> dict:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            predict_params: Parameters for model.predict()
            return_predictions: Whether to return predictions along with metrics

        Returns:
            Test metrics dictionary, optionally with predictions
        """
        y_test_pred = model.predict(X_test, **predict_params)
        test_metrics = model.evaluate(y_test, y_test_pred, suffix="_test")

        if return_predictions:
            return test_metrics, y_test_pred
        return test_metrics

    def _calculate_average_metrics(
        self, metrics_list: List[dict], prefix: str = ""
    ) -> dict:
        """
        Calculate average and standard deviation of metrics across folds.

        Args:
            metrics_list: List of metrics dictionaries from each fold
            prefix: Optional prefix for metric names

        Returns:
            Dictionary with averaged metrics and standard deviations
        """
        avg_metrics = {}
        if not metrics_list:
            return avg_metrics

        metric_names = list(metrics_list[0].keys())

        for metric_name in metric_names:
            fold_values = [
                fold_metrics.get(metric_name) for fold_metrics in metrics_list
            ]
            avg_metrics[f"{prefix}{metric_name}"] = np.mean(fold_values)
            avg_metrics[f"{prefix}std_{metric_name}"] = np.std(fold_values)

        return avg_metrics

    def _log_fold_metrics(self, val_metrics: dict, test_metrics: dict):
        """
        Log metrics for a single fold to MLflow.

        Args:
            val_metrics: Validation metrics
            test_metrics: Test metrics
        """
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

    def _calculate_ensemble_predictions(self, predictions_list: List) -> pl.DataFrame:
        """
        Calculate ensemble predictions by averaging all fold predictions.

        Args:
            predictions_list: List of predictions from each fold

        Returns:
            Averaged ensemble predictions as a polars DataFrame
        """
        # Stack all predictions and calculate mean along fold axis
        ensemble_array = np.mean(predictions_list, axis=0)

        # Convert to polars DataFrame
        # Assuming predictions are single column, but handle both cases
        if ensemble_array.ndim == 1:
            ensemble_array = ensemble_array.reshape(-1, 1)

        # Get column names from first prediction if it's a DataFrame
        if isinstance(predictions_list[0], pl.DataFrame):
            column_names = predictions_list[0].columns
        else:
            # Default column name if predictions are arrays
            column_names = [f"prediction_{i}" for i in range(ensemble_array.shape[1])]

        return pl.DataFrame(ensemble_array, schema=column_names)

    def _log_ensemble_results(
        self,
        ensemble_predictions,
        y_test,
        ModelClass: Type[BaseRegressionModel],
    ):
        """
        Log ensemble predictions and metrics to MLflow.

        Args:
            ensemble_predictions: Averaged predictions from all folds
            y_test: True test labels
            ModelClass: Model class for evaluation
        """
        with mlflow.start_run(run_name="ensemble", nested=True):
            # Calculate ensemble metrics
            ensemble_metrics = ModelClass.evaluate(
                y_test, ensemble_predictions, suffix="_test"
            )

            # Log ensemble metrics
            mlflow.log_metrics(ensemble_metrics)

            # Log ensemble predictions as artifacts
            output = self._create_output(ensemble_predictions)
            output.log_to_mlflow(basename="ensemble_prediction")

            logger.info(f"Ensemble test metrics: {ensemble_metrics}")

    def _log_experiment_params(
        self,
        model_class: str,
        init_params: dict,
        fit_params: dict,
        predict_params: dict,
        target_column: str,
        n_folds: int,
        ModelClass: Type[BaseRegressionModel],
    ):
        """
        Log experiment parameters to MLflow.

        Args:
            model_class: Full path to model class
            init_params: Model initialization parameters
            fit_params: Model fitting parameters
            predict_params: Model prediction parameters
            target_column: Name of target column
            n_folds: Number of folds
            ModelClass: Model class type
        """
        mlflow.log_params(
            {
                "model_class": model_class,
                "init_params": str(init_params),
                "fit_params": str(fit_params),
                "predict_params": str(predict_params),
                "target_column": target_column,
                "n_folds": n_folds,
                "model_name": ModelClass.model_name
                if hasattr(ModelClass, "model_name")
                else model_class,
            }
        )

    def _train_all_folds(
        self,
        train_val_pairs: list,
        test_data,
        ModelClass: Type[BaseRegressionModel],
        init_params: dict,
        fit_params: dict,
        predict_params: dict,
        target_column: str,
    ) -> Tuple[List[dict], List[dict], List[BaseRegressionModel], List]:
        """
        Train model on all folds.

        Args:
            train_val_pairs: List of (train, validation) dataframe pairs
            test_data: Test dataframe
            ModelClass: Model class to instantiate
            init_params: Model initialization parameters
            fit_params: Model fitting parameters
            predict_params: Model prediction parameters
            target_column: Name of target column

        Returns:
            Tuple of (validation_metrics_list, test_metrics_list, trained_models_list, test_predictions_list)
        """
        X_test, y_test = self._prepare_data(test_data, target_column)

        all_val_metrics = []
        all_test_metrics = []
        all_test_predictions = []
        models = []

        for fold_idx, (train_df, val_df) in enumerate(train_val_pairs, start=1):
            child_run_name = f"fold-{fold_idx:02d}"
            with mlflow.start_run(run_name=child_run_name, nested=True):
                logger.info(f"Training fold {fold_idx}/{len(train_val_pairs)}")

                # Initialize model for this fold
                model = ModelClass(**init_params)

                # Train and evaluate on validation
                val_metrics = self._train_single_fold(
                    model, train_df, val_df, target_column, fit_params, predict_params
                )

                # Evaluate on test and get predictions
                test_metrics, test_predictions = self._evaluate_on_test(
                    model, X_test, y_test, predict_params, return_predictions=True
                )

                # Log metrics for this fold
                self._log_fold_metrics(val_metrics, test_metrics)

                # log result
                config: ExperimentConfig = self.load("config")
                output = self._create_output(test_predictions)
                output.log_to_mlflow()

                # calculate feature importance and log it
                if params :=  config.model.feature_importance_params:
                    feature_importance = model.get_feature_importance(**params)
                else:
                    feature_importance = model.get_feature_importance()
                feature_importance.log_to_mlflow(log_html=True, log_png=True)

                # Store results
                all_val_metrics.append(val_metrics)
                all_test_metrics.append(test_metrics)
                all_test_predictions.append(test_predictions)
                model.save()
                model.log_to_mlflow()
                models.append(model)

        return all_val_metrics, all_test_metrics, models, all_test_predictions

    def _run(
        self,
        model_class: str,
        init_params: dict,
        fit_params: dict,
        predict_params: dict,
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Train regression model and evaluate on validation sets.
        """
        # Load data
        train_val_pairs = self.load("train_val_pairs")
        test_data = self.load("test")

        # Get model class
        ModelClass = self._get_model_class(model_class)

        # Train on all folds
        all_val_metrics, all_test_metrics, models, all_test_predictions = (
            self._train_all_folds(
                train_val_pairs,
                test_data,
                ModelClass,
                init_params,
                fit_params,
                predict_params,
                target_column,
            )
        )

        # Calculate ensemble predictions and log results
        if all_test_predictions:
            _, y_test = self._prepare_data(test_data, target_column)
            ensemble_predictions = self._calculate_ensemble_predictions(
                all_test_predictions
            )
            self._log_ensemble_results(ensemble_predictions, y_test, ModelClass)

        # Calculate and log average metrics
        avg_metrics = {}

        # Average validation metrics
        val_avg_metrics = self._calculate_average_metrics(all_val_metrics)
        avg_metrics.update(val_avg_metrics)
        if val_avg_metrics:
            mlflow.log_metrics(val_avg_metrics)

        # Average test metrics
        test_avg_metrics = self._calculate_average_metrics(all_test_metrics)
        avg_metrics.update(test_avg_metrics)
        if test_avg_metrics:
            mlflow.log_metrics(test_avg_metrics)

        # Log experiment parameters
        self._log_experiment_params(
            model_class,
            init_params,
            fit_params,
            predict_params,
            target_column,
            len(train_val_pairs),
            ModelClass,
        )

        logger.info(f"Average validation metrics: {avg_metrics}")

        return {
            "all_val_metrics": all_val_metrics,
            "all_test_metrics": all_test_metrics,
            "avg_metrics": avg_metrics,
            "n_folds": len(train_val_pairs),
        }

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Parameters for the Train task.
        """
        config: ExperimentConfig = self.load("config")
        return {
            "model_class": config.model.model_class,
            "init_params": config.model.init_params or {},
            "fit_params": config.model.fit_params or {},
            "predict_params": config.model.predict_params or {},
            "target_column": config.data.column.target,
        }

    @property
    def mlflow_run_name(self) -> str:
        """
        Run name for MLflow.
        """
        config_target = self._get_input_targets("config")

        if not config_target.exists():
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"model-{now}"

        config: ExperimentConfig = self.load("config")
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{config.model.name}-{now}"
