import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Type

import mlflow
import numpy as np
import optuna
import polars as pl
from gokart.parameter import TaskInstanceParameter
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.configs import ExperimentConfig
from src.models.base import BaseRegressionModel
from src.tasks import LoadConfigTask
from src.tasks._train import TrainTask
from src.tasks.preprocess import PreprocessPipeline

logger = logging.getLogger(__name__)


class OptunaTask(TrainTask):
    """
    Task for hyperparameter optimization using Optuna, inheriting from TrainTask.
    """

    load_config_task = TaskInstanceParameter(expected_type=LoadConfigTask)
    train_preprocess_pipeline = TaskInstanceParameter(expected_type=PreprocessPipeline)
    test_preprocess_pipeline = TaskInstanceParameter(expected_type=PreprocessPipeline)

    def _create_search_space(self, trial: optuna.Trial, search_space: dict) -> dict:
        """
        Create parameter dictionary from Optuna search space definition.
        
        Args:
            trial: Optuna trial object
            search_space: Search space configuration from YAML
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type")
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                    log=param_config.get("log", False)
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step"),
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                logger.warning(f"Unknown parameter type: {param_type} for {param_name}")
                
        return params

    def _objective(
        self,
        trial: optuna.Trial,
        train_val_pairs: list,
        test_data,
        ModelClass: Type[BaseRegressionModel],
        base_init_params: dict,
        fit_params: dict,
        predict_params: dict,
        target_column: str,
        search_space: dict,
        optimize_metric: str = "rmse_val"
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            train_val_pairs: List of (train, validation) dataframe pairs
            test_data: Test dataframe
            ModelClass: Model class to instantiate
            base_init_params: Base initialization parameters
            fit_params: Model fitting parameters
            predict_params: Model prediction parameters
            target_column: Name of target column
            search_space: Hyperparameter search space
            optimize_metric: Metric to optimize (minimize)
            
        Returns:
            Objective value to minimize
        """
        # Sample hyperparameters from search space
        trial_params = self._create_search_space(trial, search_space)
        
        # Merge with base parameters
        init_params = {**base_init_params, **trial_params}
        
        # Log trial parameters to MLflow
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_param("trial_number", trial.number)
            
            # Train on all folds
            all_val_metrics = []
            all_test_metrics = []
            
            for fold_idx, (train_df, val_df) in enumerate(train_val_pairs, start=1):
                logger.info(f"Trial {trial.number} - Fold {fold_idx}/{len(train_val_pairs)}")
                
                # Initialize model for this fold
                model = ModelClass(**init_params)
                
                # Train and evaluate on validation
                val_metrics = self._train_single_fold(
                    model, train_df, val_df, target_column, fit_params, predict_params
                )
                
                all_val_metrics.append(val_metrics)
                
                # Report intermediate value for pruning
                if fold_idx == 1 and optimize_metric in val_metrics:
                    trial.report(val_metrics[optimize_metric], fold_idx)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"Trial {trial.number} pruned at fold {fold_idx}")
                        raise optuna.TrialPruned()
            
            # Calculate average metrics
            avg_val_metrics = self._calculate_average_metrics(all_val_metrics)
            
            # Log average metrics
            if avg_val_metrics:
                mlflow.log_metrics(avg_val_metrics)
            
            # Get objective value
            objective_value = avg_val_metrics.get(optimize_metric, float("inf"))
            
            # If this is the best trial so far, evaluate on test set
            try:
                is_best = trial.study.best_value > objective_value
            except ValueError:
                # No completed trials yet, this is the first one
                is_best = True
            
            if is_best:
                X_test, y_test = self._prepare_data(test_data, target_column)
                
                # Train final model on all training data for test evaluation
                all_train_data = []
                for train_df, val_df in train_val_pairs:
                    all_train_data.append(pl.concat([train_df, val_df]))
                
                combined_train = pl.concat(all_train_data).unique()
                X_train_all, y_train_all = self._prepare_data(combined_train, target_column)
                
                # Train on all data
                final_model = ModelClass(**init_params)
                if final_model.val_is_required:
                    # Use last fold's validation as validation set
                    X_val, y_val = self._prepare_data(train_val_pairs[-1][1], target_column)
                    final_model.fit(X_train_all, y_train_all, X_val, y_val, **fit_params)
                else:
                    final_model.fit(X_train_all, y_train_all, **fit_params)
                
                # Evaluate on test
                test_metrics = self._evaluate_on_test(
                    final_model, X_test, y_test, predict_params
                )
                mlflow.log_metrics(test_metrics)
            
            return objective_value

    def _run_optimization(
        self,
        train_val_pairs: list,
        test_data,
        ModelClass: Type[BaseRegressionModel],
        base_init_params: dict,
        fit_params: dict,
        predict_params: dict,
        target_column: str,
        optuna_config
    ) -> Dict[str, Any]:
        """
        Run Optuna optimization process.
        
        Args:
            train_val_pairs: List of (train, validation) dataframe pairs
            test_data: Test dataframe
            ModelClass: Model class to instantiate
            base_init_params: Base initialization parameters
            fit_params: Model fitting parameters
            predict_params: Model prediction parameters
            target_column: Name of target column
            optuna_config: Optuna configuration object
            
        Returns:
            Dictionary containing optimization results
        """
        # Extract optimization settings from Pydantic model
        search_space = optuna_config.search_space
        n_trials = optuna_config.n_trials
        optimize_metric = optuna_config.optimize_metric
        use_pruning = optuna_config.use_pruning
        n_startup_trials = optuna_config.n_startup_trials
        n_warmup_steps = optuna_config.n_warmup_steps
        
        # Create study
        sampler = TPESampler(seed=42, n_startup_trials=n_startup_trials)
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps) if use_pruning else None
        
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"optuna-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Define objective function with fixed parameters
        def objective(trial):
            return self._objective(
                trial,
                train_val_pairs,
                test_data,
                ModelClass,
                base_init_params,
                fit_params,
                predict_params,
                target_column,
                search_space,
                optimize_metric
            )
        
        # Run optimization
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)
        
        # Log best parameters
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value ({optimize_metric}): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Train final model with best parameters
        best_params = {**base_init_params, **study.best_params}
        
        with mlflow.start_run(run_name="best-model", nested=True):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_trial_number", study.best_trial.number)
            mlflow.log_metric(f"best_{optimize_metric}", study.best_value)
            
            # Train best model on all folds for final evaluation
            all_val_metrics, all_test_metrics, models, all_test_predictions = (
                self._train_all_folds(
                    train_val_pairs,
                    test_data,
                    ModelClass,
                    best_params,
                    fit_params,
                    predict_params,
                    target_column,
                )
            )
            
            # Calculate ensemble predictions
            if all_test_predictions:
                _, y_test = self._prepare_data(test_data, target_column)
                ensemble_predictions = self._calculate_ensemble_predictions(
                    all_test_predictions
                )
                self._log_ensemble_results(ensemble_predictions, y_test, ModelClass)
            
            # Log average metrics
            avg_metrics = {}
            val_avg_metrics = self._calculate_average_metrics(all_val_metrics)
            avg_metrics.update(val_avg_metrics)
            if val_avg_metrics:
                mlflow.log_metrics(val_avg_metrics)
            
            test_avg_metrics = self._calculate_average_metrics(all_test_metrics)
            avg_metrics.update(test_avg_metrics)
            if test_avg_metrics:
                mlflow.log_metrics(test_avg_metrics)
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "all_val_metrics": all_val_metrics,
            "all_test_metrics": all_test_metrics,
            "avg_metrics": avg_metrics,
        }

    def _run(
        self,
        model_class: str,
        init_params: dict,
        fit_params: dict,
        predict_params: dict,
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization or regular training based on configuration.
        """
        # Load configuration
        config: ExperimentConfig = self.load("config")
        
        # Check if Optuna configuration exists
        optuna_config = getattr(config.model, "optuna_config", None)
        
        if optuna_config and optuna_config.enabled:
            logger.info("Running Optuna hyperparameter optimization")
            
            # Load data
            train_val_pairs = self.load("train_val_pairs")
            test_data = self.load("test")
            
            # Get model class
            ModelClass = self._get_model_class(model_class)
            
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
            
            # Run optimization
            return self._run_optimization(
                train_val_pairs,
                test_data,
                ModelClass,
                init_params,
                fit_params,
                predict_params,
                target_column,
                optuna_config
            )
        else:
            logger.info("Optuna not enabled, running regular training")
            # Fall back to regular training
            return super()._run(
                model_class,
                init_params,
                fit_params,
                predict_params,
                target_column
            )

    @property
    def mlflow_run_name(self) -> str:
        """
        Run name for MLflow.
        """
        config_target = self._get_input_targets("config")

        if not config_target.exists():
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"optuna-{now}"

        config: ExperimentConfig = self.load("config")
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"optuna-{config.model.name}-{now}"