from pathlib import Path
from pprint import pprint as print

import polars as pl

from src.configs import ExperimentConfig

# from src.models.ag import AutoGluonModel
from src.models.lgbm import LightGBMModel
from src.utils.preprocess import (
    coerce_booleans,
    convert_bool_to_numeric,
    convert_date_to_numeric,
    remove_bool_columns,
    remove_date_columns,
)

config = ExperimentConfig.from_yaml("configs/experiments/dummy.yml")


def dtype_count(df: pl.DataFrame):
    dtype_counter = {}
    for dtype in df.dtypes:
        dtype_str = str(dtype)
        if dtype_str in dtype_counter:
            dtype_counter[dtype_str] += 1
        else:
            dtype_counter[dtype_str] = 1
    return dtype_counter


df = pl.read_csv(config.data.train_path, try_parse_dates=True)
# preprocess
if config.data.column.ignore_columns:
    df = df.drop(config.data.column.ignore_columns)
print(dtype_count(df))

df = coerce_booleans(df, config.data.column.bool_columns)
df = convert_bool_to_numeric(df)
df = remove_bool_columns(df)
df = convert_date_to_numeric(df)
df = remove_date_columns(df)


X, y = df.drop(config.data.column.target), df[config.data.column.target]
model = LightGBMModel()
model.fit(X, y)
feature_importance = model.get_feature_importance()
feature_importance.save_files(out_dir=Path("."))

# model = AutoGluonModel(label="Flow")
# # model = AutoGluonModel(label="Flow", problem_type="regression", eval_metric="rmse")
# model.fit(X, y, X, y, time_limit=30)
# features = model.get_feature_importance()
# print(features)
# y = model.predict(X)
# print(type(y))
# print(y)

# X_cat = X[cat_cols].fillna("missing")
# encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# encoder.fit(X_cat)
# encoded_array = encoder.transform(X_cat)
# encoded_df = pd.DataFrame(
#     encoded_array,
#     columns=encoder.get_feature_names_out(cat_cols),
#     index=X.index,
# )

# X = pd.concat([X.drop(cat_cols, axis=1), encoded_df], axis=1)
# print(dtype_count(X))

# Optuna hyperparameter tuning for CatBoostRegressor


# def objective(trial):
#     params = {
#         "iterations": trial.suggest_int("iterations", 100, 1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
#         "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "verbose": False,
#         "random_seed": 42,
#     }

#     model = CatBoostRegressor(**params)

#     # 5-fold cross-validation
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []

#     for train_idx, val_idx in kf.split(X):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
#         y_pred = model.predict(X_val)
#         score = np.sqrt(mean_squared_error(y_val, y_pred))
#         scores.append(score)

#     return np.mean(scores)


# # Optuna study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# # Best parameters
# print("\nBest parameters found:")
# print(study.best_params)
# print(f"\nBest RMSE: {study.best_value:.4f}")

# # Train final model with best parameters
# best_params = study.best_params
# best_params["verbose"] = False
# best_params["random_seed"] = 42

# final_model = CatBoostRegressor(**best_params)
# final_model.fit(X, y)

# print("\nFinal model trained with best parameters!")
