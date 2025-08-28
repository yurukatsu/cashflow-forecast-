import mlflow
import pandas as pd
import polars as pl
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.models.base import BaseRegressionModel, FeatureImportance


class XGBoostModel(BaseRegressionModel):
    model_name: str = "XGBRegressor"

    def __init__(self, **init_params):
        if init_params:
            self.model = XGBRegressor(**init_params)
        else:
            self.model = XGBRegressor()
        self.cat_cols = None
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def _prepare_data(
        self, X: pl.DataFrame, y: pl.DataFrame | pl.Series = None, is_train: bool = True
    ):
        """Prepare data for training or prediction."""
        if is_train:
            self.cat_cols = [
                c for c, dtype in X.schema.items() if dtype in [pl.Utf8, pl.Categorical]
            ]
        X: pd.DataFrame = X.to_pandas()
        X_cat = X[self.cat_cols].fillna("missing")

        if is_train:
            self.encoder.fit(X_cat)

        encoded_array = self.encoder.transform(X_cat)
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=X.index,
        )

        X = pd.concat([X.drop(self.cat_cols, axis=1), encoded_df], axis=1)
        return X, y.to_pandas().squeeze() if y is not None else None

    def fit(
        self, X_train: pl.DataFrame, y_train: pl.DataFrame | pl.Series, **fit_params
    ):
        X_train, y_train = self._prepare_data(X_train, y_train, is_train=True)

        if fit_params:
            self.model.fit(X_train, y_train, **fit_params)
            return
        self.model.fit(X_train, y_train)

    def predict(self, X: pl.DataFrame, **predict_params):
        X, _ = self._prepare_data(X, is_train=False)

        if predict_params:
            predictions = self.model.predict(X, **predict_params)
        else:
            predictions = self.model.predict(X)

        return pl.DataFrame({"prediction": predictions})

    def get_feature_importance(self):
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        df = pl.DataFrame({"feature": feature_names, "importance": importance}).sort(
            "importance", descending=True
        )
        return FeatureImportance(df)

    def save(self):
        mlflow.xgboost.log_model(self.model)
