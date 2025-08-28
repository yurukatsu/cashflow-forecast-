import pandas as pd
import polars as pl
from autogluon.tabular import TabularPredictor

from src.models.base import BaseRegressionModel, FeatureImportance


class AutoGluonModel(BaseRegressionModel):
    model_name: str = "AutogluonTabularPredictor"
    val_is_required: bool = True

    def __init__(self, **init_params):
        if "label" not in init_params:
            raise ValueError("The 'label' parameter must be set in init_params.")
        label = init_params["label"]
        init_params.pop("label", None)
        self.model = TabularPredictor(label, **init_params)

    def _prepare_data(
        self, X: pl.DataFrame, y: pl.DataFrame | pl.Series = None
    ) -> pd.DataFrame:
        if y is not None:
            if isinstance(y, pl.Series):
                y_df = y.to_frame(self.model.label)
            else:
                y_df = y
            return pl.concat([X, y_df], how="horizontal").to_pandas()
        return X.to_pandas()

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
        **fit_params,
    ):
        train_data = self._prepare_data(X_train, y_train)
        tuning_data = self._prepare_data(X_val, y_val)
        self.model.fit(train_data, tuning_data=tuning_data, **fit_params)

    def predict(self, X: pl.DataFrame, **predict_params):
        data = self._prepare_data(X)
        predictions = self.model.predict(data, **predict_params)
        return pl.DataFrame({"prediction": predictions})

    def get_feature_importance(
        self,
        **feature_importance_params
    ):
        importance = self.model.feature_importance(
            **feature_importance_params
        )
        df = pl.DataFrame(importance.reset_index(names="feature"))
        return FeatureImportance(df)

    def save(self):
        pass
