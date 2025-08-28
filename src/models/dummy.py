import random

import polars as pl

from src.models.base import BaseRegressionModel


class DummyRegressor(BaseRegressionModel):
    model_name: str = "DummyRegressor"

    def __init__(self, **init_params):
        self.random = random.randint(0, 100)

    def fit(self, X: pl.DataFrame, y: pl.Series, **fit_params):
        return

    def predict(self, X, **predict_params):
        size = X.shape[0]
        return pl.DataFrame({"prediction": [self.random] * size})

    def save(self):
        return
