from __future__ import annotations

import tempfile
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import mlflow
import plotly.express as px
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


class BaseRegressionModel(ABC):
    model_name: str
    metrics = {
        "RMSE": root_mean_squared_error,
        "MAE": mean_absolute_error,
        "MAPE": mean_absolute_percentage_error,
        "R2": r2_score,
    }
    val_is_required: bool = False

    def __init__(self, **init_params):
        self.model = None

    def _prepare_data(self, X: pl.DataFrame, y: pl.Series = None):
        """Prepare data for training or prediction."""
        return X.to_pandas(), y.to_pandas().squeeze() if y is not None else None

    @abstractmethod
    def fit(
        self, X_train: pl.DataFrame, y_train: pl.DataFrame | pl.Series, **fit_params
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pl.DataFrame, **predict_params) -> pl.DataFrame:
        raise NotImplementedError

    @classmethod
    def evaluate(
        cls,
        y_true: pl.DataFrame,
        y_pred: pl.DataFrame,
        suffix: str = "",
        prefix: str = "",
    ):
        return {
            f"{prefix}{metric}{suffix}": func(y_true.to_pandas(), y_pred.to_pandas())
            for metric, func in cls.metrics.items()
        }

    def get_feature_importance(self) -> FeatureImportance:
        df = pl.DataFrame({"feature": [], "importance": []})
        return FeatureImportance(df)

    @abstractmethod
    def save(self):
        raise NotImplementedError

    def log_to_mlflow(
        self,
        artifact_subdir: Optional[str] = "models",
        basename: str = "model",
    ) -> dict:
        """
        MLflow に HTML と PNG を記録する。
        - HTML はインタラクティブ表示用（UIではリンク表示）
        - PNG は UI 上にサムネイル表示（MLflow 2.3+ で `log_image` 推奨）
        戻り値は {"html": bool, "png": bool}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{basename}.pkl"
            with out_path.open("wb") as f:
                pickle.dump(self.model, f)

            # モデルの保存
            mlflow.log_artifact(str(out_path), artifact_path=artifact_subdir)


class FeatureImportance:
    """
    polars DataFrame（columns: feature, importance）から
    Plotlyの特徴量重要度グラフを作り、MLflowに保存・記録するユーティリティ。
    """

    def __init__(self, df: pl.DataFrame):
        # 必要カラムと型をそろえる（欠損・文字列混在に強く）
        if "feature" not in df.columns or "importance" not in df.columns:
            raise ValueError("df には 'feature', 'importance' カラムが必要です。")
        self.df = (
            df.select(
                pl.col("feature").cast(pl.Utf8), pl.col("importance").cast(pl.Float64)
            ).drop_nulls(["feature"])  # feature が null は落とす
        )

    def to_figure(self):
        # Plotly は pandas 互換が必要なので変換
        pdf = self.df.to_pandas()

        # importance 降順・横棒・上位が上に来るように
        fig = px.bar(
            pdf.sort_values("importance", ascending=False),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig

    def save_files(
        self,
        out_dir: Path,
        basename: str = "feature_importance",
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
                # kaleido が必要（pip install -U kaleido）
                fig.write_image(str(png_path), width=width, height=height, scale=scale)
                paths["png"] = png_path
            except Exception as e:
                # kaleido 未導入などで失敗しても処理は継続
                print(
                    f"[WARN] PNG 書き出しに失敗: {e}\n"
                    f"       `pip install -U kaleido` を検討してください。"
                )
                paths["png"] = None

        return paths

    def save_csv(self, out_dir: Path, basename: str = "feature_importance") -> Path:
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
        artifact_subdir: Optional[str] = "figures",
        basename: str = "feature_importance",
        log_html: bool = True,
        log_png: bool = True,
        log_csv: bool = True,
        csv_subdir: Optional[str] = "tables",
    ) -> dict:
        """
        MLflow に HTML と PNG を記録する。
        - HTML はインタラクティブ表示用（UIではリンク表示）
        - PNG は UI 上にサムネイル表示（MLflow 2.3+ で `log_image` 推奨）
        戻り値は {"html": bool, "png": bool}
        """
        recorded = {"html": False, "png": False}
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
