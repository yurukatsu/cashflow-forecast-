# import polars as pl

# train = pl.read_csv("./data/raw/train.csv", try_parse_dates=True)
# test = pl.read_csv("./data/raw/test.csv", try_parse_dates=True)
# baseline = pl.read_csv("./data/raw/baseline.csv", try_parse_dates=True)

# train = train.sort(["dtBaseDate", "AccountCode"])
# test = test.sort(["dtBaseDate", "AccountCode"])


# def count_dtypes(df: pl.DataFrame):
#     tmp = {}
#     for dtype in df.dtypes:
#         if dtype in tmp:
#             tmp[dtype] += 1
#         else:
#             tmp[dtype] = 1
#     print(tmp)


# def check_dtype_date(df: pl.DataFrame):
#     for column in df.columns:
#         if df[column].dtype == pl.Date:
#             print(f"{column} is Date")
#         elif df[column].dtype == pl.Datetime:
#             print(f"{column} is Datetime")


# print(train)
# print(test)

from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl


def plot_target_prediction_by_company(
    df: pl.DataFrame,
    *,
    date_col: str = "date",
    company_col: str = "company_name",
    target_col: str = "target",
    pred_col: str = "prediction",
    init_company: Optional[str] = None,
    title: str = "Target vs Prediction (by Company)",
    display_mode: Literal["group", "overlay"] = "overlay",  # 👈 ここで重ね表示を指定
    bar_opacity: float = 0.5,  # 👈 半透明度
    show_legend: bool = False,  # ドロップダウンと干渉するなら False
) -> "go.Figure":
    # 必須列チェック
    for c in [date_col, company_col, target_col, pred_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")

    # 日付を可能ならパース
    if df.schema[date_col] not in (pl.Datetime, pl.Date):
        df = df.with_columns(
            pl.col(date_col)
            .str.strptime(pl.Datetime, strict=False, exact=False)
            .fill_null(
                pl.col(date_col).str.strptime(pl.Date, strict=False, exact=False)
            )
        )

    df = df.sort([company_col, date_col])

    companies = (
        df.select(pl.col(company_col).unique()).to_series().drop_nulls().to_list()
    )
    if not companies:
        raise ValueError("No companies found in the given company column.")
    if init_company is None or init_company not in companies:
        init_company = companies[0]

    fig = go.Figure()

    # 会社ごとに2トレースずつ作る（overlay時はoffsetgroupを使わない）
    for comp in companies:
        sub = df.filter(pl.col(company_col) == comp)
        x_vals = sub[date_col].to_list()
        y_target = sub[target_col].to_list()
        y_pred = sub[pred_col].to_list()
        vis = comp == init_company

        fig.add_bar(
            x=x_vals,
            y=y_pred,
            name="prediction",
            visible=vis,
            opacity=bar_opacity,  # 👈 半透明
            hovertemplate=(
                f"{company_col}: {comp}<br>"
                f"{date_col}: %{{x}}<br>"
                "prediction: %{y}<extra></extra>"
            ),
        )
        fig.add_bar(
            x=x_vals,
            y=y_target,
            name="target",
            visible=vis,
            opacity=bar_opacity,  # 👈 半透明
            hovertemplate=(
                f"{company_col}: {comp}<br>"
                f"{date_col}: %{{x}}<br>"
                "target: %{y}<extra></extra>"
            ),
        )

    # ドロップダウン（会社単位で2トレースを同時ON）
    buttons = []
    n_traces = 2 * len(companies)
    for i, comp in enumerate(companies):
        vis = [False] * n_traces
        vis[2 * i] = True  # prediction
        vis[2 * i + 1] = True  # target
        buttons.append(
            dict(
                label=str(comp),
                method="update",
                args=[{"visible": vis}, {"title": f"{title} — {comp}"}],
            )
        )

    fig.update_layout(
        title=f"{title} — {init_company}",
        barmode=("overlay" if display_mode == "overlay" else "group"),  # 👈 ここ
        xaxis_title=date_col,
        yaxis_title="value",
        hovermode="x unified",  # 同じxで両方の値を1つのツールチップに
        updatemenus=[
            dict(
                type="dropdown",
                x=1.02,
                xanchor="left",
                y=1.0,
                yanchor="top",
                buttons=buttons,
                showactive=True,
            )
        ],
        showlegend=show_legend,  # 凡例はデフォルト非表示（重なり回避）
    )
    return fig


# ==== サンプルデータ生成 ====
np.random.seed(0)

dates = pd.date_range("2025-01-01", periods=30, freq="D")
companies = ["Company A", "Company B", "Company C"]

rows = []
for comp in companies:
    for d in dates:
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "company_id": f"{comp[:3]}_{d.strftime('%m%d')}",
                "company_name": comp,
                "target": np.random.randint(80, 120),
                "prediction": np.random.randint(70, 130),
            }
        )

df = pl.DataFrame(rows)

print(df)

# ==== グラフ描画 ====
fig = plot_target_prediction_by_company(
    df,
    date_col="date",
    company_col="company_name",
    target_col="target",
    pred_col="prediction",
    init_company="Company A",
)

fig.show()
