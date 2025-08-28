from typing import Literal, Optional

import plotly.graph_objects as go
import polars as pl


def plot_target_prediction_by_company(
    df: pl.DataFrame,
    *,
    date_col: str = "dtBaseDate",
    company_col: str = "AccountName",
    target_col: str = "Flow",
    pred_col: str = "prediction",
    init_company: Optional[str] = None,
    title: str = "Target vs Prediction (by Company)",
    display_mode: Literal["group", "overlay"] = "overlay",
    bar_opacity: float = 0.5,
    show_legend: bool = False,
) -> go.Figure:
    for c in [date_col, company_col, target_col, pred_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")

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
            opacity=bar_opacity,
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
            opacity=bar_opacity,
            hovertemplate=(
                f"{company_col}: {comp}<br>"
                f"{date_col}: %{{x}}<br>"
                "target: %{y}<extra></extra>"
            ),
        )

    buttons = []
    n_traces = 2 * len(companies)
    for i, comp in enumerate(companies):
        vis = [False] * n_traces
        vis[2 * i] = True
        vis[2 * i + 1] = True
        buttons.append(
            dict(
                label=str(comp),
                method="update",
                args=[{"visible": vis}, {"title": f"{title} — {comp}"}],
            )
        )

    fig.update_layout(
        title=f"{title} — {init_company}",
        barmode=("overlay" if display_mode == "overlay" else "group"),
        xaxis_title=date_col,
        yaxis_title="value",
        hovermode="x unified",
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
        showlegend=show_legend,
    )
    return fig
