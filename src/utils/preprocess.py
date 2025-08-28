import logging
from typing import Collection

import polars as pl

logger = logging.getLogger(__name__)

TRUTHY = {"true", "t", "1", "y", "yes", "真", "True", "TRUE", "１", "はい"}
FALSY = {"false", "f", "0", "n", "no", "偽", "False", "FALSE", "０", "いいえ"}
NULLISH = {"", "na", "n/a", "none", "null", "nan", "-", "ー", "—", "不明", "未", "なし"}
TRUTHY_LOWER = frozenset(str(x).lower() for x in TRUTHY)
FALSY_LOWER = frozenset(str(x).lower() for x in FALSY)


def coerce_booleans(
    df: pl.DataFrame,
    bool_columns: list[str],
    *,
    truthy_tokens: Collection[str] = TRUTHY_LOWER,
    falsy_tokens: Collection[str] = FALSY_LOWER,
    nullish_tokens: Collection[str] = NULLISH,
) -> pl.DataFrame:
    """
    Convert Utf8 boolean representation columns to Boolean (supports Japanese 真/偽)
    """
    truthy_list = list(truthy_tokens)
    falsy_list = list(falsy_tokens)
    nullish_list = list(nullish_tokens)

    exprs: list[pl.Expr] = []

    for col in bool_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            continue

        if df.schema[col] == pl.Boolean:
            continue

        base = pl.col(col)

        norm = base.cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()

        is_truthy = norm.is_in(truthy_list)
        is_falsy = norm.is_in(falsy_list)
        is_nullish = (
            base.is_null() | norm.is_in(nullish_list) | (norm.str.len_chars() == 0)
        )

        exprs.append(
            pl.when(is_truthy)
            .then(pl.lit(True))
            .when(is_falsy)
            .then(pl.lit(False))
            .when(is_nullish)
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias(col)
        )

    return df.with_columns(exprs) if exprs else df


def impute_booleans(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Impute missing boolean values with False and create derived columns for integer conversion.
    """
    new_cols = []

    for c, dtype in df.schema.items():
        if dtype == pl.Boolean:
            new_cols.append(pl.col(c).fill_null(False).cast(pl.Int8).alias(f"{c}_int"))
    if new_cols:
        df = df.with_columns(new_cols)
    return df


def convert_bool_to_numeric(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert boolean columns to numeric format.
    """
    for col, dtype in df.schema.items():
        if dtype == pl.Boolean:
            df = df.with_columns(pl.col(col).cast(pl.Int8).alias(f"{col}__i"))
    return df


def remove_bool_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove boolean columns from the DataFrame.
    """
    bool_columns = [col for col, dtype in df.schema.items() if dtype == pl.Boolean]
    return df.drop(bool_columns)


def convert_date_to_numeric(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert date columns to numeric format.
    """
    new_cols = []

    for col, dtype in df.schema.items():
        if dtype == pl.Date:
            new_cols.append(pl.col(col).cast(pl.Int32).alias(f"{col}__i"))
        elif dtype == pl.Datetime:
            new_cols.append(pl.col(col).cast(pl.Int64).alias(f"{col}__i"))

    if new_cols:
        df = df.with_columns(new_cols)

    return df


def remove_date_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove date columns from the DataFrame.
    """
    date_columns = [
        col
        for col, dtype in df.schema.items()
        if dtype == pl.Date or dtype == pl.Datetime
    ]
    return df.drop(date_columns)
