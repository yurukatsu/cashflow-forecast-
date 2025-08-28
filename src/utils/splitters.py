from __future__ import annotations

import datetime
from typing import Generator, List, Tuple

import polars as pl
from dateutil.relativedelta import relativedelta

from src.configs import ValidationMethodConfig

Interval = Tuple[datetime.datetime, datetime.datetime]


def parse_duration(duration: str | None) -> relativedelta:
    """
    Parse a duration string into a relativedelta object.

    :param duration: The duration string to parse.
    :type duration: str
    :return: A relativedelta object representing the parsed duration.
    :rtype: relativedelta
    """

    if duration.endswith("m"):
        unit = int(duration[:-1])
        return relativedelta(months=unit)
    if duration.endswith("mo"):
        unit = int(duration[:-2])
        return relativedelta(months=unit)
    if duration.endswith(" month"):
        unit = int(duration[:-5])
        return relativedelta(months=unit)
    if duration.endswith(" months"):
        unit = int(duration[:-6])
        return relativedelta(months=unit)

    if duration.endswith("w"):
        unit = int(duration[:-1])
        return relativedelta(weeks=unit)
    if duration.endswith(" week"):
        unit = int(duration[:-5])
        return relativedelta(weeks=unit)
    if duration.endswith(" weeks"):
        unit = int(duration[:-6])
        return relativedelta(weeks=unit)

    if duration.endswith("d"):
        return relativedelta(days=int(duration[:-1]))
    if duration.endswith(" day"):
        return relativedelta(days=int(duration[:-4]))
    if duration.endswith(" days"):
        return relativedelta(days=int(duration[:-5]))

    if duration == "":
        return relativedelta()

    raise ValueError(f"Unknown duration unit: {duration}")


class TimeSeriesSplitter:
    """
    Split time series data based on the specified configuration.
    """

    date_format = "%Y-%m-%d"

    def __init__(self, config: ValidationMethodConfig):
        self.strategy = config.strategy
        self.n_splits = config.n_splits
        self.validation_duration = parse_duration(config.validation_duration)
        self.gap_duration = parse_duration(config.gap_duration or "")
        self.step_duration = parse_duration(config.step_duration)
        self.train_duration = (
            parse_duration(config.train_duration) if config.train_duration else None
        )
        self.train_start_date = (
            datetime.datetime.strptime(config.train_start_date, self.date_format)
            if config.train_start_date
            else None
        )
        self.test_end_date = (
            datetime.datetime.strptime(config.test_end_date, self.date_format)
            if config.test_end_date
            else None
        )

    def sliding_window_split(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> List[Tuple[Interval, Interval]]:
        """
        Split the data into training and validation sets based on the sliding window strategy.
        """
        if self.train_start_date and start_date < self.train_start_date:
            start_date = self.train_start_date

        if self.test_end_date and end_date > self.test_end_date:
            end_date = self.test_end_date

        splits = []
        for i in range(self.n_splits):
            current_offset = self.step_duration * i

            validation_end = end_date - current_offset
            validation_start = (
                validation_end - self.validation_duration + relativedelta(days=1)
            )
            train_end = validation_start - self.gap_duration - relativedelta(days=1)

            if self.train_duration:
                train_start = train_end - self.train_duration + relativedelta(days=1)
            else:
                train_start = start_date + (self.n_splits - 1 - i) * self.step_duration

            splits.append(
                ((train_start, train_end), (validation_start, validation_end))
            )

        return sorted(splits, key=lambda x: x[0][0])

    def expanding_window_split(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> List[Tuple[Interval, Interval]]:
        """
        Split the data into training and validation sets based on the expanding window strategy.
        """
        if start_date < self.train_start_date:
            start_date = self.train_start_date

        if end_date > self.test_end_date:
            end_date = self.test_end_date

        splits = []
        for i in range(self.n_splits):
            current_offset = self.step_duration * i

            validation_end = end_date - current_offset
            validation_start = (
                validation_end - self.validation_duration + relativedelta(days=1)
            )
            train_end = validation_start - self.gap_duration - relativedelta(days=1)
            train_start = start_date

            splits.append(
                ((train_start, train_end), (validation_start, validation_end))
            )

        return sorted(splits, key=lambda x: x[0][1])

    def split(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> List[Tuple[Interval, Interval]]:
        """
        Split the data into training and validation sets based on the specified strategy.
        """
        if self.strategy == "sliding_window":
            return self.sliding_window_split(start_date, end_date)
        elif self.strategy == "expanding_window":
            return self.expanding_window_split(start_date, end_date)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class TimeSeriesDataSplitter:
    def __init__(self, config: ValidationMethodConfig, date_column_name: str):
        self.splitter = TimeSeriesSplitter(config=config)
        self.date_column_name = date_column_name

    def split(
        self, df: pl.DataFrame
    ) -> Generator[Tuple[pl.DataFrame, pl.DataFrame], None, None]:
        """
        Split the DataFrame into training and validation sets based on the specified date column.
        """
        start_date = df[self.date_column_name].min()
        if isinstance(start_date, datetime.date):
            start_date = datetime.datetime.combine(
                start_date, datetime.datetime.min.time()
            )

        end_date = df[self.date_column_name].max()
        if isinstance(end_date, datetime.date):
            end_date = datetime.datetime.combine(end_date, datetime.datetime.max.time())

        print(start_date, end_date)

        for train_interval, val_interval in self.splitter.split(
            start_date=start_date,
            end_date=end_date,
        ):
            train_data = df.filter(
                pl.col(self.date_column_name).is_between(
                    train_interval[0], train_interval[1]
                )
            )
            val_data = df.filter(
                pl.col(self.date_column_name).is_between(
                    val_interval[0], val_interval[1]
                )
            )
            yield train_data, val_data
