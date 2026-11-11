"""Calculations on cleaned streaming history (see `load.clean_streaming_history`)."""

from typing import Literal

import pandas as pd

Metric = Literal["plays", "hours"]

# A play counts only from MIN_PLAY_MS milliseconds on. Spotify counts streams from 30 seconds.
MIN_PLAY_MS = 0  # 30_000 for Spotifys criterion

MS_PER_HOUR = 3_600_000

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def key_figures(df: pd.DataFrame) -> dict:
    """Headline numbers: hours, plays, distinct songs, artists, podcast shows and
    audiobooks, and first and last play."""
    counted = df[df["ms_played"] >= MIN_PLAY_MS]
    return {
        "hours": df["ms_played"].sum() / MS_PER_HOUR,
        "plays": len(counted),
        "songs": counted["track_uri"].nunique(),
        "artists": counted["artist"].nunique(),
        "shows": counted["show"].nunique(),
        "audiobooks": counted["audiobook_uri"].nunique(),
        "first_play": df["local_ts"].min(),
        "last_play": df["local_ts"].max(),
    }


def shares(df: pd.DataFrame, column: str, metric: Metric) -> pd.Series:
    """Share of each value in `column` (e.g. "kind" or "skipped"), summing to 1."""
    totals = _total_by(df, df[column], metric)
    return (totals / totals.sum()).sort_values(ascending=False)


def by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Plays and hours per country, most listened first."""
    table = pd.DataFrame(
        {
            "plays": _total_by(df, df["country"], "plays"),
            "hours": _total_by(df, df["country"], "hours"),
        }
    )
    return table.sort_values("hours", ascending=False)


def os_by_year(df: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """Listening per year (rows) and operating system (columns)."""
    year = df["local_ts"].dt.year.rename("year")
    return _total_by(df, [year, df["os"]], metric).unstack(fill_value=0)


def by_weekday_and_hour(df: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """Listening per weekday (rows, Monday first) and hour of day (columns 0-23)."""
    weekday = df["local_ts"].dt.day_name().rename("weekday")
    hour = df["local_ts"].dt.hour.rename("hour")
    table = _total_by(df, [weekday, hour], metric).unstack(fill_value=0)
    return table.reindex(index=WEEKDAYS, columns=range(24), fill_value=0)


def _total_by(
    df: pd.DataFrame, keys: pd.Series | list[pd.Series], metric: Metric
) -> pd.Series:
    """Sum the metric per group, e.g. per country or per (year, os)."""
    return _metric_values(df, metric).groupby(keys).sum()


def _metric_values(df: pd.DataFrame, metric: Metric) -> pd.Series:
    """Each row's contribution to the metric: 1 or 0 per play, or its hours."""
    if metric == "plays":
        return (df["ms_played"] >= MIN_PLAY_MS).astype(int)
    if metric == "hours":
        return df["ms_played"] / MS_PER_HOUR
    raise ValueError(f"Unknown metric {metric!r}, expected 'plays' or 'hours'.")
