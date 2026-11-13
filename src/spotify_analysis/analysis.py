"""Calculations on cleaned streaming history (see `load.clean_streaming_history`)."""

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

Metric = Literal["plays", "hours"]
Level = Literal["song", "artist", "album", "show", "episode", "audiobook", "chapter"]
Kind = Literal["music", "podcast", "audiobook", "unknown"]

# Columns that identify a song, artist, album, podcast show or episode, or an
# audiobook or its chapter.
# Not the Spotify URI: the same song often has several (album version, single,
# remaster).
LEVEL_COLUMNS: dict[Level, list[str]] = {
    "song": ["artist", "track"],
    "artist": ["artist"],
    "album": ["artist", "album"],
    "show": ["show"],
    "episode": ["show", "episode"],
    "audiobook": ["audiobook"],
    "chapter": ["audiobook", "chapter"],
}

# A play counts only from MIN_PLAY_MS milliseconds on. Spotify counts streams from 30 seconds.
MIN_PLAY_MS = 0  # 30_000 for Spotifys criterion

MS_PER_HOUR = 3_600_000

# Songs and artists need at least this many plays to get a skip rate.
MIN_PLAYS_FOR_SKIP_RATE = 20

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
        "songs": len(counted[LEVEL_COLUMNS["song"]].dropna().drop_duplicates()),
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
    """Average listening per weekday (rows, Monday first) and hour (columns 0-23).

    Averaged over all days from the first to the last play, including days without
    listening: e.g. the plays on a typical Tuesday between 16:00 and 16:59.
    """
    weekday = df["local_ts"].dt.day_name().rename("weekday")
    hour = df["local_ts"].dt.hour.rename("hour")
    totals = _total_by(df, [weekday, hour], metric).unstack(fill_value=0)
    totals = totals.reindex(index=WEEKDAYS, columns=range(24), fill_value=0)
    if df.empty:
        return totals
    days = pd.date_range(
        df["local_ts"].min().normalize(), df["local_ts"].max().normalize()
    )
    days_per_weekday = days.day_name().value_counts().reindex(WEEKDAYS)
    # A weekday that never occurs (period shorter than a week) gets NaN, not 0.
    return totals.div(days_per_weekday, axis=0)


def top(df: pd.DataFrame, by: Level, metric: Metric, n: int = 10) -> pd.DataFrame:
    """The `n` most listened songs, artists, albums, shows, episodes or audiobooks.

    Returns one row each, with the identifying columns (e.g. artist and track)
    and the metric. Rows without these columns are left out, e.g. podcasts when
    ranking artists.
    """
    keys = [df[column] for column in LEVEL_COLUMNS[by]]
    return _total_by(df, keys, metric).nlargest(n).rename(metric).reset_index()


def most_skipped(
    df: pd.DataFrame,
    by: Level,
    n: int = 10,
    min_plays: int = MIN_PLAYS_FOR_SKIP_RATE,
) -> pd.DataFrame:
    """The `n` songs, artists, etc. with the highest skip rate.

    Only those with at least `min_plays` plays are considered, so a song skipped
    once out of two plays doesn't top the list.
    """
    counted = df[df["ms_played"] >= MIN_PLAY_MS]
    stats = counted.groupby(LEVEL_COLUMNS[by])["skipped"].agg(
        plays="size", skip_rate="mean"
    )
    stats = stats[stats["plays"] >= min_plays]
    return (
        stats.sort_values(["skip_rate", "plays"], ascending=False).head(n).reset_index()
    )


def in_period(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Only the plays from `start` to `end` (both days included, local time)."""
    after_start = df["local_ts"] >= pd.Timestamp(start)
    before_end = df["local_ts"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[after_start & before_end]


def of_kind(df: pd.DataFrame, kind: Kind) -> pd.DataFrame:
    """Only the plays of one kind, e.g. podcasts. Empty if there are none."""
    return df[df["kind"] == kind]


def select(df: pd.DataFrame, by: Level, key: str | tuple[str, ...]) -> pd.DataFrame:
    """The plays of one song, artist, album, etc., e.g. for a detail view.

    Args:
        df: Cleaned streaming history.
        by: What `key` identifies.
        key: Values of the identifying columns: an artist name, or e.g.
            ``("Daft Punk", "Discovery")`` for an album.
    """
    key = (key,) if isinstance(key, str) else key
    mask = (df[LEVEL_COLUMNS[by]] == list(key)).all(axis=1)
    return df[mask]


def over_time(
    df: pd.DataFrame,
    metric: Metric,
    freq: Literal["day", "month", "year"] = "month",
    between: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.Series:
    """Listening per day, month or year, including periods without listening (0).

    Args:
        df: Cleaned streaming history.
        metric: Plays or hours.
        freq: The length of the periods.
        between: Optional first and last time to cover, e.g. the selected period;
            by default from the first to the last play in `df`.
    """
    code = {"day": "D", "month": "M", "year": "Y"}[freq]
    totals = _total_by(df, df["local_ts"].dt.to_period(code).rename(freq), metric)
    if between is not None:
        first, last = (pd.Period(time, code) for time in between)
    elif totals.empty:
        return totals
    else:
        first, last = totals.index.min(), totals.index.max()
    all_periods = pd.period_range(first, last, freq=code)
    return totals.reindex(all_periods, fill_value=0).to_timestamp().rename(metric)


def top_per_year(
    df: pd.DataFrame, by: Level, metric: Metric, n: int = 5
) -> pd.DataFrame:
    """The `n` most listened songs, artists, etc. of each year ("Wrapped").

    Returns one row per year and rank, with the identifying columns and the metric.
    """
    ranks = _ranks_per_year(df, by, metric)
    return ranks[ranks["rank"] <= n].reset_index(drop=True)


def all_time_ranks(
    df: pd.DataFrame,
    by: Level,
    metric: Metric,
    n: int = 10,
    freq: Literal["year", "month"] = "year",
) -> pd.DataFrame:
    """The all-time top `n` songs, artists, etc. as of the end of each year or month.

    Like the standings in a "bar chart race": each period ranks everything by the
    total listened from the start of the history up to the end of that period.

    Returns one row per period and rank (1 = most listened so far), with the
    period (column "year": the year; "month": the month's first day), the
    identifying columns and the total so far.
    """
    columns = LEVEL_COLUMNS[by]
    plays = df.dropna(subset=columns)
    if plays.empty:
        return pd.DataFrame(columns=[freq, *columns, metric, "rank"])
    # A number per song, artist, ...: much faster to reshape than their names.
    entity = plays.groupby(columns, sort=False).ngroup().rename("entity")
    if freq == "year":
        period = plays["local_ts"].dt.year.rename(freq)
        periods = range(period.min(), period.max() + 1)
    else:
        period = plays["local_ts"].dt.to_period("M").rename(freq)
        periods = pd.period_range(period.min(), period.max(), freq="M")
    # Periods as rows, entities as columns; then the running total per entity.
    per_period = _total_by(plays, [period, entity], metric).unstack(fill_value=0)
    so_far = per_period.reindex(periods, fill_value=0).cumsum()

    # The top n of each row, picked with numpy: much faster than ranking all.
    values = so_far.to_numpy()
    top = np.argsort(-values, axis=1, kind="stable")[:, :n]
    top_n = pd.DataFrame(
        {
            freq: np.repeat(so_far.index, top.shape[1]),
            "entity": so_far.columns.to_numpy()[top].ravel(),
            metric: np.take_along_axis(values, top, axis=1).ravel(),
            "rank": np.tile(np.arange(1, top.shape[1] + 1), len(so_far)),
        }
    )
    top_n = top_n[top_n[metric] > 0]  # nothing listened yet: no rank
    if freq == "month":
        top_n[freq] = top_n[freq].dt.to_timestamp()
    names = plays[columns].groupby(entity).first()
    return top_n.join(names, on="entity")[[freq, *columns, metric, "rank"]].reset_index(
        drop=True
    )


def new_per_year(df: pd.DataFrame, by: Level = "artist") -> pd.Series:
    """How many songs, artists, etc. were played for the first time in each year.

    Note that in the first year of the history, everything is new.
    """
    counted = df[df["ms_played"] >= MIN_PLAY_MS]
    first_play = counted.groupby(LEVEL_COLUMNS[by])["local_ts"].min()
    return first_play.dt.year.value_counts().sort_index().rename_axis("year")


def top_days(df: pd.DataFrame, metric: Metric, n: int = 10) -> pd.Series:
    """The `n` days with the most listening."""
    return over_time(df, metric, "day").nlargest(n)


def on_repeat(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The `n` biggest repeats: the most plays of one song on a single day.

    Returns one row each, with day, artist, track and plays.
    """
    counted = df[df["ms_played"] >= MIN_PLAY_MS]
    day = counted["local_ts"].dt.normalize().rename("day")
    song = [counted[column] for column in LEVEL_COLUMNS["song"]]
    return (
        counted.groupby([day, *song]).size().nlargest(n).rename("plays").reset_index()
    )


def _ranks_per_year(df: pd.DataFrame, by: Level, metric: Metric) -> pd.DataFrame:
    """The metric and rank (1 = most listened) of everything, per year."""
    year = df["local_ts"].dt.year.rename("year")
    keys = [year, *[df[column] for column in LEVEL_COLUMNS[by]]]
    ranks = _total_by(df, keys, metric).rename(metric).reset_index()
    ranks["rank"] = (
        ranks.groupby("year")[metric].rank(method="first", ascending=False).astype(int)
    )
    return ranks.sort_values(["year", "rank"], ignore_index=True)


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
