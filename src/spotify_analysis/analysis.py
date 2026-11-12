"""Calculations on cleaned streaming history (see `load.clean_streaming_history`)."""

from typing import Literal

import pandas as pd

Metric = Literal["plays", "hours"]
Level = Literal["song", "artist", "album", "show", "episode", "audiobook"]
Kind = Literal["music", "podcast", "audiobook", "unknown"]

# Columns that identify a song, artist, album, podcast show, episode or audiobook.
# Not the Spotify URI: the same song often has several (album version, single,
# remaster).
LEVEL_COLUMNS: dict[Level, list[str]] = {
    "song": ["artist", "track"],
    "artist": ["artist"],
    "album": ["artist", "album"],
    "show": ["show"],
    "episode": ["show", "episode"],
    "audiobook": ["audiobook"],
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
    """Listening per weekday (rows, Monday first) and hour of day (columns 0-23)."""
    weekday = df["local_ts"].dt.day_name().rename("weekday")
    hour = df["local_ts"].dt.hour.rename("hour")
    table = _total_by(df, [weekday, hour], metric).unstack(fill_value=0)
    return table.reindex(index=WEEKDAYS, columns=range(24), fill_value=0)


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
    df: pd.DataFrame, metric: Metric, freq: Literal["day", "month", "year"] = "month"
) -> pd.Series:
    """Listening per day, month or year, including periods without listening (0)."""
    code = {"day": "D", "month": "M", "year": "Y"}[freq]
    totals = _total_by(df, df["local_ts"].dt.to_period(code).rename(freq), metric)
    if totals.empty:
        return totals
    all_periods = pd.period_range(totals.index.min(), totals.index.max(), freq=code)
    return totals.reindex(all_periods, fill_value=0).to_timestamp().rename(metric)


def top_per_year(
    df: pd.DataFrame, by: Level, metric: Metric, n: int = 5
) -> pd.DataFrame:
    """The `n` most listened songs, artists, etc. of each year ("Wrapped").

    Returns one row per year and rank, with the identifying columns and the metric.
    """
    ranks = _ranks_per_year(df, by, metric)
    return ranks[ranks["rank"] <= n].reset_index(drop=True)


def rank_by_year(
    df: pd.DataFrame, by: Level, metric: Metric, n: int = 5
) -> pd.DataFrame:
    """The yearly ranks of the overall top `n` songs, artists, etc.

    Returns one row per year in which they were played, with the identifying
    columns, the metric and the rank. Rows are ordered by overall rank, then year.
    """
    overall = top(df, by, metric, n)[LEVEL_COLUMNS[by]]
    # how="right" keeps the order of `overall`, i.e. the overall ranking
    return _ranks_per_year(df, by, metric).merge(
        overall, on=LEVEL_COLUMNS[by], how="right"
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
