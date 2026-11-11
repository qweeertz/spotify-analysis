"""Read Spotify streaming history exports into pandas DataFrames."""

import json
import zipfile
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import IO

import pandas as pd

FILE_PATTERN = "Streaming_History_"

DROP_COLUMNS = [
    "ip_addr",
    "reason_start",
    "reason_end",
    "offline",
    "offline_timestamp",
    "incognito_mode",
]

RENAME_COLUMNS = {
    "master_metadata_track_name": "track",
    "master_metadata_album_artist_name": "artist",
    "master_metadata_album_album_name": "album",
    "spotify_track_uri": "track_uri",
    "episode_name": "episode",
    "episode_show_name": "show",
    "spotify_episode_uri": "episode_uri",
    "audiobook_title": "audiobook",
    "audiobook_chapter_title": "chapter",
    "conn_country": "country",
}


def load_streaming_history(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load all streams from a Spotify extended streaming history zip.

    The zip is read in memory; nothing is extracted to disk.

    Args:
        source: Path to the zip file, or a binary file-like object
            (e.g. a Streamlit upload).

    Returns:
        One row per play, with the columns as exported by Spotify.

    Raises:
        ValueError: If the zip contains no extended audio history files.
    """
    with zipfile.ZipFile(source) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if FILE_PATTERN in name and name.endswith(".json")
        )
        if not names:
            raise ValueError(
                "No extended streaming history found in the zip "
                f"(expected files named '{FILE_PATTERN}*.json')."
            )

        frames = []
        for name in names:
            with zf.open(name) as f:
                frames.append(pd.DataFrame(json.load(f)))

    return pd.concat(frames, ignore_index=True)


def clean_streaming_history(
    df: pd.DataFrame, timezones: dict[str, str] | None = None
) -> pd.DataFrame:
    """Turn raw streaming history into an analysis-ready DataFrame.

    Drops unneeded columns, renames the rest, parses timestamps, adds the
    local time of each play and its kind, and sorts chronologically.
    The input DataFrame is not modified.

    Args:
        df: Raw history as returned by `load_streaming_history`.
        timezones: Optional overrides mapping country codes to IANA time
            zones, e.g. ``{"ES": "Atlantic/Canary"}``. Countries not listed
            use their main time zone.

    Returns:
        A new DataFrame sorted by ``ts``. ``ts`` is the play's end time in UTC;
        ``local_ts`` is the wall-clock time where it was played (without time
        zone). ``kind`` is one of "music", "podcast", "audiobook" or "unknown".
    """
    df = df.drop(columns=DROP_COLUMNS, errors="ignore").rename(columns=RENAME_COLUMNS)

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["local_ts"] = _local_time(df["ts"], df["country"], timezones or {})

    df["kind"] = "unknown"
    df.loc[df["audiobook"].notna(), "kind"] = "audiobook"
    df.loc[df["episode"].notna(), "kind"] = "podcast"
    df.loc[df["track"].notna(), "kind"] = "music"

    return df.sort_values("ts", ignore_index=True)


def _local_time(
    ts: pd.Series, country: pd.Series, overrides: dict[str, str]
) -> pd.Series:
    """Convert UTC timestamps to local wall-clock time, based on the country.

    Unknown countries (e.g. Spotify's "ZZ") get the time zone of the country
    with the most plays, i.e. the user's home.
    """
    zones = _country_timezones() | overrides
    tz_per_row = country.map(zones)
    home_tz = tz_per_row.mode().iloc[0] if tz_per_row.notna().any() else "UTC"
    tz_per_row = tz_per_row.fillna(home_tz)

    parts = [
        ts[tz_per_row == tz].dt.tz_convert(tz).dt.tz_localize(None)
        for tz in tz_per_row.unique()
    ]
    return pd.concat(parts).sort_index()


@cache
def _country_timezones() -> dict[str, str]:
    """Map each country code to its main time zone, from the IANA zone.tab.

    zone.tab lists a country's most populous zone first, so that one is used.
    """
    text = files("tzdata").joinpath("zoneinfo", "zone.tab").read_text()
    zones: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        country, _coordinates, tz, *_ = line.split("\t")
        zones.setdefault(country, tz)
    return zones
