"""Read Spotify streaming history exports into pandas DataFrames."""

import json
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd

FILE_PATTERN = "Streaming_History_"


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
