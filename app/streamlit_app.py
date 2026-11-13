"""Web app: upload a Spotify extended streaming history zip and explore it.

Run with: streamlit run app/streamlit_app.py
"""

import io
import zipfile
from collections.abc import Callable
from datetime import date
from functools import partial

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from spotify_analysis import analysis, charts
from spotify_analysis.analysis import LEVEL_COLUMNS, Kind, Level, Metric
from spotify_analysis.load import clean_streaming_history, load_streaming_history

st.set_page_config(page_title="Spotify streaming history", layout="wide")

# Top lists start with this many entries; "More" and "Less" change it by
# STEP_ENTRIES per click.
FIRST_ENTRIES = 10
STEP_ENTRIES = 5

# Width of the upload box on the start page, in pixels: just enough for its
# "Upload" button and "200MB per file • ZIP" on one line (narrower, Streamlit
# stacks them, which makes the box taller).
UPLOADER_WIDTH = 370
# Height of that upload box, in pixels, measured in the browser.
UPLOADER_HEIGHT = 68

# Maximum width of a Wrapped table column, in pixels, and roughly how many
# characters fit into it. Longer names are cut off with "…".
MAX_COLUMN_WIDTH = 230
MAX_COLUMN_CHARS = 27


@st.cache_data(show_spinner="Reading your streaming history…")
def load(data: bytes) -> pd.DataFrame:
    """Load and clean an uploaded zip. Cached, so it only runs once per file."""
    return clean_streaming_history(load_streaming_history(io.BytesIO(data)))


def main() -> None:
    st.title("Your Spotify streaming history")
    # Before an upload, the uploader is on the main page, next to the hint; once a
    # file is uploaded, at the top of the sidebar, so it takes no space above the
    # charts. Thanks to its key, Streamlit keeps it (and the file) when it moves.
    if st.session_state.get("upload"):
        upload = st.sidebar.file_uploader(
            "Spotify data (zip)", type="zip", key="upload"
        )
    else:
        row = st.container(
            horizontal=True, vertical_alignment="bottom", wrap=False, key="upload_row"
        )
        # The hint as high as the upload box: less padding, and at least its height
        # (CSS for this row only, via the class Streamlit derives from the key).
        st.html(
            f"""<style>
            .st-key-upload_row [data-testid="stAlertContainer"] {{
                padding-top: 10px;
                padding-bottom: 10px;
                min-height: {UPLOADER_HEIGHT}px;
                display: flex;
                align-items: center;
            }}
            </style>"""
        )
        upload = row.file_uploader(
            "Spotify data (zip)", type="zip", key="upload", width=UPLOADER_WIDTH
        )
        row.info(
            "Request your **Extended streaming history** in your Spotify account "
            "under *Account privacy*. Spotify emails you a zip file after a few "
            "days; upload it here as it is, without unzipping."
        )
    if upload is None:
        return

    try:
        full = load(upload.getvalue())
    except (ValueError, zipfile.BadZipFile) as error:
        st.error(f"Could not read this file: {error}")
        return

    df, metric = sidebar(full)
    st.markdown(
        "Explore your listening history: what you listened to, when, and how it "
        "changed over the years. In the sidebar you choose the **period** and "
        "whether everything is counted in **plays** or in **hours**."
    )
    if df.empty:
        st.warning("No plays in the selected period.")
        return

    # Tabs only appear if the whole history has plays of that kind.
    names = ["Overview", "Music"]
    if not analysis.of_kind(full, "podcast").empty:
        names.append("Podcasts")
    if not analysis.of_kind(full, "audiobook").empty:
        names.append("Audiobooks")
    names.append("Over the years")
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Overview"]:
        overview_tab(df, metric)
    with tabs["Music"]:
        music_tab(df, metric)
    if "Podcasts" in tabs:
        with tabs["Podcasts"]:
            podcasts_tab(df, metric)
    if "Audiobooks" in tabs:
        with tabs["Audiobooks"]:
            audiobooks_tab(df, metric)
    with tabs["Over the years"]:
        years_tab(df, metric)


def sidebar(full: pd.DataFrame) -> tuple[pd.DataFrame, Metric]:
    """Date range and metric controls. Returns the filtered plays and the metric."""
    first = full["local_ts"].min().date()
    last = full["local_ts"].max().date()
    # The selected period lives in st.session_state["period"], so the button can
    # reset it. A newly uploaded file (with other dates) also resets it.
    if st.session_state.get("full_range") != (first, last):
        st.session_state["full_range"] = (first, last)
        st.session_state["period"] = (first, last)
    with st.sidebar:
        st.header("Settings")
        period = st.date_input("Period", min_value=first, max_value=last, key="period")
        st.button("Full range", on_click=reset_period, args=(first, last))
        metric = st.segmented_control(
            "Count by",
            ["plays", "hours"],
            format_func=str.capitalize,
            default="hours",
            required=True,
        )
    # While the user is picking a range, only the start date is set.
    start, end = period if len(period) == 2 else (period[0], last)
    return analysis.in_period(full, start, end), metric


def reset_period(first: date, last: date) -> None:
    """Button callback: select the full range again. Runs before the next rerun."""
    st.session_state["period"] = (first, last)


def overview_tab(df: pd.DataFrame, metric: Metric) -> None:
    section("At a glance", "Totals for the selected period.")
    figures = analysis.key_figures(df)
    tiles = {
        "Hours": figures["hours"],
        "Plays": figures["plays"],
        "Songs": figures["songs"],
        "Artists": figures["artists"],
        "Podcast shows": figures["shows"],
        "Audiobooks": figures["audiobooks"],
    }
    key_figure_tiles(tiles)
    st.caption(
        f"From {figures['first_play']:%d %b %Y} to {figures['last_play']:%d %b %Y}. "
        "Every play counts, also skipped ones."
    )

    kind, skipped, shuffle = st.columns(3)
    with kind:
        heading("Kind", f"Music, podcasts and audiobooks, by {metric}.")
        chart(
            charts.split_bar(
                analysis.shares(df, "kind", metric), charts.KIND_ORDER, "Kind"
            )
        )
    for container, column, title, text in [
        (skipped, "skipped", "Skipped", f"Share of {metric} on plays you skipped."),
        (shuffle, "shuffle", "Shuffle", f"Share of {metric} with shuffle turned on."),
    ]:
        with container:
            heading(title, text)
            chart(
                charts.split_bar(
                    analysis.shares(df, column, metric),
                    charts.YES_NO_ORDER,
                    title,
                    charts.YES_NO_LABELS,
                )
            )

    st.divider()
    section("How you listen")
    heading(
        "Devices over time",
        "Your listening per year by operating system. Plays in the web player "
        "count under the system it ran on.",
    )
    chart(charts.os_area(analysis.os_by_year(df, metric), metric))

    left, right = st.columns([1, 2])
    with left:
        heading("Countries", "Where you listened, as recorded by Spotify.")
        st.dataframe(analysis.by_country(df).round({"hours": 1}))
    with right:
        heading(
            "Play length",
            "How long each play lasted. The peak on the left are quick skips; plays "
            "of 10 minutes or more are in the last bar.",
        )
        chart(charts.play_length_histogram(df["ms_played"]))

    st.divider()
    section(
        "When you listen",
        "Your average listening per day, by weekday and hour (local time).",
    )
    by_hour = analysis.by_weekday_and_hour(df, metric)
    left, right = st.columns(2)
    with left:
        heading("By weekday and hour", "Darker means more listening.")
        chart(charts.weekday_hour_heatmap(by_hour, metric))
    with right:
        heading(
            "Over the day",
            "One line per weekday. Click a day in the legend to hide or show it.",
        )
        chart(charts.weekday_hour_lines(by_hour, metric))


def music_tab(df: pd.DataFrame, metric: Metric) -> None:
    section(
        "Top lists",
        f"Your most listened songs, artists and albums, by {metric}. Songs and "
        "albums count all their versions (e.g. single and album) together.",
    )
    for column, level in zip(st.columns(3), ["song", "artist", "album"]):
        with column:
            top_list(df, level, metric, f"Top {level}s", key=f"top_{level}")

    st.divider()
    section(
        "Most skipped",
        "The songs and artists you skip most often, as a share of their plays. "
        f"Only those with at least {analysis.MIN_PLAYS_FOR_SKIP_RATE} plays count.",
    )
    left, right = st.columns(2)
    for column, level in [(left, "song"), (right, "artist")]:
        with column:
            skip_list(df, level)

    st.divider()
    section(
        "Detail view",
        "Pick an artist, song or album to see how your listening to it developed.",
    )
    detail_view(df, metric)


def detail_view(df: pd.DataFrame, metric: Metric) -> None:
    """Pick one artist, song or album and show its listening."""
    level = st.segmented_control(
        "Look at one",
        ["artist", "song", "album"],
        format_func=str.capitalize,
        default="artist",
        required=True,
        key="detail_level",
    )
    # The 1,000 most listened ones, so the list stays manageable.
    candidates = analysis.top(df, level, metric, n=1000)[LEVEL_COLUMNS[level]]
    key = st.selectbox(
        f"Choose {'an' if level == 'artist' else 'a'} {level}",
        list(candidates.itertuples(index=False, name=None)),
        format_func=lambda key: " – ".join(reversed(key)),
    )
    if key is None:
        return
    selected = analysis.select(df, level, key)
    figures = analysis.key_figures(selected)
    tiles = {"Hours": figures["hours"], "Plays": figures["plays"]}
    if level != "song":
        tiles["Songs"] = figures["songs"]
    # Always three slots, so a song (without "Songs") has the same layout.
    key_figure_tiles(tiles, columns=3)
    st.caption(f"First heard on {figures['first_play']:%d %b %Y}")

    name = " – ".join(reversed(key))
    # Over time on the left, the songs on the right (empty for a song).
    left, right = st.columns([2, 1])
    with left:
        heading(name, f"Your {metric} per month.")
        # Always the whole selected period, so you can see when it started.
        period = (df["local_ts"].min(), df["local_ts"].max())
        listening = analysis.over_time(selected, metric, between=period)
        chart(charts.over_time_line(listening, metric, name))
    with right:
        if level == "artist":
            top_list(selected, "song", metric, "Top songs", key="detail_songs")
        elif level == "album":
            top_list(
                selected,
                "song",
                metric,
                "Songs you played",
                n=figures["songs"],
                explanation=(
                    "All songs of the album you played. Your data has no "
                    "tracklist, so unplayed songs are missing."
                ),
            )


def podcasts_tab(df: pd.DataFrame, metric: Metric) -> None:
    kind_tab(
        df, metric, "podcast", [("show", "Top shows"), ("episode", "Top episodes")]
    )


def audiobooks_tab(df: pd.DataFrame, metric: Metric) -> None:
    kind_tab(
        df,
        metric,
        "audiobook",
        [("audiobook", "Top audiobooks"), ("chapter", "Top chapters")],
    )


def kind_tab(
    df: pd.DataFrame, metric: Metric, kind: Kind, lists: list[tuple[Level, str]]
) -> None:
    """Layout shared by the podcast and audiobook tabs: two top lists in a row,
    then listening over time.

    Args:
        df: All plays in the selected period.
        metric: The selected metric.
        kind: Which plays to show.
        lists: Level and title of the two top lists, e.g. shows and episodes.
    """
    plays = analysis.of_kind(df, kind)
    if plays.empty:
        st.info(f"No {kind} plays in the selected period.")
        return
    section("Top lists", f"Your most listened {kind}s, by {metric}.")
    for column, (level, title) in zip(st.columns(2), lists):
        with column:
            top_list(plays, level, metric, title, key=f"top_{level}")

    st.divider()
    section("Over time", f"Your {kind} {metric} per month.")
    chart(
        charts.over_time_line(
            analysis.over_time(plays, metric), metric, f"{kind.capitalize()}s over time"
        )
    )


def years_tab(df: pd.DataFrame, metric: Metric) -> None:
    section(
        "Listening over time",
        f"Your {metric} per month over the whole selected period.",
    )
    chart(
        charts.over_time_line(
            analysis.over_time(df, metric), metric, "Listening per month"
        )
    )

    st.divider()
    section(
        "Your own Wrapped",
        "Your favourites of each year, and how the all-time favourites developed.",
    )
    level = st.segmented_control(
        "Show",
        ["artist", "song"],
        format_func=lambda level: f"{level.capitalize()}s",
        default="artist",
        required=True,
        key="years_level",
    )
    heading(
        f"Top {level}s per year",
        f"Your top {level}s of each year. Click a name to highlight it here and in "
        "the charts below; click it again to remove the highlight.",
    )
    clicked = wrapped_table(df, metric, level)
    heading(
        f"All-time top {level}s over the years",
        f"The all-time top 10 at the end of each year: all your {metric} up to then, "
        "ranked. A line that ends early dropped out of the top 10.",
    )
    chart(
        charts.all_time_bumps(
            analysis.all_time_ranks(df, level, metric),
            level,
            metric,
            "",
            highlight=clicked,
        )
    )
    heading(
        f"All-time top {level}s: {metric} so far",
        f"The same top 10 month by month, with their total {metric} so far. Steep "
        "stretches are times of heavy listening.",
    )
    chart(
        charts.all_time_totals(
            analysis.all_time_ranks(df, level, metric, freq="month"),
            level,
            metric,
            "",
            highlight=clicked,
        )
    )

    st.divider()
    section(
        "Discovery",
        "How many artists you played for the first time in each year. In the first "
        "year of your history, every artist counts as new.",
    )
    chart(charts.year_bars(analysis.new_per_year(df), "Discovery", "New artists"))

    st.divider()
    section("Calendar", f"One square per day; darker means more {metric}.")
    years = sorted(df["local_ts"].dt.year.unique(), reverse=True)
    year = st.selectbox("Calendar year", years, width=90)  # about as wide as its label
    daily = analysis.over_time(df, metric, "day")
    chart(charts.calendar_heatmap(daily, metric, year))

    st.divider()
    section("Records")
    left, right = st.columns(2)
    with left:
        heading("Top listening days", f"The days with the most {metric}.")
        expandable_chart(
            "top_days",
            lambda n: analysis.top_days(df, metric, n),
            lambda days: charts.top_days_bar(days, metric),
        )
    with right:
        heading("On repeat", "The most plays of a single song on one day.")
        expandable_chart(
            "on_repeat",
            lambda n: analysis.on_repeat(df, n),
            charts.on_repeat_bar,
        )


def top_list(
    df: pd.DataFrame,
    level: Level,
    metric: Metric,
    title: str,
    key: str | None = None,
    n: int = FIRST_ENTRIES,
    explanation: str | None = None,
) -> None:
    """A top list under a heading. With `key`, "Less" and "More" buttons below
    it change the number of entries; without, it shows `n` entries."""
    heading(title, explanation)

    def table_for(n: int) -> pd.DataFrame:
        return analysis.top(df, level, metric, n)

    def chart_for(table: pd.DataFrame) -> go.Figure:
        return charts.top_bar(table, level, metric, title)

    if key is None:
        chart(chart_for(table_for(n)))
    else:
        expandable_chart(key, table_for, chart_for)


def skip_list(df: pd.DataFrame, level: Level) -> None:
    """The most skipped songs or artists, with "Less" and "More" buttons."""
    heading(f"Most skipped {level}s")
    expandable_chart(
        f"skipped_{level}",
        lambda n: analysis.most_skipped(df, level, n),
        lambda table: charts.skip_bar(table, level, f"Most skipped {level}s"),
    )


def wrapped_table(df: pd.DataFrame, metric: Metric, level: Level) -> str | None:
    """The top songs or artists of each year, with "Less" and "More" buttons.

    Clicking a cell highlights every cell with the same entry, e.g. all years
    with Hans Zimmer; clicking it again removes the highlight.

    Returns the highlighted entry, if any, to highlight it elsewhere as well.
    """
    table_key = f"wrapped_{level}_table"
    highlighted = st.session_state.get(f"{table_key}_highlight")
    # A new widget key after every click (see toggle_highlight) starts the table
    # without a selected cell, so the same cell can be clicked again.
    widget_key = f"{table_key}_{st.session_state.get(f'{table_key}_version', 0)}"

    def show(n: int) -> int:
        grid = charts.wrapped_grid(analysis.top_per_year(df, level, metric, n), level)
        # Columns with longer names get a fixed width, i.e. a maximum; the others
        # keep their natural width.
        too_wide = grid.columns[
            grid.apply(lambda column: column.str.len().max()) > MAX_COLUMN_CHARS
        ]
        st.dataframe(
            charts.highlight(grid, highlighted),
            column_config={
                column: st.column_config.TextColumn(width=MAX_COLUMN_WIDTH)
                for column in too_wide
            },
            key=widget_key,
            # st.dataframe has no `args`, so they are bound to the callback here.
            on_select=partial(toggle_highlight, table_key, widget_key, grid),
            selection_mode="single-cell",
            height=35 * (len(grid) + 1) + 3,  # all rows, no scrolling
        )
        return len(grid)

    expandable(f"wrapped_{level}", show)
    return highlighted


def toggle_highlight(table_key: str, widget_key: str, grid: pd.DataFrame) -> None:
    """Table callback: highlight the clicked entry, or remove the highlight if it
    is already highlighted. Runs before the next rerun.

    The highlighted entry is kept in st.session_state[f"{table_key}_highlight"].

    Args:
        table_key: Name of the table, for the highlight and version keys.
        widget_key: Key of the clicked table widget, holding the click.
        grid: The table as it was clicked.
    """
    highlight_key = f"{table_key}_highlight"
    cells = st.session_state[widget_key]["selection"]["cells"]
    clicked = None
    if cells:
        row, column = cells[0]
        if row < len(grid) and column in grid.columns:
            clicked = grid.iloc[row][column] or None
    if clicked is None or clicked == st.session_state.get(highlight_key):
        st.session_state[highlight_key] = None
    else:
        st.session_state[highlight_key] = clicked
    # The browser keeps the clicked cell selected, and clicking a selected cell
    # again sends nothing. A new version makes the next run draw a fresh table
    # (new widget key) without a selection.
    version_key = f"{table_key}_version"
    st.session_state[version_key] = st.session_state.get(version_key, 0) + 1


def expandable_chart(
    key: str,
    table_for: Callable[[int], pd.DataFrame | pd.Series],
    chart_for: Callable[[pd.DataFrame | pd.Series], go.Figure],
) -> None:
    """A chart of a top list with "Less" and "More" buttons below it.

    Args:
        key: Unique name of this list; its length is kept in st.session_state.
        table_for: Calculates the list for a given number of entries.
        chart_for: Draws the list.
    """

    def show(n: int) -> int:
        table = table_for(n)
        chart(chart_for(table))
        return len(table)

    expandable(key, show)


def expandable(key: str, show: Callable[[int], int]) -> None:
    """Something with a variable number of entries, with "Less" and "More" below.

    It starts with FIRST_ENTRIES entries; each click adds or removes STEP_ENTRIES.
    "Less" is disabled at FIRST_ENTRIES, "More" once there are no more entries.

    Args:
        key: Unique name; the number of entries is kept in st.session_state.
        show: Draws `n` entries and returns how many there actually were.
    """
    n = st.session_state.get(key, FIRST_ENTRIES)
    shown = show(n)
    # Both buttons next to each other, centered below.
    buttons = st.container(horizontal=True, horizontal_alignment="center")
    buttons.button(
        "Less",
        key=f"{key}_less",
        icon=":material/expand_less:",
        icon_position="right",
        type="tertiary",
        on_click=set_length,
        args=(key, n - STEP_ENTRIES),
        disabled=n <= FIRST_ENTRIES,
    )
    buttons.button(
        "More",
        key=f"{key}_more",
        icon=":material/expand_more:",
        icon_position="right",
        type="tertiary",
        on_click=set_length,
        args=(key, n + STEP_ENTRIES),
        disabled=shown < n,
    )


def set_length(key: str, n: int) -> None:
    """Button callback: show `n` entries of list `key`."""
    st.session_state[key] = n


def section(title: str, explanation: str | None = None) -> None:
    """The heading of a section of a tab, with an optional short explanation."""
    st.subheader(title)
    if explanation:
        st.caption(explanation)


def heading(title: str, explanation: str | None = None) -> None:
    """The heading of a single chart or table, with an optional short explanation.

    One level smaller than a section heading, but larger than a chart's own title.
    """
    st.markdown(f"#### {title}")
    if explanation:
        st.caption(explanation)


def chart(fig: go.Figure) -> None:
    """Show a chart under a `heading`, without the chart's own (smaller) title.

    The space above the plot is reduced accordingly (charts with a legend keep
    room for it), and so is a fixed chart height, so the plot keeps its size.
    """
    has_legend = fig.layout.showlegend is not False and len(fig.data) > 1
    top = 40 if has_legend else 10
    saved = fig.layout.margin.t - top
    fig.update_layout(title_text="", margin_t=top)
    if fig.layout.height:
        fig.update_layout(height=fig.layout.height - saved)
    st.plotly_chart(fig)


def key_figure_tiles(tiles: dict[str, float], columns: int | None = None) -> None:
    """A row of number tiles. Tiles with 0 are hidden (e.g. no audiobooks).

    With `columns`, the row has that many slots, and the ones after the tiles stay
    empty; otherwise the tiles share the full width.
    """
    tiles = {label: value for label, value in tiles.items() if value}
    slots = st.columns(columns or len(tiles))
    for column, (label, value) in zip(slots, tiles.items()):
        column.metric(label, f"{value:,.0f}", border=True)


main()
