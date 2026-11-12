"""Plotly figures (and display tables) for the results of `analysis`."""

from collections.abc import Hashable, Sequence

import pandas as pd
import plotly.graph_objects as go

from spotify_analysis.analysis import LEVEL_COLUMNS, Level, Metric

# Categorical colors, assigned in this fixed order. A value always gets the color of
# its position in an *_ORDER list below, so it keeps its color when others are
# filtered out. The first 8 are a palette tested for color-blind readability; the
# last 2 were added so a top 10 can be fully colored, chosen to be as distinct as
# possible from the others (also with color blindness) and from the grey MUTED.
COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
    "#82074d",  # burgundy
    "#683a19",  # brown
]
# One hue from light to dark, for magnitudes (heatmap).
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE = "#fcfcfb"
# COLORS[0] as a translucent wash, for highlighted table cells.
HIGHLIGHT = "rgba(42, 120, 214, 0.35)"
GRID = "#e1e0d9"
# Grey for lines and labels that are not highlighted.
MUTED = "#a8a7a0"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

KIND_ORDER = ["music", "podcast", "audiobook", "unknown"]
YES_NO_ORDER = [True, False]
YES_NO_LABELS = {True: "Yes", False: "No"}
OS_ORDER = [
    "Windows",
    "Android",
    "macOS",
    "iOS",
    "Linux",
    "TV",
    "Other device",
    "Other",
]

METRIC_LABELS = {"plays": "Plays", "hours": "Hours"}
METRIC_FORMATS = {"plays": ",.0f", "hours": ",.1f"}

# Longer titles and artist names are shortened on chart axes; the tooltip shows
# them in full.
MAX_TITLE_LENGTH = 28
MAX_ARTIST_LENGTH = 20

# Horizontal bar charts estimate label widths from this many characters across a
# chart (about a third of a wide screen), to leave room for the names right of
# the bars. A bar with a very long name still keeps at least MIN_BAR_SHARE.
CHARS_PER_WIDTH = 60
# The same for charts across the full width of a wide screen.
WIDE_CHARS_PER_WIDTH = 170
MIN_BAR_SHARE = 0.25

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def split_bar(
    shares: pd.Series,
    order: Sequence[Hashable],
    title: str,
    labels: dict[Hashable, str] | None = None,
) -> go.Figure:
    """One horizontal bar split into shares, e.g. music vs. podcasts.

    Args:
        shares: Share per value, as returned by `analysis.shares`.
        order: All possible values; fixes their position and color.
        title: Chart title.
        labels: Optional display names, e.g. ``{True: "Yes"}``.
    """
    labels = labels or {}
    fig = go.Figure()
    for color, value in zip(COLORS, order):
        if value not in shares.index:
            continue
        name = labels.get(value, str(value).capitalize())
        fig.add_trace(
            go.Bar(
                x=[shares[value]],
                y=[""],
                orientation="h",
                name=name,
                marker={"color": color, "line": {"color": SURFACE, "width": 2}},
                texttemplate="%{x:.0%}",
                textposition="inside",
                insidetextanchor="start",
                hovertemplate=f"{name}: %{{x:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=150,
        uniformtext={"minsize": 11, "mode": "hide"},  # hide labels that don't fit
    )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)
    return _style(fig, title)


def os_area(table: pd.DataFrame, metric: Metric) -> go.Figure:
    """Stacked area of listening per year and operating system.

    Args:
        table: Years × operating systems, as returned by `analysis.os_by_year`.
        metric: The metric the table was calculated with.
    """
    fig = go.Figure()
    for color, os_name in zip(COLORS, OS_ORDER):
        if os_name not in table.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=table.index,
                y=table[os_name],
                name=os_name,
                mode="lines",
                stackgroup="os",
                line={"color": color, "width": 2},
                hovertemplate=f"%{{y:{METRIC_FORMATS[metric]}}}",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title=METRIC_LABELS[metric])
    return _style(fig, "Devices over time")


def play_length_histogram(ms_played: pd.Series, max_minutes: int = 10) -> go.Figure:
    """Distribution of play lengths in minutes, in bars of 15 seconds.

    Plays longer than `max_minutes` are counted in the last bar.
    """
    # Counted here rather than with go.Histogram, so the tooltip can show
    # readable ranges like "3 min – 3 min 15 s" instead of "3–3.25".
    bin_seconds = 15
    seconds = (ms_played / 1000).clip(upper=max_minutes * 60)
    last_bin = max_minutes * 60 // bin_seconds  # holds everything >= max_minutes
    counts = (
        (seconds // bin_seconds)
        .astype(int)
        .value_counts()
        .reindex(range(last_bin + 1), fill_value=0)
    )
    starts = counts.index * bin_seconds
    labels = [
        f"{_duration(start)} – {_duration(start + bin_seconds)}" for start in starts
    ]
    labels[-1] = f"{max_minutes} min or more"
    fig = go.Figure(
        go.Bar(
            x=(starts + bin_seconds / 2) / 60,  # bar centers, in minutes
            y=counts.values,
            width=bin_seconds / 60,
            customdata=labels,
            marker={"color": COLORS[0], "line": {"color": SURFACE, "width": 1}},
            hovertemplate="%{customdata}: %{y:,} plays<extra></extra>",
        )
    )
    fig.update_xaxes(title=f"Minutes played (last bar: {max_minutes}+ min)")
    fig.update_yaxes(title="Plays")
    return _style(fig, "Play length")


def weekday_hour_heatmap(table: pd.DataFrame, metric: Metric) -> go.Figure:
    """Heatmap of average listening per weekday and hour of day.

    Args:
        table: Weekdays × hours, as returned by `analysis.by_weekday_and_hour`.
        metric: The metric the table was calculated with.
    """
    table, unit, fmt = _per_day(table, metric)
    fig = go.Figure(
        go.Heatmap(
            z=table.values,
            x=table.columns,
            y=table.index,
            colorscale=SEQUENTIAL,
            xgap=2,
            ygap=2,
            colorbar={"title": f"{unit.capitalize()}<br>per day"},
            hovertemplate=(
                f"%{{y}}s, %{{x}}:00–%{{x}}:59<br>%{{z:{fmt}}} {unit} on average"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Hour of day", dtick=2, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Monday on top
    return _style(fig, "When you listen")


def weekday_hour_lines(
    table: pd.DataFrame, metric: Metric, weekdays: Sequence[str] | None = None
) -> go.Figure:
    """Average listening over the hours of the day, one line per weekday.

    Weekdays can also be hidden by clicking them in the legend.

    Args:
        table: Weekdays × hours, as returned by `analysis.by_weekday_and_hour`.
        metric: The metric the table was calculated with.
        weekdays: Only show these weekdays. Default: all.
    """
    table, unit, fmt = _per_day(table, metric)
    fig = go.Figure()
    # The table always has all weekdays, Monday first; short names keep the legend
    # on one line.
    for color, label, (weekday, row) in zip(COLORS, WEEKDAY_LABELS, table.iterrows()):
        if weekdays is not None and weekday not in weekdays:
            continue
        fig.add_trace(
            go.Scatter(
                x=row.index,
                y=row.values,
                name=label,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"%{{y:{fmt}}} {unit}",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title=f"{unit.capitalize()} per day", rangemode="tozero")
    # No title: shown next to the heatmap, whose title covers both.
    return _style(fig, "")


def top_bar(table: pd.DataFrame, by: Level, metric: Metric, title: str) -> go.Figure:
    """Horizontal bars of a top list, the first entry on top.

    Args:
        table: Top list, as returned by `analysis.top`.
        by: The level the table was calculated for.
        metric: The metric the table was calculated with.
        title: Chart title.
    """
    fmt = METRIC_FORMATS[metric]
    fig = _hbar(
        table[metric],
        _names(table, by, shorten=True),
        hover_value=f"%{{x:{fmt}}} {metric}",
    )
    return _style(fig, title)


def skip_bar(table: pd.DataFrame, by: Level, title: str) -> go.Figure:
    """Horizontal bars of skip rates, the highest on top.

    Args:
        table: As returned by `analysis.most_skipped`.
        by: The level the table was calculated for.
        title: Chart title.
    """
    fig = _hbar(
        table["skip_rate"],
        _names(table, by, shorten=True),
        hover_value="%{x:.0%} of %{customdata:,} plays skipped",
        extra=table["plays"],
    )
    return _style(fig, title)


def top_days_bar(series: pd.Series, metric: Metric) -> go.Figure:
    """Horizontal bars of the days with the most listening.

    Args:
        series: As returned by `analysis.top_days`.
        metric: The metric the series was calculated with.
    """
    fmt = METRIC_FORMATS[metric]
    days = [f"{day:%a, %d %b %Y}" for day in series.index]
    fig = _hbar(series, days, hover_value=f"%{{x:{fmt}}} {metric}")
    return _style(fig, "Top listening days")


def on_repeat_bar(table: pd.DataFrame) -> go.Figure:
    """Horizontal bars of the most plays of one song on a single day.

    Args:
        table: As returned by `analysis.on_repeat`.
    """
    days = [f"{day:%d %b %Y}" for day in table["day"]]
    names = [
        f"{name} · {day}"
        for name, day in zip(_names(table, "song", shorten=True), days)
    ]
    fig = _hbar(table["plays"], names, hover_value="%{x:,} plays")
    return _style(fig, "On repeat")


def over_time_line(series: pd.Series, metric: Metric, title: str) -> go.Figure:
    """Line of listening over time, e.g. per month.

    Args:
        series: As returned by `analysis.over_time`.
        metric: The metric the series was calculated with.
        title: Chart title.
    """
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line={"color": COLORS[0], "width": 2},
            fill="tozeroy",
            fillcolor="rgba(42, 120, 214, 0.1)",  # COLORS[0] as a light wash
            hovertemplate=f"%{{y:{METRIC_FORMATS[metric]}}} {metric}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(hoverformat="%b %Y")
    fig.update_yaxes(title=METRIC_LABELS[metric], rangemode="tozero")
    return _style(fig, title)


def wrapped_grid(table: pd.DataFrame, by: Level) -> pd.DataFrame:
    """The top entries of each year as a grid: one row per rank, one column per year.

    A DataFrame rather than a figure, meant for `st.dataframe`, which follows the
    app's light or dark theme and scrolls sideways when there are many years.

    Args:
        table: As returned by `analysis.top_per_year`.
        by: The level the table was calculated for.
    """
    grid = table.assign(name=_names(table, by)).pivot(
        index="rank", columns="year", values="name"
    )
    grid.index = [f"#{rank}" for rank in grid.index]
    grid.columns = grid.columns.astype(str)
    return grid.fillna("")


def highlight(grid: pd.DataFrame, value: str | None) -> "pd.io.formats.style.Styler":
    """The grid with every cell equal to `value` highlighted, e.g. one artist.

    The highlight is a semi-transparent blue, readable in light and dark mode.
    """
    style = f"background-color: {HIGHLIGHT}"
    return grid.style.map(lambda cell: style if value and cell == value else "")


def all_time_bumps(
    table: pd.DataFrame,
    by: Level,
    metric: Metric,
    title: str,
    highlight: str | None = None,
) -> go.Figure:
    """Bump chart of the all-time top list over the years, rank 1 on top.

    One line per song, artist, etc. while it is in the top list; a line that ends
    early is one that dropped out. Without `highlight`, the current top 10 are
    colored and the others grey; with it, only that one is colored. Only the
    current standings are named; the tooltip names every line.

    Args:
        table: As returned by `analysis.all_time_ranks`.
        by: The level the table was calculated for.
        metric: The metric the table was calculated with.
        title: Chart title.
        highlight: Full name (as in `wrapped_grid`) of the one to color.
    """
    fig = _all_time_lines(table, by, metric, highlight, y="rank")
    n = table["rank"].max()
    fig.update_yaxes(
        title="All-time rank",
        tickvals=list(range(1, n + 1)),
        range=[n + 0.5, 0.5],
        showgrid=False,
    )
    return _style(fig, title)


def all_time_totals(
    table: pd.DataFrame,
    by: Level,
    metric: Metric,
    title: str,
    highlight: str | None = None,
) -> go.Figure:
    """Like `all_time_bumps`, but with the total so far instead of the rank.

    Shows how far apart the entries are and how fast they grow, like the scores
    in a "bar chart race" video. Arguments as for `all_time_bumps`.
    """
    fig = _all_time_lines(table, by, metric, highlight, y=metric)
    fig.update_yaxes(title=f"{METRIC_LABELS[metric]} so far", rangemode="tozero")
    return _style(fig, title)


def _all_time_lines(
    table: pd.DataFrame, by: Level, metric: Metric, highlight: str | None, y: str
) -> go.Figure:
    """The lines shared by `all_time_bumps` and `all_time_totals`.

    Works with yearly and monthly tables (a "year" or "month" column). The periods
    are placed at 0, 1, 2, ... on the x-axis, labeled with their years.

    Args:
        y: The column to plot: "rank", or the metric for the totals so far.
    """
    freq = "month" if "month" in table.columns else "year"
    table = table.assign(name=_names(table, by), label=_names(table, by, shorten=True))
    if freq == "year":
        periods = list(range(table["year"].min(), table["year"].max() + 1))
        period_names = [str(year) for year in periods]
        ticks = {i: str(year) for i, year in enumerate(periods)}
    else:
        periods = list(
            pd.date_range(table["month"].min(), table["month"].max(), freq="MS")
        )
        period_names = [f"{month:%b %Y}" for month in periods]
        ticks = {
            i: str(month.year) for i, month in enumerate(periods) if month.month == 1
        }
    x = list(range(len(periods)))
    last = x[-1]
    current = table[table[freq] == periods[-1]]
    # Dots only when there are few enough periods to tell them apart.
    mode = "lines+markers" if len(periods) <= 30 else "lines"

    def color(name: str) -> str | None:
        if highlight:
            return COLORS[0] if name == highlight else None
        position = (
            list(current["name"]).index(name)
            if name in set(current["name"])
            else len(COLORS)
        )
        return COLORS[position] if position < len(COLORS) else None

    fig = go.Figure()
    # Grey lines first, so the colored ones are drawn on top of them.
    groups = sorted(table.groupby("name", sort=False), key=lambda g: bool(color(g[0])))
    for name, rows in groups:
        rows = rows.set_index(freq).reindex(periods)  # gaps where not in the top
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rows[y],
                customdata=list(zip(period_names, rows["rank"], rows[metric])),
                mode=mode,
                line={"color": color(name) or MUTED, "width": 3 if color(name) else 2},
                marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                hovertemplate=(
                    f"{name}<br>%{{customdata[0]}}: rank %{{customdata[1]}}, "
                    f"%{{customdata[2]:{METRIC_FORMATS[metric]}}} {metric} so far"
                    "<extra></extra>"
                ),
            )
        )

    # The current standings get their name on the right. Dropped-out ones don't:
    # their names would collide with other lines; hover shows them. Names too
    # close to each other are moved apart, with a thin line to their line's end.
    ends = list(current[y])
    positions = _spread(ends, min_gap=_label_gap(table[y], ranks=y == "rank"))
    offset = max(0.25, 0.02 * len(periods))
    for (_, row), end, position in zip(current.iterrows(), ends, positions):
        ink = None if color(row["name"]) else MUTED
        fig.add_trace(
            go.Scatter(
                x=[last, last + offset],
                y=[end, position],
                mode="lines+text",
                line={"color": GRID, "width": 1},
                text=["", f" {row['label']}"],
                textposition="middle right",
                textfont={"size": 11, "color": ink},
                cliponaxis=False,
                hoverinfo="skip",
            )
        )

    # Room on the right for the longest name, estimated from its length: it takes
    # `share` of the chart width, so the periods get the rest.
    longest = current["label"].str.len().max() + 2
    share = min(longest / WIDE_CHARS_PER_WIDTH, 0.6)
    room_for_names = (last + 0.3 + offset) * share / (1 - share) + offset
    fig.update_layout(height=520, showlegend=False)
    fig.update_xaxes(
        tickvals=list(ticks),
        ticktext=list(ticks.values()),
        range=[-0.3, last + room_for_names],
    )
    return fig


def _label_gap(values: pd.Series, ranks: bool) -> float:
    """The smallest vertical distance between two names, in data units.

    Ranks are one apart anyway; for totals it is about a name's height
    (1/26 of a chart of the usual height).
    """
    return 1 if ranks else values.max() / 26


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Positions for labels near `values`, at least `min_gap` apart.

    Goes from the top value down and pushes each label below the previous one if
    needed, so the order stays the same.
    """
    order = sorted(range(len(values)), key=lambda i: -values[i])
    positions = list(values)
    previous = None
    for i in order:
        if previous is not None and positions[i] > previous - min_gap:
            positions[i] = previous - min_gap
        previous = positions[i]
    return positions


def year_bars(series: pd.Series, title: str, y_title: str) -> go.Figure:
    """Columns of one value per year, e.g. new artists per year.

    Args:
        series: Values indexed by year, e.g. from `analysis.new_per_year`.
        title: Chart title.
        y_title: What the values are, e.g. "New artists".
    """
    fig = go.Figure(
        go.Bar(
            x=series.index,
            y=series.values,
            marker={"color": COLORS[0]},
            texttemplate="%{y:,}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{x}}: %{{y:,}} {y_title.lower()}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.35, barcornerradius=4)
    fig.update_xaxes(dtick=1, showgrid=False)
    fig.update_yaxes(title=y_title)
    return _style(fig, title)


def calendar_heatmap(daily: pd.Series, metric: Metric, year: int) -> go.Figure:
    """One square per day of `year`: weeks as columns, weekdays as rows.

    Args:
        daily: Listening per day, as returned by `analysis.over_time(..., "day")`.
        metric: The metric the series was calculated with.
        year: The year to show.
    """
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    values = daily.reindex(days)  # days outside the history stay empty (NaN)
    first_monday = days[0] - pd.Timedelta(days=days[0].dayofweek)
    week = (days - first_monday).days // 7
    grid = pd.DataFrame({"week": week, "weekday": days.dayofweek, "value": values})
    z = grid.pivot(index="weekday", columns="week", values="value")
    dates = grid.assign(date=days.strftime("%a, %d %b %Y")).pivot(
        index="weekday", columns="week", values="date"
    )
    month_starts = days[days.day == 1]
    label = METRIC_LABELS[metric]
    fig = go.Figure(
        go.Heatmap(
            z=z.values,
            x=z.columns,
            y=WEEKDAY_LABELS,
            customdata=dates.values,
            colorscale=SEQUENTIAL,
            xgap=2,
            ygap=2,
            colorbar={"title": label},
            hovertemplate=(
                f"%{{customdata}}<br>%{{z:{METRIC_FORMATS[metric]}}} "
                f"{label.lower()}<extra></extra>"
            ),
        )
    )
    fig.update_layout(height=280)
    fig.update_xaxes(
        tickvals=(month_starts - first_monday).days // 7,
        ticktext=month_starts.strftime("%b"),
        showgrid=False,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, scaleanchor="x")
    return _style(fig, f"Listening in {year}")


def _duration(seconds: int) -> str:
    """Readable duration, e.g. "3 min 15 s", "3 min" or "45 s"."""
    minutes, seconds = divmod(int(seconds), 60)
    if not minutes:
        return f"{seconds} s"
    if not seconds:
        return f"{minutes} min"
    return f"{minutes} min {seconds} s"


def _per_day(table: pd.DataFrame, metric: Metric) -> tuple[pd.DataFrame, str, str]:
    """Daily averages in readable units: plays, or minutes instead of hours.

    Returns the converted table, the unit and its number format.
    """
    if metric == "hours":
        return table * 60, "minutes", ",.0f"
    return table, "plays", ",.1f"


def _hbar(
    x: pd.Series,
    names: list[str],
    hover_value: str,
    extra: pd.Series | None = None,
) -> go.Figure:
    """Horizontal bar chart: ranks on the left, then the bars, names to their right.

    The values are only shown in the tooltip.

    Args:
        x: Bar lengths.
        names: Names, shown to the right of the bars.
        hover_value: Tooltip text, e.g. ``"%{x:,} plays"``. Can use
            ``%{customdata}`` for `extra`.
        extra: Optional additional value per bar for the tooltip.
    """
    # Bars at 0, 1, 2, ... rather than at their names, which may repeat.
    positions = list(range(len(names)))
    fig = go.Figure(
        [
            go.Bar(
                x=x,
                y=positions,
                width=0.6,
                customdata=extra,
                orientation="h",
                marker={"color": COLORS[0]},
                hovertemplate=f"{hover_value}<extra></extra>",
            ),
            # The names, as text just to the right of each bar's end.
            go.Scatter(
                x=x,
                y=positions,
                mode="text",
                text=[f"  {name}" for name in names],
                textposition="middle right",
                cliponaxis=False,
                hoverinfo="skip",
            ),
        ]
    )
    fig.update_layout(height=120 + 36 * len(names), barcornerradius=4, showlegend=False)
    # The axis reaches past the longest bar to make room for the names.
    fig.update_xaxes(range=[0, _hbar_range(x, names)], visible=False)
    # The rank of each bar on its left: 1., 2., 3., ...
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        tickmode="array",
        tickvals=positions,
        ticktext=[f"{position + 1}." for position in positions],
        ticks="",
    )
    return fig


def _hbar_range(x: pd.Series, names: list[str]) -> float:
    """The x-axis end, so that every bar plus the name to its right fits.

    Name widths are estimated from their length, assuming about CHARS_PER_WIDTH
    characters fit across the chart.
    """
    x_max = 0.0
    for length, name in zip(x, names):
        name_share = (len(name) + 2) / CHARS_PER_WIDTH
        x_max = max(x_max, length / max(1 - name_share, MIN_BAR_SHARE))
    return x_max or 1


def _names(table: pd.DataFrame, by: Level, shorten: bool = False) -> list[str]:
    """Display names, e.g. "song – artist", "episode – show" or just "artist".

    With `shorten`, long titles (song, album, episode) and long artist or show
    names are cut off with "…", so they fit narrow charts.
    """
    columns = list(reversed(LEVEL_COLUMNS[by]))
    names = []
    for title, *rest in table[columns].astype(str).itertuples(index=False):
        if shorten:
            title = _shorten(title, MAX_TITLE_LENGTH)
            rest = [_shorten(part, MAX_ARTIST_LENGTH) for part in rest]
        names.append(" – ".join([title, *rest]))
    return names


def _shorten(text: str, max_length: int) -> str:
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def _style(fig: go.Figure, title: str) -> go.Figure:
    """Shared look: title, legend above the plot, light hairline grid."""
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0, "xanchor": "left"},
        font={"family": FONT},
        margin={"l": 10, "r": 30, "t": 80, "b": 10},
        # "normal": same order as the traces (stacked charts reverse it by default)
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.02,
            "yanchor": "bottom",
            "traceorder": "normal",
        },
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
