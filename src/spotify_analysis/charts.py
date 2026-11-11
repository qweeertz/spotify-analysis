"""Plotly figures for the results of `analysis`."""

from collections.abc import Hashable, Sequence

import pandas as pd
import plotly.graph_objects as go

from spotify_analysis.analysis import LEVEL_COLUMNS, Level, Metric

# Categorical colors, assigned in this fixed order. A value always gets the color of
# its position in an *_ORDER list below, so it keeps its color when others are
# filtered out.
COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
# One hue from light to dark, for magnitudes (heatmap).
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
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

# Longer song and album titles are shortened on the axis; the tooltip shows them
# in full.
MAX_TITLE_LENGTH = 35


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
    """Distribution of play lengths in minutes.

    Plays longer than `max_minutes` are counted in the last bar.
    """
    bin_size = 0.25
    minutes = (ms_played / 60_000).clip(upper=max_minutes)
    fig = go.Figure(
        go.Histogram(
            x=minutes,
            xbins={"start": 0, "end": max_minutes + bin_size, "size": bin_size},
            marker={"color": COLORS[0], "line": {"color": SURFACE, "width": 1}},
            hovertemplate="%{x} min: %{y:,} plays<extra></extra>",
        )
    )
    fig.update_xaxes(title=f"Minutes played (last bar: {max_minutes}+ min)")
    fig.update_yaxes(title="Plays")
    return _style(fig, "Play length")


def weekday_hour_heatmap(table: pd.DataFrame, metric: Metric) -> go.Figure:
    """Heatmap of listening per weekday and hour of day.

    Args:
        table: Weekdays × hours, as returned by `analysis.by_weekday_and_hour`.
        metric: The metric the table was calculated with.
    """
    label = METRIC_LABELS[metric]
    fig = go.Figure(
        go.Heatmap(
            z=table.values,
            x=table.columns,
            y=table.index,
            colorscale=SEQUENTIAL,
            xgap=2,
            ygap=2,
            colorbar={"title": label},
            hovertemplate=(
                f"%{{y}}, %{{x}}:00<br>%{{z:{METRIC_FORMATS[metric]}}} "
                f"{label.lower()}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Hour of day", dtick=2, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Monday on top
    return _style(fig, "When you listen")


def weekday_hour_lines(
    table: pd.DataFrame, metric: Metric, weekdays: Sequence[str] | None = None
) -> go.Figure:
    """Listening over the hours of the day, one line per weekday.

    Weekdays can also be hidden by clicking them in the legend.

    Args:
        table: Weekdays × hours, as returned by `analysis.by_weekday_and_hour`.
        metric: The metric the table was calculated with.
        weekdays: Only show these weekdays. Default: all.
    """
    fig = go.Figure()
    for color, (weekday, row) in zip(COLORS, table.iterrows()):
        if weekdays is not None and weekday not in weekdays:
            continue
        fig.add_trace(
            go.Scatter(
                x=row.index,
                y=row.values,
                name=weekday,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"%{{y:{METRIC_FORMATS[metric]}}}",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title=METRIC_LABELS[metric], rangemode="tozero")
    return _style(fig, "Listening by hour of day")


def top_bar(table: pd.DataFrame, by: Level, metric: Metric, title: str) -> go.Figure:
    """Horizontal bars of a top list, the first entry on top.

    Args:
        table: Top list, as returned by `analysis.top`.
        by: The level the table was calculated for.
        metric: The metric the table was calculated with.
        title: Chart title.
    """
    names = _names(table, by)
    fmt = METRIC_FORMATS[metric]
    fig = go.Figure(
        go.Bar(
            x=table[metric],
            y=names,
            customdata=names,
            orientation="h",
            marker={"color": COLORS[0]},
            texttemplate=f"%{{x:{fmt}}}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(f"%{{customdata}}<br>%{{x:{fmt}}} {metric}<extra></extra>"),
        )
    )
    fig.update_xaxes(title=METRIC_LABELS[metric])
    return _style_bars(fig, title, table, by)


def skip_bar(table: pd.DataFrame, by: Level, title: str) -> go.Figure:
    """Horizontal bars of skip rates, the highest on top.

    Args:
        table: As returned by `analysis.most_skipped`.
        by: The level the table was calculated for.
        title: Chart title.
    """
    names = _names(table, by)
    fig = go.Figure(
        go.Bar(
            x=table["skip_rate"],
            y=names,
            customdata=list(zip(names, table["plays"])),
            orientation="h",
            marker={"color": COLORS[0]},
            texttemplate="%{x:.0%}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{customdata[0]}<br>%{x:.0%} of %{customdata[1]:,} plays skipped"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Skip rate", tickformat=".0%", range=[0, 1])
    return _style_bars(fig, title, table, by)


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


def _names(
    table: pd.DataFrame, by: Level, max_title_length: int | None = None
) -> list[str]:
    """Display names: "song – artist", "album – artist" or just "artist".

    With `max_title_length`, a longer song or album title is shortened with "…";
    the artist is always kept in full.
    """
    columns = list(reversed(LEVEL_COLUMNS[by]))
    names = []
    for title, *rest in table[columns].astype(str).itertuples(index=False):
        if max_title_length and len(title) > max_title_length:
            title = title[: max_title_length - 1] + "…"
        names.append(" – ".join([title, *rest]))
    return names


def _style_bars(
    fig: go.Figure, title: str, table: pd.DataFrame, by: Level
) -> go.Figure:
    """Shared look of horizontal bar charts: thin rounded bars, first entry on top.

    The bars are placed by their full name, which is unique (Plotly merges bars
    with equal names); the axis shows the shortened names.
    """
    fig.update_layout(height=120 + 36 * len(table), bargap=0.35, barcornerradius=4)
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        tickmode="array",
        tickvals=_names(table, by),
        ticktext=_names(table, by, MAX_TITLE_LENGTH),
    )
    return _style(fig, title)


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
