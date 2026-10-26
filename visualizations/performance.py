"""Performance-oriented charts for pace, sectors, and consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.analytics import format_lap_time
from .common import TEXT_COLOR, MUTED_TEXT, FONT_FAMILY, apply_dark_theme, driver_line_style, empty_figure


def create_lap_time_progression(
    laps: pd.DataFrame,
    drivers: list[str],
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Plot lap-by-lap pace progression for the selected drivers with clean top-right legend."""
    if laps.empty or "Driver" not in laps.columns or "LapTimeSeconds" not in laps.columns:
        return empty_figure("Lap timing data is unavailable for this selection.", height=380)

    filtered = laps.loc[laps["Driver"].isin(drivers) & laps["LapTimeSeconds"].notna()].copy()
    if filtered.empty:
        return empty_figure("Lap timing data is unavailable for this selection.", height=380)

    fig = go.Figure()
    for index, driver in enumerate(drivers):
        driver_laps = filtered.loc[filtered["Driver"] == driver].sort_values("LapNumber")
        if driver_laps.empty:
            continue
        dash, symbol = driver_line_style(index)
        fig.add_trace(
            go.Scatter(
                x=driver_laps["LapNumber"],
                y=driver_laps["LapTimeSeconds"],
                mode="lines+markers",
                name=driver,
                line=dict(color=driver_colors.get(driver, "#E10600"), width=2.4, dash=dash),
                marker=dict(size=6, symbol=symbol),
                customdata=np.stack(
                    [
                        driver_laps["Compound"].astype(str),
                        driver_laps["LapTimeSeconds"].round(3),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Lap %{x}<br>"
                    "Time %{customdata[1]}s<br>"
                    "Compound %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        yaxis_title="Lap Time (s)",
        xaxis_title="Lap Number",
    )
    return apply_dark_theme(fig, title=title, height=430, legend_orientation="h")


def create_fastest_lap_comparison(
    summary: pd.DataFrame,
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Compare fastest laps across drivers with clear horizontal bar layout."""
    if summary.empty:
        return empty_figure("Fastest lap comparison will appear once the session loads.", height=360)

    ordered = summary.sort_values("FastestLapSeconds", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["FastestLapSeconds"],
            y=ordered["Driver"],
            orientation="h",
            marker=dict(
                color=[driver_colors.get(driver, "#E10600") for driver in ordered["Driver"]],
                line=dict(color="#1C1C1C", width=1),
            ),
            text=ordered["FastestLap"],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11, family=FONT_FAMILY),
            hovertemplate="<b>%{y}</b><br>Fastest lap %{text}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Fastest Lap (s)", yaxis_title="")
    fig.update_yaxes(categoryorder="array", categoryarray=ordered["Driver"][::-1])
    return apply_dark_theme(
        fig,
        title=title,
        height=420,
        show_legend=False,
        margin=dict(l=45, r=60, t=25 if not title else 50, b=40),
    )


def create_sector_comparison(
    sector_summary: pd.DataFrame,
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Render side-by-side average sector comparisons across three clean subplots."""
    if sector_summary.empty:
        return empty_figure("Sector timing data is unavailable for this session.", height=380)

    figure = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.045,
        subplot_titles=("Sector 1", "Sector 2", "Sector 3"),
    )

    sectors = [("AvgSector1", 1), ("AvgSector2", 2), ("AvgSector3", 3)]
    ordered = sector_summary.sort_values("AvgSector1")
    for column, subplot_column in sectors:
        figure.add_trace(
            go.Bar(
                x=ordered[column],
                y=ordered["Driver"],
                orientation="h",
                marker=dict(
                    color=[driver_colors.get(driver, "#E10600") for driver in ordered["Driver"]],
                    line=dict(color="#1C1C1C", width=1),
                ),
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>%{x:.3f}s<extra></extra>",
            ),
            row=1,
            col=subplot_column,
        )

    figure.update_xaxes(title_text="Seconds", row=1, col=1)
    figure.update_xaxes(title_text="Seconds", row=1, col=2)
    figure.update_xaxes(title_text="Seconds", row=1, col=3)

    # Style subplot title annotations so they are clearly legible
    for annotation in figure["layout"]["annotations"]:
        annotation["font"] = dict(color=TEXT_COLOR, size=12, family=FONT_FAMILY)

    return apply_dark_theme(
        figure,
        title=title,
        height=420,
        show_legend=False,
        margin=dict(l=45, r=20, t=35 if not title else 55, b=40),
    )


def create_consistency_scatter(
    summary: pd.DataFrame,
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Plot average pace versus consistency for the selected drivers."""
    if summary.empty:
        return empty_figure("Consistency metrics need valid racing laps.", height=360)

    fig = go.Figure()
    for _, row in summary.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["AverageLapSeconds"]],
                y=[row["ConsistencyPct"]],
                mode="markers+text",
                name=row["Driver"],
                marker=dict(
                    size=max(14, min(28, row["Laps"] * 1.2)),
                    color=driver_colors.get(row["Driver"], "#E10600"),
                    line=dict(color="#3A3A3A", width=1.5),
                ),
                text=[row["Driver"]],
                textposition="top center",
                textfont=dict(color=TEXT_COLOR, size=11, family=FONT_FAMILY),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Average lap {row['AverageLap']}<br>"
                    f"Consistency {row['Consistency']}<br>"
                    f"Laps {int(row['Laps'])}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(xaxis_title="Average Lap (s)", yaxis_title="Consistency (%)")
    return apply_dark_theme(
        fig,
        title=title,
        height=410,
        show_legend=False,
        margin=dict(l=50, r=25, t=25 if not title else 50, b=40),
    )


def create_driver_heatmap(
    heatmap: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Render a driver performance heatmap using lap delta to median pace with dark-grey palette."""
    if heatmap.empty:
        return empty_figure("Heatmap data is unavailable for this session.", height=360)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns,
            y=heatmap.index,
            colorscale=[
                [0.0, "#3B82F6"],   # Quicker than median (blue)
                [0.5, "#242424"],   # Median pace (neutral dark-grey surface)
                [1.0, "#E10600"],   # Slower than median (F1 red)
            ],
            colorbar=dict(
                title=dict(text="Delta (s)", font=dict(color=MUTED_TEXT, size=11, family=FONT_FAMILY)),
                tickfont=dict(color=MUTED_TEXT, size=10, family=FONT_FAMILY),
                thickness=14,
                len=0.85,
                outlinewidth=0,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Lap %{x}<br>"
                "Delta to median %{z:.3f}s<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis_title="Lap Number", yaxis_title="")
    return apply_dark_theme(
        fig,
        title=title,
        height=420,
        show_legend=False,
        margin=dict(l=55, r=75, t=25 if not title else 50, b=40),
    )
