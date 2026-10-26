"""Telemetry and racing line comparisons for two drivers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import TEXT_COLOR, MUTED_TEXT, FONT_FAMILY, apply_dark_theme, empty_figure


def create_telemetry_overlay(
    bundle: dict[str, object],
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Overlay speed, throttle, brake, gear, and RPM traces with clear separation."""
    merged = pd.DataFrame(bundle["merged"]).copy()
    driver_a = str(bundle["driver_a"])
    driver_b = str(bundle["driver_b"])
    if merged.empty:
        return empty_figure("Telemetry overlays require valid fastest laps for both drivers.", height=520)

    channels = [
        ("Speed", "km/h"),
        ("Throttle", "%"),
        ("Brake", "On/Off"),
        ("nGear", "Gear"),
        ("RPM", "RPM"),
    ]
    subplot_titles = [f"{channel} ({unit})" for channel, unit in channels]
    figure = make_subplots(
        rows=len(channels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=subplot_titles,
    )

    color_a = driver_colors.get(driver_a, "#E10600")
    color_b = driver_colors.get(driver_b, "#3B82F6")

    for row_index, (channel, _) in enumerate(channels, start=1):
        column_a = f"{channel}_{driver_a}"
        column_b = f"{channel}_{driver_b}"
        if column_a not in merged.columns or column_b not in merged.columns:
            continue

        figure.add_trace(
            go.Scatter(
                x=merged["Distance"],
                y=merged[column_a],
                name=driver_a,
                legendgroup=driver_a,
                showlegend=row_index == 1,
                line=dict(color=color_a, width=2.2),
                hovertemplate=f"<b>{driver_a}</b><br>Distance %{{x:.0f}} m<br>{channel} %{{y}}<extra></extra>",
            ),
            row=row_index,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=merged["Distance"],
                y=merged[column_b],
                name=driver_b,
                legendgroup=driver_b,
                showlegend=row_index == 1,
                line=dict(color=color_b, width=2.0, dash="dash"),
                hovertemplate=f"<b>{driver_b}</b><br>Distance %{{x:.0f}} m<br>{channel} %{{y}}<extra></extra>",
            ),
            row=row_index,
            col=1,
        )

    figure.update_xaxes(title_text="Distance (m)", row=len(channels), col=1)

    # Style subplot title annotations for high contrast and readability
    for annotation in figure["layout"]["annotations"]:
        annotation["font"] = dict(color=TEXT_COLOR, size=11, family=FONT_FAMILY)

    return apply_dark_theme(
        figure,
        title=title,
        height=880,
        legend_orientation="h",
        margin=dict(l=60, r=25, t=35 if not title else 55, b=40),
    )


def create_speed_delta_chart(
    bundle: dict[str, object],
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Plot speed delta between the two selected drivers."""
    merged = pd.DataFrame(bundle["merged"]).copy()
    driver_a = str(bundle["driver_a"])
    driver_b = str(bundle["driver_b"])
    speed_a = f"Speed_{driver_a}"
    speed_b = f"Speed_{driver_b}"
    if merged.empty or speed_a not in merged.columns or speed_b not in merged.columns:
        return empty_figure("Speed delta requires telemetry from both drivers.", height=320)

    merged["SpeedDelta"] = merged[speed_a] - merged[speed_b]
    color_a = driver_colors.get(driver_a, "#E10600")

    fig = go.Figure(
        go.Scatter(
            x=merged["Distance"],
            y=merged["SpeedDelta"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(225, 6, 0, 0.12)",
            line=dict(color=color_a, width=2.4),
            hovertemplate=(
                f"<b>{driver_a} - {driver_b}</b><br>"
                "Distance %{x:.0f} m<br>"
                "Speed delta %{y:.1f} km/h<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis_title="Distance (m)", yaxis_title="Delta Speed (km/h)")
    return apply_dark_theme(
        fig,
        title=title,
        height=320,
        show_legend=False,
        margin=dict(l=55, r=20, t=25 if not title else 50, b=40),
    )


def create_racing_line_chart(
    bundle: dict[str, object],
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Compare racing lines for the fastest lap of each driver with equal scaling."""
    position_a = pd.DataFrame(bundle["position_a"]).copy()
    position_b = pd.DataFrame(bundle["position_b"]).copy()
    driver_a = str(bundle["driver_a"])
    driver_b = str(bundle["driver_b"])

    if position_a.empty or position_b.empty or "X" not in position_a.columns or "Y" not in position_a.columns:
        return empty_figure("Racing line data is unavailable for one or both laps.", height=520)

    color_a = driver_colors.get(driver_a, "#E10600")
    color_b = driver_colors.get(driver_b, "#3B82F6")
    if str(color_a).lower() == str(color_b).lower():
        color_b = "#FFC857"

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=position_a["X"],
            y=position_a["Y"],
            mode="lines",
            name=driver_a,
            line=dict(color=color_a, width=2.8),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=position_b["X"],
            y=position_b["Y"],
            mode="lines",
            name=driver_b,
            line=dict(color=color_b, width=2.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis_title="Track X",
        yaxis_title="Track Y",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )
    return apply_dark_theme(
        figure,
        title=title,
        height=560,
        legend_orientation="h",
        margin=dict(l=40, r=20, t=35 if not title else 55, b=40),
    )
