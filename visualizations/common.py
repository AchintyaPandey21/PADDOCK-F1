"""Shared Plotly styling helpers."""

from __future__ import annotations

from typing import Any
import plotly.graph_objects as go

from utils.constants import DASH_PATTERNS, MARKER_SYMBOLS

# Dark-grey palette constants matching theme.css
PAPER_BG = "rgba(0, 0, 0, 0)"
PLOT_BG = "rgba(0, 0, 0, 0)"
CARD_BG = "#242424"
GRID_COLOR = "#333333"
BORDER_COLOR = "#3A3A3A"
TEXT_COLOR = "#F5F5F5"
MUTED_TEXT = "#B3B3B3"
ACCENT_COLOR = "#E10600"
ACCENT_HOVER = "#FF2A2A"
FONT_FAMILY = "Space Grotesk, Segoe UI, -apple-system, sans-serif"
MONO_FONT = "JetBrains Mono, monospace"


def apply_dark_theme(
    fig: go.Figure,
    title: str | None = None,
    height: int = 420,
    show_legend: bool = True,
    legend_orientation: str = "h",
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Apply a consistent dark-grey theme across dashboard charts.

    Avoids overlapping headings and legends by placing legends outside the plot area
    or top-right with appropriate margin separation.
    """
    if margin is None:
        if legend_orientation == "v":
            top_m = 40 if title else 25
            right_m = 95
            bottom_m = 40
            left_m = 50
        elif title:
            top_m = 50
            right_m = 25
            bottom_m = 40
            left_m = 50
        else:
            top_m = 35
            right_m = 25
            bottom_m = 40
            left_m = 50
        margin = dict(l=left_m, r=right_m, t=top_m, b=bottom_m)

    if legend_orientation == "v":
        legend_layout: dict[str, Any] = dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=MUTED_TEXT, size=10, family=FONT_FAMILY),
        )
    else:
        legend_layout = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=MUTED_TEXT, size=11, family=FONT_FAMILY),
        )

    title_layout = None
    if title:
        title_layout = dict(
            text=title,
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(color=TEXT_COLOR, size=14, family=FONT_FAMILY),
        )

    fig.update_layout(
        title=title_layout,
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=margin,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
        hoverlabel=dict(
            bgcolor="#242424",
            bordercolor="#3A3A3A",
            font=dict(color=TEXT_COLOR, family=MONO_FONT, size=12),
        ),
        showlegend=show_legend,
        legend=legend_layout,
    )

    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=BORDER_COLOR,
        zeroline=False,
        showline=False,
        tickfont=dict(color=MUTED_TEXT, size=11, family=FONT_FAMILY),
        title_font=dict(color=MUTED_TEXT, size=12, family=FONT_FAMILY),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=BORDER_COLOR,
        zeroline=False,
        showline=False,
        tickfont=dict(color=MUTED_TEXT, size=11, family=FONT_FAMILY),
        title_font=dict(color=MUTED_TEXT, size=12, family=FONT_FAMILY),
        automargin=True,
    )
    return fig


def empty_figure(message: str, height: int = 320) -> go.Figure:
    """Return a branded placeholder figure when data is unavailable."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color=MUTED_TEXT, family=FONT_FAMILY),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_dark_theme(fig, height=height, show_legend=False)


def driver_line_style(index: int) -> tuple[str, str]:
    """Return a dash pattern and marker symbol for a driver index."""
    dash = DASH_PATTERNS[index % len(DASH_PATTERNS)]
    symbol = MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)]
    return dash, symbol
