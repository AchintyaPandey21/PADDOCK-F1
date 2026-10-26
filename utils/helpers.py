"""Reusable UI helpers for the dashboard."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from .constants import APP_SUBTITLE, APP_TITLE


def render_page_header(title: str, subtitle: str, kicker: str = "Track Intelligence") -> None:
    """Render a broadcast-inspired page header."""
    subtitle_markup = f'<p class="hero-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        (
            f'<div class="hero-panel">'
            f'<div class="hero-kicker">{kicker}</div>'
            f'<h1 class="hero-title">{title}</h1>'
            f"{subtitle_markup}"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_app_introduction() -> None:
    """Render the persistent hero section for the app landing area."""
    render_page_header(APP_TITLE, APP_SUBTITLE, kicker="F1 Analytics Platform")


def render_insight_strip(items: Iterable[tuple[str, str]]) -> None:
    """Render a compact strip of highlight metrics beneath a header."""
    pills = "".join(
        (
            f'<div class="insight-pill">'
            f'<span class="insight-label">{label}</span>'
            f'<span class="insight-value">{value}</span>'
            f"</div>"
        )
        for label, value in items
    )
    st.markdown(f'<div class="insight-strip">{pills}</div>', unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    """Render a lightweight section label."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_surface_open() -> None:
    """Open a styled surface wrapper."""
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)


def render_surface_close() -> None:
    """Close a styled surface wrapper."""
    st.markdown("</div>", unsafe_allow_html=True)


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Return a dataframe as UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def render_download(label: str, df: pd.DataFrame, file_name: str, key: str) -> None:
    """Render a CSV download button for a dataframe."""
    st.download_button(
        label=label,
        data=dataframe_to_csv(df),
        file_name=file_name,
        mime="text/csv",
        key=key,
        width="stretch",
    )
