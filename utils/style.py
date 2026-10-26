"""Styling helpers for the Streamlit interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def apply_global_styles() -> None:
    """Inject the dashboard CSS into the Streamlit app."""
    css_path = Path(__file__).resolve().parents[1] / "assets" / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
