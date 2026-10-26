"""FastF1 session loading, caching, and metadata helpers."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import streamlit as st

from .constants import CACHE_DIR, FALLBACK_DRIVER_COLORS, MIN_SEASON_YEAR

_CACHE_READY = False


def _frame_column(frame: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    """Return a dataframe column or a same-length fallback series."""
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def setup_fastf1_cache() -> None:
    """Enable local FastF1 caching once per process."""
    global _CACHE_READY
    if _CACHE_READY:
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    ff1.Cache.enable_cache(CACHE_DIR)
    _CACHE_READY = True


def get_season_options() -> list[int]:
    """Return recent season years for the sidebar selector."""
    current_year = datetime.now().year
    return list(range(current_year, MIN_SEASON_YEAR - 1, -1))


def _event_value(event: Any, key: str, default: str = "") -> str:
    """Safely read values from FastF1 event objects or dict-like rows."""
    try:
        value = event[key]
    except Exception:
        value = getattr(event, key, default)
    return default if value is None else str(value)


def _normalize_schedule(schedule: Any) -> pd.DataFrame:
    """Convert a FastF1 event schedule into a clean dataframe."""
    schedule_df = pd.DataFrame(schedule).copy()
    if "EventFormat" in schedule_df.columns:
        mask = ~schedule_df["EventFormat"].astype(str).str.contains("testing", case=False, na=False)
        schedule_df = schedule_df.loc[mask]

    sort_columns = [column for column in ("RoundNumber", "EventDate") if column in schedule_df.columns]
    if sort_columns:
        schedule_df = schedule_df.sort_values(sort_columns)

    if "EventName" not in schedule_df.columns:
        schedule_df["EventName"] = schedule_df.get("Location", schedule_df.index.astype(str))

    return schedule_df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_event_schedule(year: int) -> pd.DataFrame:
    """Fetch and cache the official event schedule for a season."""
    setup_fastf1_cache()
    return _normalize_schedule(ff1.get_event_schedule(year))


@st.cache_resource(show_spinner=False)
def load_session(year: int, grand_prix: str, session_code: str):
    """Load a FastF1 session with laps, telemetry, weather, and messages."""
    setup_fastf1_cache()
    session = ff1.get_session(year, grand_prix, session_code)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    return session


def get_session_metadata(session: Any) -> dict[str, str]:
    """Build a concise metadata block for the current session."""
    event = session.event
    event_name = _event_value(event, "EventName", "Unknown Grand Prix")
    country = _event_value(event, "Country", "")
    location = _event_value(event, "Location", "")
    round_number = _event_value(event, "RoundNumber", "")
    event_date = getattr(session, "date", None)
    date_label = ""
    if event_date is not None:
        try:
            date_label = pd.Timestamp(event_date).strftime("%d %b %Y")
        except Exception:
            date_label = str(event_date)

    return {
        "event_name": event_name,
        "country": country,
        "location": location,
        "round": round_number,
        "date": date_label,
        "session_name": str(getattr(session, "name", "")),
        "track": session.session_info.get("Meeting", {}).get("Circuit", {}).get("ShortName", ""),
    }


def get_event_sessions(schedule: pd.DataFrame, grand_prix: str) -> list[str]:
    """Return the official list of session names for the selected Grand Prix."""
    fallback_sessions = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
    if schedule.empty or "EventName" not in schedule.columns:
        return fallback_sessions

    matching = schedule.loc[schedule["EventName"].astype(str) == str(grand_prix)]
    if matching.empty:
        return fallback_sessions

    row = matching.iloc[0]
    sessions = []
    for index in range(1, 6):
        session_name = row.get(f"Session{index}")
        if pd.notna(session_name) and str(session_name).strip():
            sessions.append(str(session_name).strip())

    return sessions or fallback_sessions


def get_driver_roster(session: Any) -> pd.DataFrame:
    """Return driver identity information for the loaded session."""
    try:
        results = pd.DataFrame(session.results).copy()
    except Exception:
        results = pd.DataFrame()

    if results.empty:
        if hasattr(session, "laps") and not session.laps.empty and "Driver" in session.laps.columns:
            drivers = sorted(pd.Series(session.laps["Driver"]).dropna().astype(str).unique().tolist())
        else:
            drivers = []
        return pd.DataFrame({"Driver": drivers, "Label": drivers, "Team": ["" for _ in drivers], "FullName": drivers})

    results["Driver"] = _frame_column(results, "Abbreviation", "").astype(str)
    broadcast_name = _frame_column(results, "BroadcastName", "").astype(str)
    results["Driver"] = results["Driver"].replace("", pd.NA).fillna(broadcast_name).astype(str)
    full_name = _frame_column(results, "FullName", "").astype(str)
    results["FullName"] = full_name.replace("", pd.NA).fillna(broadcast_name).fillna(results["Driver"]).astype(str)
    results["Team"] = _frame_column(results, "TeamName", "").astype(str)
    results["Label"] = results.apply(
        lambda row: f"{row['Driver']} | {row['FullName']} ({row['Team']})".strip(),
        axis=1,
    )
    return results[["Driver", "FullName", "Team", "Label"]].drop_duplicates("Driver")


def get_driver_color_map(session: Any, drivers: list[str]) -> dict[str, str]:
    """Resolve FastF1 driver colors with resilient fallbacks."""
    color_map: dict[str, str] = {}
    for index, driver in enumerate(drivers):
        try:
            color_map[driver] = plotting.get_driver_color(driver, session, colormap="official")
        except Exception:
            color_map[driver] = FALLBACK_DRIVER_COLORS[index % len(FALLBACK_DRIVER_COLORS)]
    return color_map


def build_sidebar_caption(metadata: dict[str, str]) -> str:
    """Create a small sidebar summary sentence for the active session."""
    parts = [metadata["event_name"], metadata["session_name"]]
    if metadata["date"]:
        parts.append(metadata["date"])
    return " | ".join(part for part in parts if part)
