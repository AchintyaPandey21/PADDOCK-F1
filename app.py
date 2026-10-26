"""Paddock F1 Streamlit dashboard entrypoint."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from pages import driver_analytics, overview, strategy_simulator, telemetry, tyre_strategy
from utils.analytics import build_dashboard_dataset
from utils.constants import SESSION_OPTIONS
from utils.helpers import render_download
from utils.session import (
    build_sidebar_caption,
    get_driver_color_map,
    get_driver_roster,
    get_event_schedule,
    get_event_sessions,
    get_season_options,
    get_session_metadata,
    load_session,
    setup_fastf1_cache,
)
from db.config import validate_db_config
from db.repository import is_session_stored, load_session_data, save_session_data
from utils.style import apply_global_styles

st.set_page_config(
    page_title="Paddock F1",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_default_schedule(seasons: list[int]) -> tuple[int, pd.DataFrame]:
    """Return the newest season with a schedule FastF1 can resolve."""
    for season in seasons:
        try:
            schedule = get_event_schedule(season)
            if not schedule.empty:
                return season, schedule
        except Exception:
            continue
    raise RuntimeError("FastF1 could not load an event schedule for any supported season.")


def resolve_default_event(schedule: pd.DataFrame) -> str:
    """Default to the latest event at or before today when possible."""
    if schedule.empty:
        return ""

    if "EventDate" in schedule.columns:
        event_dates = pd.to_datetime(schedule["EventDate"], errors="coerce")
        completed = schedule.loc[event_dates <= pd.Timestamp(datetime.now())]
        if not completed.empty:
            return str(completed.iloc[-1]["EventName"])

    return str(schedule.iloc[-1]["EventName"])


def build_context(year: int, grand_prix: str, session_code: str) -> dict[str, object]:
    """Load session data by querying PostgreSQL first, or loading via FastF1 and caching to DB."""
    if is_session_stored(year, grand_prix, session_code):
        db_data = load_session_data(year, grand_prix, session_code)
        return {
            "session": None,
            "metadata": db_data["metadata"],
            "dataset": db_data["dataset"],
            "roster": db_data["roster"],
            "driver_colors": db_data["driver_colors"],
            "driver_labels": db_data["driver_labels"],
            "source": "PostgreSQL",
            "selection": {
                "year": year,
                "grand_prix": grand_prix,
                "session_code": session_code,
            },
        }

    session = load_session(year, grand_prix, session_code)
    metadata = get_session_metadata(session)
    dataset = build_dashboard_dataset(session)
    roster = get_driver_roster(session)
    driver_order = dataset["drivers"] or roster["Driver"].tolist()
    driver_colors = get_driver_color_map(session, driver_order)
    driver_labels = dict(zip(roster["Driver"], roster["Label"]))

    save_session_data(
        year=year,
        grand_prix=grand_prix,
        session_name=session_code,
        session_obj=session,
        dataset=dataset,
        metadata=metadata,
        roster=roster,
    )

    return {
        "session": session,
        "metadata": metadata,
        "dataset": dataset,
        "roster": roster,
        "driver_colors": driver_colors,
        "driver_labels": driver_labels,
        "source": "FastF1 (Cached to PostgreSQL)",
        "selection": {
            "year": year,
            "grand_prix": grand_prix,
            "session_code": session_code,
        },
    }


def render_overview_page() -> None:
    """Run the overview page with the shared session context."""
    overview.render(st.session_state["dashboard_context"])


def render_driver_page() -> None:
    """Run the driver analytics page with the shared session context."""
    driver_analytics.render(st.session_state["dashboard_context"])


def render_strategy_page() -> None:
    """Run the strategy page with the shared session context."""
    tyre_strategy.render(st.session_state["dashboard_context"])


def render_telemetry_page() -> None:
    """Run the telemetry page with the shared session context."""
    telemetry.render(st.session_state["dashboard_context"])


def render_strategy_simulator_page() -> None:
    """Run the ML what-if strategy simulator with the shared session context."""
    strategy_simulator.render(st.session_state["dashboard_context"])


def main() -> None:
    """Render the Paddock F1 dashboard."""
    setup_fastf1_cache()
    apply_global_styles()

    try:
        validate_db_config()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    pages = [
        st.Page(render_overview_page, title="Race Centre", icon=":material/speed:", default=True),
        st.Page(render_driver_page, title="Driver Analytics", icon=":material/insights:"),
        st.Page(render_strategy_page, title="Tyre Strategy", icon=":material/tire_repair:"),
        st.Page(render_strategy_simulator_page, title="Strategy Simulator", icon=":material/model_training:"),
        st.Page(render_telemetry_page, title="Telemetry Lab", icon=":material/timeline:"),
    ]
    current_page = st.navigation(pages, position="sidebar", expanded=True)

    seasons = get_season_options()
    requested_year = st.session_state.get("season_year")
    if requested_year in seasons:
        default_year = requested_year
        try:
            default_schedule = get_event_schedule(default_year)
        except Exception as exc:
            st.error(f"Unable to load the {default_year} schedule: {exc}")
            st.stop()
    else:
        try:
            default_year, default_schedule = resolve_default_schedule(seasons)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.25rem 0 0.5rem 0;">
                <div style="font-size: 1.35rem; font-weight: 700; color: #F5F5F5; letter-spacing: -0.02em; display: flex; align-items: center; gap: 0.4rem;">
                    <span style="color: #E10600; font-weight: 800;">///</span> Paddock F1
                </div>
                <div style="color: #B3B3B3; font-size: 0.78rem; margin-top: 0.2rem; line-height: 1.4;">
                    Formula 1 Race Intelligence & Telemetry Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            '<div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #B3B3B3; margin-bottom: 0.4rem;">Select Session</div>',
            unsafe_allow_html=True,
        )

        year = st.selectbox(
            "Season Year",
            options=seasons,
            index=seasons.index(default_year),
            key="season_year",
        )

        if year == default_year:
            schedule = default_schedule
        else:
            try:
                schedule = get_event_schedule(year)
            except Exception as exc:
                st.error(f"Unable to load the {year} schedule: {exc}")
                st.stop()

        if schedule.empty:
            st.error(f"No events were found for the {year} season.")
            st.stop()

        event_names = schedule["EventName"].astype(str).tolist()
        default_event = resolve_default_event(schedule)
        event_index = event_names.index(default_event) if default_event in event_names else 0
        grand_prix = st.selectbox("Grand Prix", options=event_names, index=event_index)

        available_sessions = get_event_sessions(schedule, grand_prix)
        default_session_index = (
            available_sessions.index("Race")
            if "Race" in available_sessions
            else len(available_sessions) - 1
        )
        session_code = st.selectbox(
            "Session Type",
            options=available_sessions,
            index=default_session_index,
        )

    current_session_key = (year, grand_prix, session_code)
    cached_session_key = st.session_state.get("active_session_key")

    if "dashboard_context" not in st.session_state or cached_session_key != current_session_key:
        with st.spinner(f"Loading {year} {grand_prix} {session_code} data..."):
            try:
                context = build_context(year, grand_prix, session_code)
            except Exception as exc:
                st.error(
                    "The selected session could not be loaded. "
                    "Try a completed event or a different session type.\n\n"
                    f"Reason: {exc}"
                )
                st.stop()
            st.session_state["dashboard_context"] = context
            st.session_state["active_session_key"] = current_session_key
    else:
        context = st.session_state["dashboard_context"]

    with st.sidebar:
        st.divider()
        st.markdown(
            '<div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #B3B3B3; margin-bottom: 0.4rem;">Session Overview</div>',
            unsafe_allow_html=True,
        )
        source_label = context.get("source", "PostgreSQL")
        if "PostgreSQL" in source_label:
            st.caption("🟢 **Store:** Local PostgreSQL")
        else:
            st.caption("🔵 **Store:** FastF1 (Cached)")
        st.markdown(
            f'<p class="sidebar-caption" style="margin-top: 0.2rem;">{build_sidebar_caption(context["metadata"])}</p>',
            unsafe_allow_html=True,
        )
        dataset = context["dataset"]
        full_lap_export = dataset["laps"].copy()
        if not full_lap_export.empty:
            render_download(
                label="Download full lap data",
                df=full_lap_export,
                file_name=f"paddock_f1_{year}_{grand_prix.lower().replace(' ', '_')}_{session_code.lower()}_laps.csv",
                key="sidebar_laps_download",
            )

    current_page.run()


if __name__ == "__main__":
    main()
