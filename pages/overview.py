"""Race centre overview page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.helpers import render_download, render_insight_strip, render_page_header, render_section_title
from visualizations.performance import (
    create_driver_heatmap,
    create_fastest_lap_comparison,
    create_lap_time_progression,
    create_sector_comparison,
)
from visualizations.strategy import create_position_changes_chart


def render(context: dict[str, object]) -> None:
    """Render the executive overview page."""
    metadata = context["metadata"]
    dataset = context["dataset"]
    driver_colors = context["driver_colors"]
    summary = dataset["driver_summary"]
    heatmap = dataset["heatmap"]
    positions = dataset["positions"]
    sector_summary = dataset["sector_summary"]
    laps = dataset["laps"]
    weather = dataset["weather_summary"]

    title = f"{metadata['event_name']} | {metadata['session_name']}"
    subtitle = (
        f"Round {metadata['round']} • {metadata['location']}, {metadata['country']} • "
        f"{metadata['date'] or 'Date TBD'}"
    )
    render_page_header(title, subtitle, kicker="Race Centre")

    fastest_lap = summary.iloc[0]["FastestLap"] if not summary.empty else "N/A"
    compounds_used = int(dataset["stints"]["Compound"].nunique()) if not dataset["stints"].empty else 0
    render_insight_strip(
        [
            ("Fastest Lap", fastest_lap),
            ("Drivers", str(len(dataset["drivers"]))),
            ("Compounds", str(compounds_used)),
            ("Circuit", metadata["track"] or metadata["location"]),
        ]
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Timed Laps", f"{int(laps['IsTimedLap'].sum())}" if not laps.empty else "0")
    with col2:
        st.metric("Quickest Driver", summary.iloc[0]["Driver"] if not summary.empty else "N/A")
    with col3:
        st.metric("Track Temp", f"{weather.get('track_temp', 'N/A')}°C" if weather else "N/A")
    with col4:
        st.metric("Air Temp", f"{weather.get('air_temp', 'N/A')}°C" if weather else "N/A")

    featured_drivers = summary["Driver"].head(6).tolist() if not summary.empty else dataset["drivers"][:6]
    chart_col1, chart_col2 = st.columns((1.35, 1))
    with chart_col1:
        render_section_title("Pace Evolution")
        st.plotly_chart(
            create_lap_time_progression(dataset["pace_laps"], featured_drivers, driver_colors),
            width="stretch",
            key="overview_pace_progression",
        )
    with chart_col2:
        render_section_title("Fastest Laps")
        st.plotly_chart(
            create_fastest_lap_comparison(summary.head(12), driver_colors),
            width="stretch",
            key="overview_fastest_laps",
        )

    chart_col3, chart_col4 = st.columns((1.2, 1))
    with chart_col3:
        render_section_title("Track Position Story")
        st.plotly_chart(
            create_position_changes_chart(positions, driver_colors),
            width="stretch",
            key="overview_position_changes",
        )
    with chart_col4:
        render_section_title("Driver Heatmap")
        st.plotly_chart(
            create_driver_heatmap(heatmap),
            width="stretch",
            key="overview_driver_heatmap",
        )

    chart_col5, chart_col6 = st.columns((1.15, 0.85))
    with chart_col5:
        render_section_title("Sector Snapshot")
        st.plotly_chart(
            create_sector_comparison(sector_summary.head(12), driver_colors),
            width="stretch",
            key="overview_sector_comparison",
        )
    with chart_col6:
        render_section_title("Leaderboard")
        desired_columns = [
            "Driver", "Team", "FastestLap", "GapToBest", "AverageLap", "Consistency", "Compounds"
        ]
        available_columns = [col for col in desired_columns if col in summary.columns]
        if summary.empty or not available_columns:
            st.info("Leaderboard data will appear once valid lap timing is available.")
        else:
            overview_table = summary[available_columns].copy()
            st.dataframe(overview_table, width="stretch", hide_index=True, key="overview_leaderboard_table")
            render_download(
                label="Download leaderboard CSV",
                df=overview_table,
                file_name="paddock_f1_leaderboard.csv",
                key="overview_download",
            )

    if weather:
        render_section_title("Weather Context")
        weather_frame = pd.DataFrame(
            [
                {
                    "Air Temp (°C)": weather.get("air_temp", "N/A"),
                    "Track Temp (°C)": weather.get("track_temp", "N/A"),
                    "Humidity (%)": weather.get("humidity", "N/A"),
                    "Wind Speed": weather.get("wind_speed", "N/A"),
                    "Rainfall": weather.get("rainfall", "0%"),
                }
            ]
        )
        st.dataframe(weather_frame, width="stretch", hide_index=True, key="overview_weather_table")
