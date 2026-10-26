"""Telemetry comparison page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.analytics import build_telemetry_bundle, format_delta, format_lap_time
from utils.helpers import render_download, render_insight_strip, render_page_header, render_section_title
from visualizations.telemetry import (
    create_racing_line_chart,
    create_speed_delta_chart,
    create_telemetry_overlay,
)


def render(context: dict[str, object]) -> None:
    """Render the telemetry lab page."""
    metadata = context["metadata"]
    dataset = context["dataset"]
    drivers = dataset["drivers"]
    label_map = context["driver_labels"]
    driver_colors = context["driver_colors"]

    render_page_header(
        "Telemetry Lab",
        f"Overlay the fastest laps from two drivers to inspect speed traces, control inputs, gears, RPM, and racing line differences in the {metadata['event_name']} {metadata['session_name']}.",
        kicker="Telemetry Comparison",
    )

    if len(drivers) < 2:
        st.warning("Telemetry comparison requires at least two drivers with valid lap data.")
        return

    default_a = drivers[0]
    default_b = drivers[1] if len(drivers) > 1 else drivers[0]
    control_col1, control_col2 = st.columns(2)
    with control_col1:
        driver_a = st.selectbox(
            "Driver A",
            options=drivers,
            index=drivers.index(default_a),
            format_func=lambda driver: label_map.get(driver, driver),
            key="telemetry_driver_a",
        )
    with control_col2:
        alternative_drivers = [driver for driver in drivers if driver != driver_a] or drivers
        default_b_index = alternative_drivers.index(default_b) if default_b in alternative_drivers else 0
        driver_b = st.selectbox(
            "Driver B",
            options=alternative_drivers,
            index=default_b_index,
            format_func=lambda driver: label_map.get(driver, driver),
            key=f"telemetry_driver_b_{driver_a}",
        )

    # Ensure contrasting colors when comparing teammates with identical team colors
    comparison_colors = dict(driver_colors)
    color_a = comparison_colors.get(driver_a, "#E10600")
    color_b = comparison_colors.get(driver_b, "#3B82F6")
    if str(color_a).lower() == str(color_b).lower():
        comparison_colors[driver_b] = "#3B82F6" if str(color_a).lower() != "#3b82f6" else "#FFC857"

    session = context.get("session")
    if session is None:
        with st.spinner(f"Loading detailed telemetry from FastF1 for {metadata['event_name']}..."):
            selection = context.get("selection", {})
            from utils.session import load_session
            session = load_session(
                selection.get("year", 2026),
                selection.get("grand_prix", metadata.get("event_name")),
                selection.get("session_code", metadata.get("session_name")),
            )
            context["session"] = session

    with st.spinner(f"Building telemetry comparison for {driver_a} vs {driver_b}..."):
        try:
            bundle = build_telemetry_bundle(session, driver_a, driver_b)
        except Exception as exc:
            st.error(f"Telemetry comparison is unavailable: {exc}")
            return

    merged = pd.DataFrame(bundle["merged"]).copy()
    lap_time_a = format_lap_time(bundle["lap_a"]["LapTime"])
    lap_time_b = format_lap_time(bundle["lap_b"]["LapTime"])

    speed_a = merged[f"Speed_{driver_a}"].max() if f"Speed_{driver_a}" in merged.columns else float("nan")
    speed_b = merged[f"Speed_{driver_b}"].max() if f"Speed_{driver_b}" in merged.columns else float("nan")
    delta_seconds = bundle["lap_a"]["LapTime"].total_seconds() - bundle["lap_b"]["LapTime"].total_seconds()

    render_insight_strip(
        [
            ("Driver A", driver_a),
            ("Driver B", driver_b),
            ("Lap Delta", format_delta(delta_seconds)),
            ("Samples", str(len(merged))),
        ]
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric(f"{driver_a} Fastest", lap_time_a)
    with metric_col2:
        st.metric(f"{driver_b} Fastest", lap_time_b)
    with metric_col3:
        st.metric(f"{driver_a} Top Speed", f"{speed_a:.1f} km/h" if pd.notna(speed_a) else "N/A")
    with metric_col4:
        speed_delta = speed_a - speed_b if pd.notna(speed_a) and pd.notna(speed_b) else float("nan")
        st.metric("Top Speed Delta", f"{speed_delta:+.1f} km/h" if pd.notna(speed_delta) else "N/A")

    render_section_title("Telemetry Overlay")
    st.plotly_chart(
        create_telemetry_overlay(bundle, comparison_colors),
        width="stretch",
        key="telemetry_overlay_chart",
    )

    chart_col1, chart_col2 = st.columns((0.85, 1.15))
    with chart_col1:
        render_section_title("Speed Delta")
        st.plotly_chart(
            create_speed_delta_chart(bundle, comparison_colors),
            width="stretch",
            key="telemetry_speed_delta_chart",
        )
    with chart_col2:
        render_section_title("Racing Line")
        st.plotly_chart(
            create_racing_line_chart(bundle, comparison_colors),
            width="stretch",
            key="telemetry_racing_line_chart",
        )

    preview_columns = [column for column in merged.columns if column != "index"]
    telemetry_preview = merged[preview_columns].round(3)
    render_section_title("Telemetry Export")
    st.dataframe(telemetry_preview.head(200), width="stretch", hide_index=True, key="telemetry_export_table")
    render_download(
        label="Download telemetry CSV",
        df=telemetry_preview,
        file_name=f"paddock_f1_telemetry_{driver_a.lower()}_{driver_b.lower()}.csv",
        key="telemetry_download",
    )
