"""Tyre strategy analytics page."""

from __future__ import annotations

import streamlit as st

from utils.helpers import render_download, render_insight_strip, render_page_header, render_section_title
from visualizations.strategy import (
    create_compound_usage_chart,
    create_position_changes_chart,
    create_stint_analysis_chart,
    create_tyre_strategy_chart,
)


def render(context: dict[str, object]) -> None:
    """Render the tyre strategy page."""
    metadata = context["metadata"]
    dataset = context["dataset"]
    driver_colors = context["driver_colors"]
    label_map = context["driver_labels"]
    stints = dataset["stints"]

    render_page_header(
        "Tyre Strategy",
        f"Explore stint windows, tyre compounds, mean pace per run, and position flow for the {metadata['event_name']} {metadata['session_name']}.",
        kicker="Strategy Room",
    )

    if stints.empty:
        st.warning("No tyre strategy data is available for this session.")
        return

    drivers = dataset["drivers"]
    selected_drivers = st.multiselect(
        "Drivers",
        options=drivers,
        default=drivers[: min(10, len(drivers))],
        format_func=lambda driver: label_map.get(driver, driver),
    )
    if not selected_drivers:
        st.info("Select at least one driver to inspect tyre strategy.")
        return

    filtered_stints = stints.loc[stints["Driver"].isin(selected_drivers)].copy()
    filtered_positions = dataset["positions"].loc[dataset["positions"]["Driver"].isin(selected_drivers)].copy()

    longest_stint = int(filtered_stints["StintLength"].max()) if not filtered_stints.empty else 0
    average_stint = filtered_stints["StintLength"].mean() if not filtered_stints.empty else 0
    compounds = filtered_stints["Compound"].nunique()
    render_insight_strip(
        [
            ("Drivers In View", str(len(selected_drivers))),
            ("Total Stints", str(len(filtered_stints))),
            ("Longest Stint", f"{longest_stint} laps"),
            ("Compounds Used", str(compounds)),
        ]
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Average Stint", f"{average_stint:.1f} laps")
    with metric_col2:
        st.metric("Fresh Tyre Starts", int(filtered_stints["FreshTyre"].fillna(False).sum()))
    with metric_col3:
        st.metric("Mean Tyre Life", f"{filtered_stints['MeanTyreLife'].mean():.1f}" if filtered_stints["MeanTyreLife"].notna().any() else "N/A")
    with metric_col4:
        st.metric("Mean Stint Pace", f"{filtered_stints['MeanLapSeconds'].mean():.2f}s" if filtered_stints["MeanLapSeconds"].notna().any() else "N/A")

    render_section_title("Tyre Strategy Timeline")
    st.plotly_chart(
        create_tyre_strategy_chart(filtered_stints, context["session"], selected_drivers),
        width="stretch",
        key="strategy_timeline_chart",
    )

    chart_col1, chart_col2 = st.columns((1.15, 0.85))
    with chart_col1:
        render_section_title("Stint Pace")
        st.plotly_chart(
            create_stint_analysis_chart(filtered_stints, driver_colors),
            width="stretch",
            key="strategy_pace_chart",
        )
    with chart_col2:
        render_section_title("Compound Mix")
        st.plotly_chart(
            create_compound_usage_chart(filtered_stints, context["session"]),
            width="stretch",
            key="strategy_compound_mix_chart",
        )

    render_section_title("Position Changes")
    st.plotly_chart(
        create_position_changes_chart(filtered_positions, driver_colors),
        width="stretch",
        key="strategy_positions_chart",
    )

    desired_columns = [
        "Driver", "Team", "Stint", "Compound", "StartLap", "EndLap", "StintLength", "MeanLap", "FreshTyre"
    ]
    available_columns = [c for c in desired_columns if c in filtered_stints.columns]
    strategy_table = filtered_stints[available_columns].copy() if available_columns else filtered_stints.copy()
    render_section_title("Strategy Table")
    st.dataframe(strategy_table, width="stretch", hide_index=True, key="strategy_summary_table")
    render_download(
        label="Download strategy CSV",
        df=strategy_table,
        file_name="paddock_f1_strategy_report.csv",
        key="strategy_download",
    )
