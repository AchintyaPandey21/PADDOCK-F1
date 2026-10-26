"""Driver-centric analytics page."""

from __future__ import annotations

import streamlit as st

from utils.analytics import get_driver_lap_table
from utils.helpers import render_download, render_insight_strip, render_page_header, render_section_title
from visualizations.performance import (
    create_consistency_scatter,
    create_lap_time_progression,
    create_sector_comparison,
)


def render(context: dict[str, object]) -> None:
    """Render the driver analytics page."""
    metadata = context["metadata"]
    dataset = context["dataset"]
    summary = dataset["driver_summary"]
    driver_colors = context["driver_colors"]
    label_map = context["driver_labels"]

    render_page_header(
        "Driver Analytics",
        f"Dive into individual pace, sector execution, compounds, and lap-by-lap consistency for the {metadata['event_name']} {metadata['session_name']}.",
        kicker="Performance Breakdown",
    )

    drivers = dataset["drivers"]
    if not drivers:
        st.warning("No driver data is available for this session yet.")
        return

    default_focus = drivers[0]
    default_compare = drivers[: min(6, len(drivers))]

    control_col1, control_col2 = st.columns((0.9, 1.3))
    with control_col1:
        focus_driver = st.selectbox(
            "Focus Driver",
            options=drivers,
            format_func=lambda driver: label_map.get(driver, driver),
        )
    with control_col2:
        comparison_drivers = st.multiselect(
            "Comparison Group",
            options=drivers,
            default=default_compare,
            format_func=lambda driver: label_map.get(driver, driver),
        )

    if focus_driver not in comparison_drivers:
        comparison_drivers = [focus_driver, *comparison_drivers]

    filtered_summary = summary.loc[summary["Driver"].isin(comparison_drivers)].copy()
    filtered_sector = dataset["sector_summary"].loc[
        dataset["sector_summary"]["Driver"].isin(comparison_drivers)
    ].copy()
    focus_matches = summary.loc[summary["Driver"] == focus_driver]
    if not focus_matches.empty:
        focus_row = focus_matches.iloc[0]
    else:
        focus_row = pd.Series(
            {
                "FastestLap": "N/A",
                "AverageLap": "N/A",
                "Consistency": "N/A",
                "GapToBest": "N/A",
                "Laps": 0,
                "Stints": 0,
            }
        )
    focus_stints = dataset["stints"].loc[dataset["stints"]["Driver"] == focus_driver].copy()

    render_insight_strip(
        [
            ("Focus Driver", focus_driver),
            ("Fastest Lap", str(focus_row.get("FastestLap", "N/A"))),
            ("Average Lap", str(focus_row.get("AverageLap", "N/A"))),
            ("Consistency", str(focus_row.get("Consistency", "N/A"))),
        ]
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Gap To Best", str(focus_row.get("GapToBest", "N/A")))
    with metric_col2:
        st.metric("Laps Analysed", int(focus_row.get("Laps", 0)))
    with metric_col3:
        st.metric("Stints", int(focus_row.get("Stints", 0)))
    with metric_col4:
        longest_stint = int(focus_stints["StintLength"].max()) if not focus_stints.empty else 0
        st.metric("Longest Stint", f"{longest_stint} laps")

    chart_col1, chart_col2 = st.columns((1.2, 1))
    with chart_col1:
        render_section_title("Lap Pace Trace")
        st.plotly_chart(
            create_lap_time_progression(dataset["pace_laps"], comparison_drivers, driver_colors),
            width="stretch",
            key="driver_pace_progression",
        )
    with chart_col2:
        render_section_title("Pace vs Consistency")
        st.plotly_chart(
            create_consistency_scatter(filtered_summary, driver_colors),
            width="stretch",
            key="driver_consistency_scatter",
        )

    render_section_title("Sector Execution")
    st.plotly_chart(
        create_sector_comparison(filtered_sector, driver_colors),
        width="stretch",
        key="driver_sector_comparison",
    )

    table_col1, table_col2 = st.columns((1.05, 0.95))
    with table_col1:
        render_section_title("Comparison Table")
        desired_columns = [
            "Driver", "Team", "FastestLap", "GapToBest", "AverageLap", "Consistency", "Compounds"
        ]
        available_columns = [c for c in desired_columns if c in filtered_summary.columns]
        comparison_table = filtered_summary[available_columns].copy() if available_columns else filtered_summary.copy()
        st.dataframe(comparison_table, width="stretch", hide_index=True, key="driver_comparison_table")
        render_download(
            label="Download driver comparison CSV",
            df=comparison_table,
            file_name="paddock_f1_driver_comparison.csv",
            key="driver_compare_download",
        )
    with table_col2:
        render_section_title("Lap-by-Lap Report")
        lap_table = get_driver_lap_table(dataset["laps"], focus_driver)
        st.dataframe(lap_table, width="stretch", hide_index=True, key="driver_lap_table")
        render_download(
            label=f"Download {focus_driver} lap report",
            df=lap_table,
            file_name=f"paddock_f1_{focus_driver.lower()}_laps.csv",
            key="driver_laps_download",
        )
