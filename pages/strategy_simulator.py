"""ML-powered what-if tyre strategy simulator page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ml.data import VALID_COMPOUNDS
from ml.inference import is_model_available, load_trained_pipeline
from ml.simulator import extract_actual_driver_strategy, get_recorded_baseline, run_strategy_simulation
from utils.helpers import render_download, render_page_header, render_section_title
from utils.time_format import format_delta, format_duration, format_lap_time
from visualizations.strategy import (
    create_recorded_degradation_chart,
    create_recorded_lap_time_chart,
    create_simulation_degradation_chart,
    create_simulation_delta_chart,
    create_simulation_strategy_chart,
    create_simulation_time_chart,
)


def _scenario_controls(driver: str, total_laps: int, actual: dict[str, object]) -> tuple[list[int], list[str]]:
    """Render ordered pit and compound controls for a bounded what-if plan."""
    actual_pits = [int(lap) for lap in actual["actual_pit_laps"] if 0 < int(lap) < total_laps]
    max_stops = min(5, total_laps - 1)
    default_stops = min(len(actual_pits), max_stops)
    stop_count = st.slider("Pit stops", min_value=0, max_value=max_stops, value=default_stops, key=f"sim_stops_{driver}")
    pit_laps: list[int] = []
    compounds: list[str] = []
    previous_lap = 0
    stop_labels = ["First", "Second", "Third", "Fourth", "Fifth"]

    for stint_index in range(stop_count + 1):
        left, right = st.columns((1, 1))
        with right:
            default_compounds = list(actual["actual_compounds"])
            default_compound = str(default_compounds[stint_index]).upper() if stint_index < len(default_compounds) else "MEDIUM"
            compound_index = VALID_COMPOUNDS.index(default_compound) if default_compound in VALID_COMPOUNDS else VALID_COMPOUNDS.index("MEDIUM")
            compounds.append(
                st.selectbox(
                    f"Stint {stint_index + 1} compound",
                    VALID_COMPOUNDS,
                    index=compound_index,
                    key=f"sim_compound_{driver}_{stint_index}",
                )
            )
        if stint_index < stop_count:
            with left:
                min_lap = previous_lap + 1
                max_lap = total_laps - (stop_count - stint_index)
                recorded_pit = actual_pits[stint_index] if stint_index < len(actual_pits) else round(total_laps * (stint_index + 1) / (stop_count + 1))
                default_lap = max(min_lap, min(max_lap, recorded_pit))
                pit_lap = st.number_input(
                    f"{stop_labels[stint_index]} pit stop on lap",
                    min_value=min_lap,
                    max_value=max_lap,
                    value=default_lap,
                    step=1,
                    key=f"sim_pit_{driver}_{stint_index}",
                )
                pit_laps.append(int(pit_lap))
                previous_lap = int(pit_lap)
        else:
            with left:
                st.caption(f"Stint {stint_index + 1}: laps {previous_lap + 1}-{total_laps}")

    return pit_laps, compounds


def _recorded_stint_summary(driver_laps: pd.DataFrame) -> pd.DataFrame:
    """Build a display-only recorded stint summary from stored lap timing."""
    rows: list[dict[str, object]] = []
    for stint, group in driver_laps.groupby("Stint", dropna=False, sort=True):
        valid_times = pd.to_numeric(group["LapTimeSeconds"], errors="coerce")
        valid_times = valid_times[valid_times > 0]
        rows.append(
            {
                "Stint": int(stint) if pd.notna(stint) else "N/A",
                "Compound": str(group["Compound"].iloc[0]).title(),
                "Lap range": f"{int(group['LapNumber'].min())}-{int(group['LapNumber'].max())}",
                "Length": f"{int(group['LapNumber'].max() - group['LapNumber'].min() + 1)} laps",
                "Average lap": format_lap_time(valid_times.mean() if not valid_times.empty else None),
            }
        )
    return pd.DataFrame(rows)


def _strategy_label(stints: list[dict[str, object]]) -> str:
    """Return a compact, comparable tyre-plan label for the results table."""
    return " → ".join(f"{stint['compound']} {stint['stint_length']}L" for stint in stints)


def _comparison_table(result) -> pd.DataFrame:
    """Build the aligned Actual-versus-What-if results table for Streamlit."""
    summary = result.summary
    actual_total = float(summary["actual_total_time"])
    simulated_total = float(summary["simulated_total_time"])
    actual_average = float(summary["actual_avg_lap"])
    simulated_average = float(summary["simulated_avg_lap"])
    actual_stops = int(summary["actual_pit_stops"])
    simulated_stops = int(summary["simulated_pit_stops"])
    actual_laps = int(summary["comparison_lap_count"])
    simulated_laps = int(summary["comparison_lap_count"])
    strategy_difference = "Recorded-plan replay" if summary["matches_recorded_plan"] else "Strategy changed"

    return pd.DataFrame(
        [
            ["Total race time", format_duration(actual_total), format_duration(simulated_total), format_delta(actual_total - simulated_total)],
            ["Average lap time", format_lap_time(actual_average), format_lap_time(simulated_average), format_delta(actual_average - simulated_average)],
            ["Completed laps", f"{actual_laps} valid", f"{simulated_laps} valid", "0 laps"],
            ["Pit stops", str(actual_stops), str(simulated_stops), f"{actual_stops - simulated_stops:+d} stops"],
            [
                "Pit-stop loss",
                f"Observed transition: {format_duration(summary['actual_pit_loss'])}",
                f"Modelled transition: {format_duration(summary['simulated_pit_loss'])}",
                "Not directly comparable",
            ],
            ["Tyre strategy", _strategy_label(result.stints_actual), _strategy_label(result.stints_simulated), strategy_difference],
        ],
        columns=["Metric", "Actual (Recorded)", "What-if (Simulated)", "Difference"],
    )


def render(context: dict[str, object]) -> None:
    """Render the database-backed what-if strategy workspace."""
    metadata = context["metadata"]
    dataset = context["dataset"]
    session_name = str(metadata.get("session_name", ""))

    render_page_header(
        "What-if Strategy Simulator",
        "",
        kicker="Strategy Lab",
    )

    if session_name != "Race":
        st.info("The simulator currently supports Race sessions because pit strategy and lap-by-lap position data are required.")
        return
    if not is_model_available():
        st.warning("No trained strategy model is available. Run `python -m ml.train` after loading historical race sessions into PostgreSQL.")
        return
    if dataset["laps"].empty:
        st.warning("No lap data is available for this race.")
        return

    drivers = dataset["drivers"]
    if not drivers:
        st.warning("No drivers are available for simulation.")
        return

    driver = st.selectbox(
        "Driver to simulate",
        options=drivers,
        format_func=lambda code: context["driver_labels"].get(code, code),
        key="sim_driver",
    )
    actual = extract_actual_driver_strategy(dataset, driver)
    baseline = get_recorded_baseline(dataset, driver)

    render_section_title("Actual Race Baseline")
    baseline_time, baseline_pace, baseline_laps, baseline_stops = st.columns(4)
    baseline_time.metric("Recorded elapsed time", format_duration(baseline["total_time"]))
    baseline_pace.metric("Recorded average lap", format_lap_time(baseline["average_lap"]))
    baseline_laps.metric("Completed valid laps", f"{baseline['valid_laps']} / {baseline['total_laps']}")
    baseline_stops.metric("Recorded pit stops", baseline["pit_stops"])
    st.caption("Recorded tyre plan: " + " / ".join(f"{stint['compound']} {stint['stint_length']}L" for stint in baseline["stints"]))
    if baseline["valid_laps"] < baseline["total_laps"]:
        st.info("Some recorded lap timings are unavailable; baseline totals and any comparison use valid recorded laps only.")
    baseline_left, baseline_right = st.columns(2)
    with baseline_left:
        st.plotly_chart(create_recorded_lap_time_chart(baseline["driver_laps"]), width="stretch", key="recorded_lap_time_chart")
    with baseline_right:
        st.plotly_chart(create_recorded_degradation_chart(baseline["driver_laps"]), width="stretch", key="recorded_degradation_chart")
    render_section_title("Recorded Stint Summary")
    stint_summary = _recorded_stint_summary(baseline["driver_laps"])
    if stint_summary.empty:
        st.info("Recorded stint details are unavailable for this driver.")
    else:
        st.dataframe(stint_summary, width="stretch", hide_index=True, key="recorded_stint_summary")

    with st.expander("Scenario inputs", expanded=True):
        st.caption(
            f"{driver} completed {actual['total_laps']} laps with {len(actual['actual_pit_laps'])} recorded stop(s). "
            "Choose the lap after which each new tyre stint starts."
        )
        pit_laps, compounds = _scenario_controls(driver, actual["total_laps"], actual)
        run_simulation = st.button("Run what-if simulation", type="primary", key="run_strategy_simulation")

    scenario_key = (context["selection"], driver, tuple(pit_laps), tuple(compounds))
    if run_simulation:
        with st.spinner("Estimating stint pace and equivalent-lap race time..."):
            try:
                result = run_strategy_simulation(
                    dataset=dataset,
                    metadata=metadata,
                    driver=driver,
                    simulated_pit_laps=pit_laps,
                    simulated_compounds=compounds,
                    pipeline=load_trained_pipeline(),
                )
            except ValueError as exc:
                st.error(f"The scenario is invalid: {exc}")
                return
        st.session_state["strategy_simulation_result"] = result
        st.session_state["strategy_simulation_key"] = scenario_key

    if st.session_state.get("strategy_simulation_key") != scenario_key:
        st.info("Set a tyre plan and select **Run what-if simulation** to compare it with the recorded race.")
        return

    result = st.session_state.get("strategy_simulation_result")
    if result is None:
        return

    summary = result.summary
    comparison_scope = (
        "full race" if summary["race_complete"] and summary["comparison_lap_count"] == summary["recorded_lap_count"]
        else f"{summary['comparison_lap_count']} equivalent recorded lap(s) of {summary['recorded_lap_count']}"
    )
    render_section_title("Actual vs What-if")
    st.caption(f"Comparison scope: {comparison_scope}. Difference is Actual minus What-if, so a positive time means the simulated plan is faster.")
    st.dataframe(_comparison_table(result), width="stretch", hide_index=True, key="strategy_comparison_table")
    for warning in summary["warnings"]:
        st.warning(warning)
    if summary["matches_recorded_plan"]:
        st.success("Recorded-plan sanity check: the stored tyre ages and pit model are replayed, so the simulated total should match the recorded equivalent-lap total.")

    render_section_title("Strategy Comparison")
    st.plotly_chart(
        create_simulation_strategy_chart(result.stints_actual, result.stints_simulated, context["session"]),
        width="stretch",
        key="simulation_strategy_chart",
    )

    st.plotly_chart(
        create_simulation_time_chart(result.simulated_laps),
        width="stretch",
        key="simulation_time_chart",
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            create_simulation_degradation_chart(result.simulated_laps),
            width="stretch",
            key="simulation_degradation_chart",
        )
    with chart_right:
        st.plotly_chart(
            create_simulation_delta_chart(result.simulated_laps),
            width="stretch",
            key="simulation_delta_chart",
        )

    render_section_title("Lap-by-Lap Simulation")
    export_columns = [
        "LapNumber", "Stint", "Compound", "TyreLife", "StintLength", "IsPitLap", "ActualLapTime",
        "SimulatedLapTime", "ActualCumulativeTime", "SimulatedCumulativeTime", "ActualPitLoss", "ModelledActualPitLoss", "AssumedPitLoss",
        "LapDelta", "CumulativeDelta", "ExceedsObservedTyreAge",
    ]
    simulation_table = result.simulated_laps[export_columns]
    display_table = simulation_table.copy()
    for column in ["ActualLapTime", "SimulatedLapTime", "ActualPitLoss", "ModelledActualPitLoss", "AssumedPitLoss"]:
        display_table[column] = display_table[column].map(format_lap_time)
    for column in ["ActualCumulativeTime", "SimulatedCumulativeTime"]:
        display_table[column] = display_table[column].map(format_duration)
    for column in ["LapDelta", "CumulativeDelta"]:
        display_table[column] = display_table[column].map(format_delta)
    st.dataframe(display_table, width="stretch", hide_index=True, key="simulation_results_table")
    render_download(
        label="Download what-if simulation CSV",
        df=simulation_table,
        file_name=f"paddock_f1_what_if_{result.event_name.lower().replace(' ', '_')}_{driver}.csv",
        key="simulation_download",
    )

    st.caption(
        "The what-if estimate applies the modelled tyre-plan delta to recorded driver timing and adds data-derived pit losses. It does not model safety cars, traffic, tyre availability, or adaptive rival strategies."
    )
