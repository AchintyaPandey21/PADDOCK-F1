"""Data-backed what-if strategy simulation for Paddock F1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.inference import (
    get_circuit_benchmark,
    get_pit_loss_estimate,
    load_metadata,
    load_trained_pipeline,
    predict_laps,
)


@dataclass
class StintPlan:
    """A contiguous simulated tyre run."""

    stint_index: int
    compound: str
    start_lap: int
    end_lap: int
    stint_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """A single consistent actual-versus-what-if simulation result."""

    driver: str
    event_name: str
    team: str
    total_laps: int
    pit_loss: float
    simulated_laps: pd.DataFrame
    summary: dict[str, Any]
    stints_simulated: list[dict[str, Any]]
    stints_actual: list[dict[str, Any]]


def build_stints_from_pit_laps(pit_laps: list[int], compounds: list[str], total_laps: int) -> list[StintPlan]:
    """Build continuous stints and reject impossible pit windows."""
    if total_laps < 1:
        raise ValueError("Total laps must be at least 1.")
    if len(compounds) != len(pit_laps) + 1:
        raise ValueError("The number of compounds must equal pit stops plus one.")

    previous = 0
    for pit_lap in pit_laps:
        if not isinstance(pit_lap, int) or pit_lap <= previous or pit_lap >= total_laps:
            raise ValueError("Pit laps must be strictly increasing and within the race distance.")
        previous = pit_lap

    start_lap = 1
    stints: list[StintPlan] = []
    for index, (end_lap, compound) in enumerate(zip([*pit_laps, total_laps], compounds), start=1):
        stints.append(
            StintPlan(
                stint_index=index,
                compound=str(compound).upper(),
                start_lap=start_lap,
                end_lap=end_lap,
                stint_length=end_lap - start_lap + 1,
            )
        )
        start_lap = end_lap + 1
    return stints


def extract_actual_driver_strategy(dataset: dict[str, Any], driver: str) -> dict[str, Any]:
    """Return the recorded driver strategy without replacing missing timings."""
    laps_df: pd.DataFrame = dataset.get("laps", pd.DataFrame())
    stints_df: pd.DataFrame = dataset.get("stints", pd.DataFrame())
    if laps_df.empty:
        raise ValueError("Session laps dataset is empty.")

    driver_laps = laps_df.loc[laps_df["Driver"] == driver].sort_values("LapNumber").copy()
    if driver_laps.empty:
        raise ValueError(f"No laps found for driver '{driver}'.")
    total_laps = int(driver_laps["LapNumber"].max())
    team = str(driver_laps["Team"].iloc[0]) if "Team" in driver_laps else "Unknown"

    actual_stints: list[dict[str, Any]] = []
    if not stints_df.empty and "Driver" in stints_df:
        for _, row in stints_df.loc[stints_df["Driver"] == driver].sort_values("StartLap").iterrows():
            actual_stints.append(
                {
                    "stint_index": int(row.get("Stint", len(actual_stints) + 1)),
                    "compound": str(row.get("Compound", "UNKNOWN")).upper(),
                    "start_lap": int(row.get("StartLap", 1)),
                    "end_lap": int(row.get("EndLap", total_laps)),
                    "stint_length": int(row.get("StintLength", 1)),
                }
            )
    if not actual_stints:
        for index, (_, group) in enumerate(driver_laps.groupby("Stint", dropna=False), start=1):
            actual_stints.append(
                {
                    "stint_index": index,
                    "compound": str(group["Compound"].iloc[0]).upper(),
                    "start_lap": int(group["LapNumber"].min()),
                    "end_lap": int(group["LapNumber"].max()),
                    "stint_length": int(group["LapNumber"].max() - group["LapNumber"].min() + 1),
                }
            )

    pit_markers = driver_laps.loc[driver_laps["pit_in_time_seconds"].notna(), "LapNumber"].astype(int).tolist()
    pit_laps = pit_markers or [int(stint["end_lap"]) for stint in actual_stints[:-1]]
    return {
        "driver": driver,
        "team": team,
        "total_laps": total_laps,
        "actual_pit_laps": pit_laps,
        "actual_compounds": [stint["compound"] for stint in actual_stints],
        "actual_stints": actual_stints,
        "driver_laps": driver_laps,
    }


def get_recorded_baseline(dataset: dict[str, Any], driver: str) -> dict[str, Any]:
    """Summarize stored race timing before the user configures a scenario."""
    actual = extract_actual_driver_strategy(dataset, driver)
    lap_times = pd.to_numeric(actual["driver_laps"]["LapTimeSeconds"], errors="coerce")
    valid_laps = lap_times[lap_times > 0]
    return {
        "total_laps": actual["total_laps"],
        "valid_laps": int(valid_laps.size),
        "total_time": float(valid_laps.sum()) if not valid_laps.empty else float("nan"),
        "average_lap": float(valid_laps.mean()) if not valid_laps.empty else float("nan"),
        "pit_stops": len(actual["actual_pit_laps"]),
        "stints": actual["actual_stints"],
        "driver_laps": actual["driver_laps"],
    }


def _weather(dataset: dict[str, Any]) -> dict[str, float]:
    """Convert stored weather summary values to model-ready numeric values."""
    summary = dataset.get("weather_summary", {})

    def value(key: str, default: float) -> float:
        try:
            return float(str(summary.get(key, default)).replace("%", ""))
        except (TypeError, ValueError):
            return default

    return {
        "air_temp": value("air_temp", 25.0),
        "track_temp": value("track_temp", 35.0),
        "humidity": value("humidity", 50.0),
        "rainfall": float(value("rainfall", 0.0) > 0),
    }


def _simulated_features(
    stints: list[StintPlan], driver: str, team: str, event_name: str, circuit_ref: float, weather: dict[str, float]
) -> pd.DataFrame:
    """Create simulated features with tyre age reset to one after every pit stop."""
    rows: list[dict[str, Any]] = []
    for stint in stints:
        for tyre_age, lap_number in enumerate(range(stint.start_lap, stint.end_lap + 1), start=1):
            rows.append(
                {
                    "event_name": event_name,
                    "driver": driver,
                    "team": team,
                    "compound": stint.compound,
                    "circuit_ref_time": circuit_ref,
                    "lap_number": lap_number,
                    "tyre_life": float(tyre_age),
                    "stint_length": float(stint.stint_length),
                    "track_temp": weather["track_temp"],
                    "air_temp": weather["air_temp"],
                    "humidity": weather["humidity"],
                    "rainfall": int(weather["rainfall"]),
                    "stint_index": stint.stint_index,
                }
            )
    return pd.DataFrame(rows)


def _actual_features(
    driver_laps: pd.DataFrame, total_laps: int, driver: str, team: str, event_name: str, circuit_ref: float, weather: dict[str, float]
) -> pd.DataFrame:
    """Build baseline features from the recorded compounds, ages, and stint lengths."""
    indexed = driver_laps.set_index("LapNumber").reindex(range(1, total_laps + 1)).copy()
    indexed["Stint"] = indexed["Stint"].ffill().bfill().fillna(1)
    indexed["Compound"] = indexed["Compound"].ffill().bfill().fillna("MEDIUM").astype(str).str.upper()
    indexed["Team"] = indexed.get("Team", pd.Series(team, index=indexed.index)).ffill().bfill().fillna(team)
    rows: list[dict[str, Any]] = []
    for stint_index, (_, group) in enumerate(indexed.groupby("Stint", sort=False, dropna=False), start=1):
        known_age = pd.to_numeric(group["TyreLife"], errors="coerce")
        first_known = known_age.first_valid_index()
        offset = 1.0
        if first_known is not None:
            offset = max(1.0, float(known_age.loc[first_known]) - list(group.index).index(first_known))
        for sequence, (lap_number, row) in enumerate(group.iterrows()):
            tyre_age = known_age.loc[lap_number]
            if pd.isna(tyre_age):
                tyre_age = offset + sequence
            rows.append(
                {
                    "event_name": event_name,
                    "driver": driver,
                    "team": str(row["Team"]),
                    "compound": str(row["Compound"]).upper(),
                    "circuit_ref_time": circuit_ref,
                    "lap_number": int(lap_number),
                    "tyre_life": float(tyre_age),
                    "stint_length": float(len(group)),
                    "track_temp": weather["track_temp"],
                    "air_temp": weather["air_temp"],
                    "humidity": weather["humidity"],
                    "rainfall": int(weather["rainfall"]),
                    "stint_index": stint_index,
                }
            )
    return pd.DataFrame(rows).sort_values("lap_number").reset_index(drop=True)


def _recorded_pit_losses(actual_laps: np.ndarray, raw_model_laps: np.ndarray, pit_laps: set[int]) -> np.ndarray:
    """Infer full recorded pit-transition loss from entry and following out-lap."""
    penalties = np.zeros(len(actual_laps), dtype=float)
    affected_indices = {lap - 1 for lap in pit_laps}
    affected_indices.update(lap for lap in pit_laps if lap < len(actual_laps))
    for lap_number in pit_laps:
        index = lap_number - 1
        if index < 0 or index >= len(actual_laps):
            continue
        transition = [candidate for candidate in (index, index + 1) if candidate < len(actual_laps) and np.isfinite(actual_laps[candidate])]
        if not transition:
            continue
        neighbors = [
            actual_laps[candidate]
            for candidate in range(max(0, index - 3), min(len(actual_laps), index + 5))
            if candidate not in affected_indices and np.isfinite(actual_laps[candidate])
        ]
        baseline = float(np.median(neighbors)) if neighbors else float(np.nanmedian(raw_model_laps[transition]))
        for candidate in transition:
            penalties[candidate] = max(0.0, actual_laps[candidate] - baseline)
    return penalties


def _modelled_pit_losses(
    pit_laps: set[int], total_laps: int, entry_seconds: float, out_lap_seconds: float
) -> np.ndarray:
    """Place one historical pit-transition estimate across entry and out-laps."""
    losses = np.zeros(total_laps, dtype=float)
    for pit_lap in pit_laps:
        index = pit_lap - 1
        if 0 <= index < total_laps:
            losses[index] += entry_seconds
        if 0 <= index + 1 < total_laps:
            losses[index + 1] += out_lap_seconds
    return losses


def _normalised_actual_pit_losses(
    recorded_losses: np.ndarray,
    pit_laps: set[int],
    total_laps: int,
    entry_seconds: float,
    out_lap_seconds: float,
) -> np.ndarray:
    """Allocate each modelled stop across recorded entry/out-lap timing.

    The total comes from historical pit transitions, while the split follows
    the recorded stop where available. That avoids an artificial fast entry
    lap when a counterfactual removes the recorded stop.
    """
    losses = np.zeros(total_laps, dtype=float)
    transition_total = entry_seconds + out_lap_seconds
    fallback = np.array([entry_seconds, out_lap_seconds], dtype=float)
    for pit_lap in pit_laps:
        index = pit_lap - 1
        indices = [candidate for candidate in (index, index + 1) if 0 <= candidate < total_laps]
        if not indices:
            continue
        observed = np.array([recorded_losses[candidate] for candidate in indices], dtype=float)
        if observed.sum() > 0:
            shares = observed / observed.sum()
        else:
            shares = fallback[:len(indices)]
            shares = shares / shares.sum() if shares.sum() > 0 else np.full(len(indices), 1 / len(indices))
        for candidate, share in zip(indices, shares):
            losses[candidate] += transition_total * share
    return losses


def run_strategy_simulation(
    dataset: dict[str, Any], metadata: dict[str, Any], driver: str,
    simulated_pit_laps: list[int], simulated_compounds: list[str], pipeline: Pipeline | None = None,
) -> SimulationResult:
    """Compare a what-if tyre plan with equivalent recorded race laps.

    Recorded timing remains the baseline. The model supplies only the strategy
    pace delta caused by compound, tyre age, stint length, lap, and weather.
    """
    pipeline = pipeline or load_trained_pipeline()
    event_name = str(metadata.get("event_name", ""))
    actual = extract_actual_driver_strategy(dataset, driver)
    total_laps = actual["total_laps"]
    team = actual["team"]
    weather = _weather(dataset)
    circuit_ref = get_circuit_benchmark(event_name)
    sim_stints = build_stints_from_pit_laps(simulated_pit_laps, simulated_compounds, total_laps)
    recorded_plan = (
        simulated_pit_laps == actual["actual_pit_laps"]
        and [stint.compound for stint in sim_stints] == [str(compound).upper() for compound in actual["actual_compounds"]]
    )
    if not recorded_plan and not weather["rainfall"] and any(stint.compound in {"INTERMEDIATE", "WET"} for stint in sim_stints):
        raise ValueError("Intermediate and wet compounds are not feasible because the selected session has no recorded rainfall.")
    observed_ages = load_metadata().get("dataset_audit", {}).get("observed_tyre_age_by_compound", {})
    unsupported_compounds = [stint.compound for stint in sim_stints if int(observed_ages.get(stint.compound, 0)) < 1]
    if unsupported_compounds and not recorded_plan:
        raise ValueError("No historical race-lap training data is available for: " + ", ".join(sorted(set(unsupported_compounds))))

    actual_features = _actual_features(actual["driver_laps"], total_laps, driver, team, event_name, circuit_ref, weather)
    # An unchanged plan is a replay sanity check, so preserve FastF1's stored
    # tyre ages (including used sets) rather than pretending every stop fitted
    # a brand-new tyre. Changed strategies start each simulated stint at age 1.
    sim_features = actual_features.copy() if recorded_plan else _simulated_features(
        sim_stints, driver, team, event_name, circuit_ref, weather
    )
    simulated_raw = predict_laps(pipeline, sim_features)
    actual_raw = predict_laps(pipeline, actual_features)

    actual_lap_times = actual["driver_laps"].set_index("LapNumber")["LapTimeSeconds"].reindex(range(1, total_laps + 1)).to_numpy(dtype=float)
    valid_laps = np.isfinite(actual_lap_times) & (actual_lap_times > 0)
    actual_pit_set = set(actual["actual_pit_laps"])
    simulated_pit_set = set(simulated_pit_laps)
    recorded_pit_loss = _recorded_pit_losses(actual_lap_times, actual_raw, actual_pit_set)

    pit_estimate = get_pit_loss_estimate(event_name)
    assumed_pit_loss = float(pit_estimate["seconds"])
    entry_pit_loss = float(pit_estimate.get("entry_lap_seconds", assumed_pit_loss))
    out_pit_loss = float(pit_estimate.get("out_lap_seconds", 0.0))
    raw_strategy_delta = simulated_raw - actual_raw
    assumed_pit_losses = _modelled_pit_losses(simulated_pit_set, total_laps, entry_pit_loss, out_pit_loss)
    modelled_actual_pit_losses = _normalised_actual_pit_losses(
        recorded_pit_loss, actual_pit_set, total_laps, entry_pit_loss, out_pit_loss
    )
    # Use the same historical transition estimate on both plans. This prevents
    # a safety-car or timing anomaly on a recorded pit lap becoming a false
    # counterfactual advantage for a different stop count.
    pit_delta = assumed_pit_losses - modelled_actual_pit_losses
    simulated_lap_times = actual_lap_times + raw_strategy_delta + pit_delta
    simulated_lap_times[~valid_laps] = np.nan

    actual_cumulative = np.cumsum(np.where(valid_laps, actual_lap_times, 0.0)).astype(float)
    simulated_cumulative = np.cumsum(np.where(valid_laps, simulated_lap_times, 0.0)).astype(float)
    actual_cumulative[~valid_laps] = np.nan
    simulated_cumulative[~valid_laps] = np.nan

    all_laps: pd.DataFrame = dataset.get("laps", pd.DataFrame())
    max_session_lap = int(all_laps["LapNumber"].max()) if not all_laps.empty else total_laps
    race_complete = total_laps == max_session_lap

    extrapolated = np.array([
        int(age) > int(observed_ages.get(str(compound).upper(), 0))
        for age, compound in zip(sim_features["tyre_life"], sim_features["compound"])
    ])
    warnings: list[str] = []
    if extrapolated.any():
        warnings.append("Tyre-age extrapolation: " + ", ".join(sorted(sim_features.loc[extrapolated, "compound"].unique())) + " exceeds the observed historical age range.")
    if pit_estimate["samples"] < 3:
        warnings.append("Pit-loss estimate has fewer than three observed historical pit laps; treat the assumed pit loss as uncertain.")
    if not valid_laps.all():
        warnings.append("Recorded timing is missing for one or more laps; elapsed-time metrics use only equivalent valid laps.")
    if not race_complete:
        warnings.append("Recorded totals cover completed laps only; no full-race extrapolation is shown.")

    actual_total = float(actual_lap_times[valid_laps].sum())
    simulated_total = float(simulated_lap_times[valid_laps].sum())
    frame = pd.DataFrame(
        {
            "LapNumber": sim_features["lap_number"], "Stint": sim_features["stint_index"], "Compound": sim_features["compound"],
            "TyreLife": sim_features["tyre_life"].astype(int), "StintLength": sim_features["stint_length"].astype(int),
            "IsPitLap": [lap in simulated_pit_set for lap in sim_features["lap_number"]], "ModelRawLapTime": np.round(simulated_raw, 3),
            "ActualLapTime": np.round(actual_lap_times, 3), "SimulatedLapTime": np.round(simulated_lap_times, 3),
            "ActualCumulativeTime": np.round(actual_cumulative, 3), "SimulatedCumulativeTime": np.round(simulated_cumulative, 3),
            "ActualPitLoss": np.round(recorded_pit_loss, 3),
            "ModelledActualPitLoss": np.round(modelled_actual_pit_losses, 3),
            "AssumedPitLoss": np.round(assumed_pit_losses, 3),
            "LapDelta": np.round(actual_lap_times - simulated_lap_times, 3), "CumulativeDelta": np.round(actual_cumulative - simulated_cumulative, 3),
            "ActualStint": actual_features["stint_index"].astype(int), "ActualTyreLife": actual_features["tyre_life"].round().astype(int),
            "ExceedsObservedTyreAge": extrapolated,
        }
    )

    summary = {
        "actual_total_time": round(actual_total, 3), "simulated_total_time": round(simulated_total, 3),
        "net_time_delta": round(actual_total - simulated_total, 3), "comparison_lap_count": int(valid_laps.sum()),
        "recorded_lap_count": total_laps, "race_complete": race_complete,
        "actual_avg_lap": round(float(actual_lap_times[valid_laps].mean()), 3),
        "simulated_avg_lap": round(float(simulated_lap_times[valid_laps].mean()), 3),
        "actual_pit_stops": len(actual_pit_set), "simulated_pit_stops": len(simulated_pit_set),
        "actual_pit_loss": round(float(recorded_pit_loss.sum()), 3),
        "modelled_actual_pit_loss": round(float(modelled_actual_pit_losses.sum()), 3),
        "assumed_pit_loss": round(assumed_pit_loss, 3),
        "simulated_pit_loss": round(float(sum(frame["AssumedPitLoss"])), 3), "pit_loss_samples": int(pit_estimate["samples"]),
        "pit_loss_source": str(pit_estimate["source"]), "driver_team": team, "warnings": warnings,
        "matches_recorded_plan": recorded_plan,
    }
    return SimulationResult(driver, event_name, team, total_laps, assumed_pit_loss, frame, summary, [stint.to_dict() for stint in sim_stints], actual["actual_stints"])
