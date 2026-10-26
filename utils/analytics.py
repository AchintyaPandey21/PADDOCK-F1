"""Data transformation helpers used throughout the dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    """Return a column if it exists, otherwise a same-length fallback series."""
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def timedelta_to_seconds(value: Any) -> float:
    """Convert a timedelta-like value to seconds."""
    if value is None or pd.isna(value):
        return np.nan
    try:
        return pd.to_timedelta(value).total_seconds()
    except Exception:
        return np.nan


def format_lap_time(value: Any) -> str:
    """Format a lap time like 1:32.456."""
    if value is None or pd.isna(value):
        return "N/A"

    total_seconds = timedelta_to_seconds(value)
    if pd.isna(total_seconds):
        return "N/A"

    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    return f"{minutes}:{seconds:06.3f}"


def format_sector_time(value: Any) -> str:
    """Format a sector time in seconds with milliseconds."""
    if value is None or pd.isna(value):
        return "N/A"
    seconds = timedelta_to_seconds(value)
    if pd.isna(seconds):
        return "N/A"
    return f"{seconds:.3f}s"


def format_delta(seconds: float) -> str:
    """Format a signed delta in seconds."""
    if seconds is None or pd.isna(seconds):
        return "N/A"
    sign = "+" if seconds > 0 else ""
    return f"{sign}{seconds:.3f}s"


def prepare_lap_data(session: Any) -> pd.DataFrame:
    """Create a cleaned lap dataframe with numeric helper columns."""
    laps = pd.DataFrame(session.laps).copy()
    if laps.empty:
        return laps

    time_columns = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for column in time_columns:
        if column in laps.columns:
            laps[f"{column}Seconds"] = laps[column].apply(timedelta_to_seconds)
        else:
            laps[f"{column}Seconds"] = np.nan

    laps["Driver"] = laps["Driver"].astype(str)
    laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    laps["Position"] = pd.to_numeric(_series_or_default(laps, "Position", np.nan), errors="coerce")
    laps["Compound"] = _series_or_default(laps, "Compound", "UNKNOWN").fillna("UNKNOWN").astype(str).str.upper()
    laps["Stint"] = pd.to_numeric(_series_or_default(laps, "Stint", 1), errors="coerce").fillna(1).astype(int)
    laps["TyreLife"] = pd.to_numeric(_series_or_default(laps, "TyreLife", np.nan), errors="coerce")
    laps["FreshTyre"] = _series_or_default(laps, "FreshTyre", False).fillna(False).astype(bool)
    laps["Team"] = _series_or_default(laps, "Team", "").fillna("").astype(str)
    laps["TrackStatus"] = _series_or_default(laps, "TrackStatus", "").fillna("").astype(str)
    laps["IsAccurate"] = _series_or_default(laps, "IsAccurate", True).fillna(True).astype(bool)
    laps["IsTimedLap"] = laps["LapTimeSeconds"].notna()

    pit_out = laps["PitOutTime"] if "PitOutTime" in laps.columns else None
    pit_in = laps["PitInTime"] if "PitInTime" in laps.columns else None
    laps["IsPushLap"] = laps["IsTimedLap"]
    if pit_out is not None:
        laps["IsPushLap"] &= pit_out.isna()
    if pit_in is not None:
        laps["IsPushLap"] &= pit_in.isna()
    laps["IsPushLap"] &= laps["IsAccurate"]

    return laps


def get_pace_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Filter to representative racing laps for pace analysis."""
    if laps.empty or "Driver" not in laps.columns or "LapTimeSeconds" not in laps.columns:
        return laps

    push_laps = laps.loc[laps.get("IsPushLap", False) & laps["LapTimeSeconds"].notna()].copy()
    timed_laps = laps.loc[laps.get("IsTimedLap", False) & laps["LapTimeSeconds"].notna()].copy()

    if timed_laps.empty:
        return laps.head(0).copy()

    # For drivers who have push laps, use push laps. For drivers without push laps, fall back to timed laps.
    push_drivers = set(push_laps["Driver"].unique()) if not push_laps.empty else set()
    timed_only_laps = timed_laps.loc[~timed_laps["Driver"].isin(push_drivers)]
    pace_laps = pd.concat([push_laps, timed_only_laps], ignore_index=True) if not timed_only_laps.empty else push_laps

    if pace_laps.empty:
        pace_laps = timed_laps.copy()

    # Remove extreme outliers while keeping genuine strategic variation.
    median_per_driver = pace_laps.groupby("Driver")["LapTimeSeconds"].transform("median")
    cutoff = median_per_driver * 1.15
    valid_mask = (pace_laps["LapTimeSeconds"] <= cutoff) | cutoff.isna()
    pace_laps = pace_laps.loc[valid_mask].copy()

    return pace_laps


def build_driver_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate driver-level pace, sector, and stint metrics."""
    if laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    pace_laps = get_pace_laps(laps)
    timed_laps = laps.loc[laps.get("IsTimedLap", False) & laps["LapTimeSeconds"].notna()].copy()

    # Base on all drivers appearing in the session laps
    drivers_df = laps[["Driver", "Team"]].drop_duplicates("Driver").copy()

    # Lap counts
    lap_counts = (
        laps.groupby("Driver", as_index=False)
        .agg(
            Laps=("LapNumber", "count"),
            Team=("Team", "last"),
        )
    )

    # Fastest lap per driver across all valid timed laps
    if not timed_laps.empty:
        fastest_agg = (
            timed_laps.groupby("Driver", as_index=False)
            .agg(
                FastestLapSeconds=("LapTimeSeconds", "min"),
            )
        )
        sector_metrics = (
            timed_laps.groupby("Driver", as_index=False)
            .agg(
                BestSector1=("Sector1TimeSeconds", "min"),
                BestSector2=("Sector2TimeSeconds", "min"),
                BestSector3=("Sector3TimeSeconds", "min"),
            )
            .fillna(np.nan)
        )
    else:
        fastest_agg = pd.DataFrame(columns=["Driver", "FastestLapSeconds"])
        sector_metrics = pd.DataFrame(columns=["Driver", "BestSector1", "BestSector2", "BestSector3"])

    # Representative pace and consistency metrics from pace_laps
    # (fallback to timed laps for drivers who had no push laps)
    driver_pace_source = pace_laps if not pace_laps.empty else timed_laps
    if not timed_laps.empty and not driver_pace_source.empty:
        pace_drivers = set(driver_pace_source["Driver"].unique())
        missing_from_pace = timed_laps.loc[~timed_laps["Driver"].isin(pace_drivers)]
        if not missing_from_pace.empty:
            driver_pace_source = pd.concat([driver_pace_source, missing_from_pace], ignore_index=True)

    if not driver_pace_source.empty:
        pace_agg = (
            driver_pace_source.groupby("Driver", as_index=False)
            .agg(
                AverageLapSeconds=("LapTimeSeconds", "mean"),
                MedianLapSeconds=("LapTimeSeconds", "median"),
                ConsistencySeconds=("LapTimeSeconds", "std"),
            )
            .fillna({"ConsistencySeconds": 0.0})
        )
    else:
        pace_agg = pd.DataFrame(columns=["Driver", "AverageLapSeconds", "MedianLapSeconds", "ConsistencySeconds"])

    stint_counts = (
        laps.groupby("Driver", as_index=False)
        .agg(
            Stints=("Stint", "nunique"),
            Compounds=("Compound", lambda values: " / ".join(sorted(set(str(v) for v in values if pd.notna(v) and str(v) != "UNKNOWN")))),
        )
        .fillna("")
    )

    summary = (
        drivers_df[["Driver"]]
        .merge(lap_counts, on="Driver", how="left")
        .merge(fastest_agg, on="Driver", how="left")
        .merge(pace_agg, on="Driver", how="left")
        .merge(sector_metrics, on="Driver", how="left")
        .merge(stint_counts, on="Driver", how="left")
    )

    summary["ConsistencyPct"] = np.where(
        summary["AverageLapSeconds"] > 0,
        (summary["ConsistencySeconds"] / summary["AverageLapSeconds"]) * 100,
        np.nan,
    )
    summary = summary.sort_values(
        ["FastestLapSeconds", "Laps"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    if not summary["FastestLapSeconds"].dropna().empty:
        session_best = summary["FastestLapSeconds"].dropna().min()
        summary["PaceDeltaSeconds"] = summary["FastestLapSeconds"] - session_best
    else:
        summary["PaceDeltaSeconds"] = np.nan

    summary["FastestLap"] = summary["FastestLapSeconds"].apply(
        lambda value: format_lap_time(pd.to_timedelta(value, unit="s")) if pd.notna(value) else "N/A"
    )
    summary["AverageLap"] = summary["AverageLapSeconds"].apply(
        lambda value: format_lap_time(pd.to_timedelta(value, unit="s")) if pd.notna(value) else "N/A"
    )
    summary["Consistency"] = summary["ConsistencyPct"].apply(
        lambda value: f"{value:.2f}%" if pd.notna(value) else "N/A"
    )
    summary["GapToBest"] = summary["PaceDeltaSeconds"].apply(format_delta)
    return summary


def build_sector_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Create average and best sector metrics per driver."""
    timed = laps.loc[laps["IsTimedLap"]].copy()
    if timed.empty:
        return pd.DataFrame()

    sector_summary = (
        timed.groupby("Driver", as_index=False)
        .agg(
            Team=("Team", "last"),
            AvgSector1=("Sector1TimeSeconds", "mean"),
            AvgSector2=("Sector2TimeSeconds", "mean"),
            AvgSector3=("Sector3TimeSeconds", "mean"),
            BestSector1=("Sector1TimeSeconds", "min"),
            BestSector2=("Sector2TimeSeconds", "min"),
            BestSector3=("Sector3TimeSeconds", "min"),
        )
        .sort_values(["AvgSector1", "AvgSector2", "AvgSector3"])
        .reset_index(drop=True)
    )
    return sector_summary


def build_stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Group laps into tyre stints by driver, stint, and compound."""
    if laps.empty:
        return pd.DataFrame()

    stints = (
        laps.groupby(["Driver", "Team", "Stint", "Compound"], dropna=False, as_index=False)
        .agg(
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            StintLength=("LapNumber", "count"),
            MeanLapSeconds=("LapTimeSeconds", "mean"),
            FreshTyre=("FreshTyre", "max"),
            MeanTyreLife=("TyreLife", "mean"),
        )
        .sort_values(["Driver", "StartLap", "Stint"])
        .reset_index(drop=True)
    )
    stints["MeanLap"] = stints["MeanLapSeconds"].apply(
        lambda value: format_lap_time(pd.to_timedelta(value, unit="s")) if pd.notna(value) else "N/A"
    )
    return stints


def build_position_data(laps: pd.DataFrame) -> pd.DataFrame:
    """Prepare lap-by-lap race positions."""
    if laps.empty or "Position" not in laps.columns:
        return pd.DataFrame()

    positions = laps.loc[laps["Position"].notna(), ["Driver", "Team", "LapNumber", "Position"]].copy()
    return positions.sort_values(["LapNumber", "Position", "Driver"]).reset_index(drop=True)


def build_heatmap_matrix(laps: pd.DataFrame) -> pd.DataFrame:
    """Build a lap-time delta heatmap matrix using each driver's median as baseline."""
    pace_laps = get_pace_laps(laps)
    if pace_laps.empty:
        return pd.DataFrame()

    working = pace_laps[["Driver", "LapNumber", "LapTimeSeconds"]].copy()
    driver_baseline = working.groupby("Driver")["LapTimeSeconds"].transform("median")
    working["DeltaToMedian"] = working["LapTimeSeconds"] - driver_baseline
    heatmap = working.pivot_table(
        index="Driver",
        columns="LapNumber",
        values="DeltaToMedian",
        aggfunc="min",
    )
    return heatmap.sort_index()


def build_weather_summary(session: Any) -> dict[str, str]:
    """Summarise the weather conditions if data is available."""
    weather = pd.DataFrame(getattr(session, "weather_data", pd.DataFrame())).copy()
    if weather.empty:
        return {}

    summary: dict[str, str] = {}
    for column, label in (
        ("AirTemp", "air_temp"),
        ("TrackTemp", "track_temp"),
        ("Humidity", "humidity"),
        ("WindSpeed", "wind_speed"),
    ):
        if column in weather.columns:
            summary[label] = f"{weather[column].mean():.1f}"

    if "Rainfall" in weather.columns:
        rainfall_ratio = weather["Rainfall"].fillna(False).astype(bool).mean() * 100
        summary["rainfall"] = f"{rainfall_ratio:.0f}%"

    return summary


def build_dashboard_dataset(session: Any) -> dict[str, Any]:
    """Build the reusable analytics dataset shared by all pages."""
    laps = prepare_lap_data(session)
    driver_summary = build_driver_summary(laps)
    if not driver_summary.empty:
        drivers = driver_summary["Driver"].tolist()
    elif not laps.empty and "Driver" in laps.columns:
        drivers = sorted(laps["Driver"].dropna().unique().tolist())
    else:
        drivers = []

    dataset = {
        "laps": laps,
        "pace_laps": get_pace_laps(laps),
        "driver_summary": driver_summary,
        "sector_summary": build_sector_summary(laps),
        "stints": build_stint_table(laps),
        "positions": build_position_data(laps),
        "heatmap": build_heatmap_matrix(laps),
        "weather_summary": build_weather_summary(session),
        "drivers": drivers,
    }
    return dataset


def get_driver_lap_table(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Build a lap-by-lap report for one driver."""
    driver_laps = laps.loc[laps["Driver"] == driver].copy()
    if driver_laps.empty:
        return driver_laps

    driver_laps["LapTimeDisplay"] = driver_laps["LapTime"].apply(format_lap_time)
    driver_laps["Sector1Display"] = driver_laps["Sector1Time"].apply(format_sector_time)
    driver_laps["Sector2Display"] = driver_laps["Sector2Time"].apply(format_sector_time)
    driver_laps["Sector3Display"] = driver_laps["Sector3Time"].apply(format_sector_time)

    columns = [
        "LapNumber",
        "LapTimeDisplay",
        "Sector1Display",
        "Sector2Display",
        "Sector3Display",
        "Compound",
        "Stint",
        "TyreLife",
        "Position",
        "TrackStatus",
    ]
    available_columns = [column for column in columns if column in driver_laps.columns]
    return driver_laps[available_columns].sort_values("LapNumber")


def build_telemetry_bundle(session: Any, driver_a: str, driver_b: str) -> dict[str, Any]:
    """Prepare telemetry and racing-line data for two drivers."""
    lap_a = session.laps.pick_drivers(driver_a).pick_fastest()
    lap_b = session.laps.pick_drivers(driver_b).pick_fastest()

    if lap_a is None or lap_b is None or getattr(lap_a, "empty", True) or getattr(lap_b, "empty", True):
        raise ValueError("Fastest laps could not be resolved for the selected drivers.")

    if pd.isna(lap_a.get("LapTime")) or pd.isna(lap_b.get("LapTime")):
        raise ValueError("Telemetry comparison requires valid timed laps for both drivers.")

    car_data_a = lap_a.get_car_data()
    car_data_b = lap_b.get_car_data()
    if car_data_a.empty or car_data_b.empty:
        raise ValueError("Car telemetry is not available for one or both fastest laps.")

    telemetry_a = car_data_a.add_distance().copy()
    telemetry_b = car_data_b.add_distance().copy()

    try:
        position_a = lap_a.get_pos_data().copy()
    except Exception:
        position_a = pd.DataFrame()

    try:
        position_b = lap_b.get_pos_data().copy()
    except Exception:
        position_b = pd.DataFrame()

    # Positional telemetry already contains the X/Y channels required for the
    # racing-line overlay. Unlike car telemetry it has no Speed channel, so
    # calling add_distance() on it raises a FastF1 ValueError.

    channels = [channel for channel in ["Speed", "Throttle", "Brake", "nGear", "RPM"] if channel in telemetry_a.columns and channel in telemetry_b.columns]
    trimmed_a = telemetry_a[["Distance", *channels]].dropna(subset=["Distance"]).sort_values("Distance").drop_duplicates(subset=["Distance"])
    trimmed_b = telemetry_b[["Distance", *channels]].dropna(subset=["Distance"]).sort_values("Distance").drop_duplicates(subset=["Distance"])

    merged = pd.merge_asof(
        trimmed_a,
        trimmed_b,
        on="Distance",
        direction="nearest",
        suffixes=(f"_{driver_a}", f"_{driver_b}"),
    )

    for driver in (driver_a, driver_b):
        brake_column = f"Brake_{driver}"
        if brake_column in merged.columns:
            merged[brake_column] = merged[brake_column].fillna(0).astype(int)

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "lap_a": lap_a,
        "lap_b": lap_b,
        "merged": merged,
        "position_a": position_a,
        "position_b": position_b,
    }
