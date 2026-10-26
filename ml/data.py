"""Data extraction and cleaning for Paddock F1 ML Strategy Simulator."""

from __future__ import annotations

from typing import Any
import pandas as pd

from db.connection import get_db_connection

VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def get_raw_lap_dataset() -> pd.DataFrame:
    """Query raw lap records joined with race, session, circuit, driver, team, and weather data."""
    query = """
        WITH weather_summary AS (
            SELECT 
                session_id,
                ROUND(AVG(air_temp)::numeric, 2) AS air_temp,
                ROUND(AVG(track_temp)::numeric, 2) AS track_temp,
                ROUND(AVG(humidity)::numeric, 2) AS humidity,
                CASE WHEN BOOL_OR(rainfall) THEN 1 ELSE 0 END AS rainfall
            FROM weather_observations
            GROUP BY session_id
        )
        SELECT 
            l.lap_id,
            l.session_id,
            r.race_id,
            r.event_name,
            r.season_year,
            r.round_number,
            COALESCE(c.circuit_name, r.event_name) AS circuit_name,
            d.abbreviation AS driver,
            d.full_name AS driver_name,
            COALESCE(t.team_name, 'Unknown') AS team,
            l.lap_number,
            l.stint,
            UPPER(TRIM(l.compound)) AS compound,
            l.tyre_life,
            COALESCE(l.fresh_tyre, FALSE) AS fresh_tyre,
            l.track_status,
            l.is_accurate,
            l.is_timed_lap,
            l.pit_in_time_seconds,
            l.pit_out_time_seconds,
            w.air_temp,
            w.track_temp,
            w.humidity,
            w.rainfall,
            l.lap_time_seconds
        FROM laps l
        JOIN sessions s ON l.session_id = s.session_id
        JOIN races r ON s.race_id = r.race_id
        LEFT JOIN circuits c ON r.circuit_id = c.circuit_id
        JOIN drivers d ON l.driver_id = d.driver_id
        LEFT JOIN teams t ON l.team_id = t.team_id
        LEFT JOIN weather_summary w ON s.session_id = w.session_id
        WHERE s.session_name = 'Race'
        ORDER BY r.race_id, d.abbreviation, l.lap_number;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            records = cur.fetchall()
            return pd.DataFrame(records, columns=columns)


def load_clean_training_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Query, filter, and engineer features from PostgreSQL laps.

    Filters:
      - Valid positive lap_time_seconds
      - Accurate laps (is_accurate = TRUE)
      - Green-flag racing laps only (track_status = '1')
      - Exclude pit-in and pit-out laps (transit time loss modeled separately in simulator)
      - Standard known tyre compounds (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
      - Valid tyre_life > 0

    Returns:
      tuple of (clean_df, audit_metadata)
    """
    df = get_raw_lap_dataset()
    initial_count = len(df)

    # Preserve the observed stint distance so the model can distinguish an
    # early-life tyre on a short run from one planned for a long run.
    stint_groups = ["session_id", "driver", "stint"]
    df["stint_length"] = (
        df.groupby(stint_groups, dropna=False)["lap_number"]
        .transform(lambda laps: laps.max() - laps.min() + 1)
    )

    audit: dict[str, Any] = {
        "raw_laps_count": initial_count,
        "filter_steps": {},
    }

    # Step 1: Must have valid timed lap
    step1 = df[df["lap_time_seconds"].notna() & (df["lap_time_seconds"] > 0)].copy()
    audit["filter_steps"]["valid_lap_time"] = len(step1)

    # Step 2: Must be marked accurate by FastF1 timing telemetry
    step2 = step1[step1["is_accurate"] == True].copy()
    audit["filter_steps"]["accurate_laps"] = len(step2)

    # Step 3: Green flag racing laps only (track_status = '1')
    step3 = step2[step2["track_status"].astype(str) == "1"].copy()
    audit["filter_steps"]["green_flag_laps"] = len(step3)

    # Step 4: Exclude pit-in and pit-out laps (pit stops handled explicitly in simulation engine)
    step4 = step3[
        step3["pit_in_time_seconds"].isna() & step3["pit_out_time_seconds"].isna()
    ].copy()
    audit["filter_steps"]["non_pit_laps"] = len(step4)

    # Step 5: Valid known compounds
    step5 = step4[step4["compound"].isin(VALID_COMPOUNDS)].copy()
    audit["filter_steps"]["known_compounds"] = len(step5)

    # Step 6: Valid tyre life / age
    step6 = step5[step5["tyre_life"].notna() & (step5["tyre_life"] > 0)].copy()
    audit["filter_steps"]["valid_tyre_life"] = len(step6)

    # Convert datatypes cleanly
    clean_df = step6.copy()
    clean_df["lap_time_seconds"] = clean_df["lap_time_seconds"].astype(float)
    clean_df["lap_number"] = clean_df["lap_number"].astype(int)
    clean_df["tyre_life"] = clean_df["tyre_life"].astype(float)
    clean_df["stint_length"] = clean_df["stint_length"].astype(float)
    clean_df["air_temp"] = clean_df["air_temp"].astype(float).fillna(25.0)
    clean_df["track_temp"] = clean_df["track_temp"].astype(float).fillna(35.0)
    clean_df["humidity"] = clean_df["humidity"].astype(float).fillna(50.0)
    clean_df["rainfall"] = clean_df["rainfall"].astype(int).fillna(0)

    # Engineer circuit baseline reference time (median racing lap per circuit)
    circuit_ref_map = (
        clean_df.groupby("event_name")["lap_time_seconds"].median().round(3).to_dict()
    )
    clean_df["circuit_ref_time"] = clean_df["event_name"].map(circuit_ref_map)

    # Record event counts and distributions
    audit["final_usable_sample_size"] = len(clean_df)
    audit["events"] = clean_df["event_name"].value_counts().to_dict()
    audit["compounds"] = clean_df["compound"].value_counts().to_dict()
    audit["circuit_benchmarks"] = circuit_ref_map
    audit["observed_tyre_age_by_compound"] = (
        clean_df.groupby("compound")["tyre_life"].max().astype(int).to_dict()
    )
    audit["pit_loss_estimates"] = estimate_pit_loss_estimates(df)

    return clean_df.reset_index(drop=True), audit


def estimate_pit_loss_estimates(raw_laps: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Estimate full pit-transition loss from the entry and following out-lap.

    FastF1 records a stop over two laps: the pit-entry lap and normally the
    following out-lap. Measuring only entry systematically understates the
    stop cost, so both laps are compared to the same driver's nearby racing
    pace. A stint change is retained as a fallback only for incomplete timing.
    """
    estimates: dict[str, list[tuple[float, float]]] = {}
    global_samples: list[tuple[float, float]] = []

    for (event_name, session_id, driver), group in raw_laps.groupby(
        ["event_name", "session_id", "driver"], dropna=False
    ):
        ordered = group.sort_values("lap_number").reset_index(drop=True)
        lap_times = pd.to_numeric(ordered["lap_time_seconds"], errors="coerce")
        pit_marker = ordered["pit_in_time_seconds"].notna()
        next_stint = ordered["stint"].shift(-1)
        stint_change = ordered["stint"].notna() & next_stint.notna() & ordered["stint"].ne(next_stint)
        candidate_rows = ordered.index[pit_marker]
        if not len(candidate_rows):
            candidate_rows = ordered.index[stint_change]

        for idx in candidate_rows:
            affected = [idx]
            if idx + 1 < len(ordered) and pd.notna(lap_times.iloc[idx + 1]) and lap_times.iloc[idx + 1] > 0:
                affected.append(idx + 1)
            if not any(pd.notna(lap_times.iloc[row]) and lap_times.iloc[row] > 0 for row in affected):
                continue
            neighborhood = ordered.index[(ordered.index >= idx - 3) & (ordered.index <= idx + 4)]
            valid_neighbors = [
                row for row in neighborhood
                if row not in affected
                and not bool(pit_marker.iloc[row])
                and row - 1 not in set(candidate_rows)
                and pd.notna(lap_times.iloc[row])
                and lap_times.iloc[row] > 0
            ]
            if not valid_neighbors:
                continue
            baseline = float(lap_times.iloc[valid_neighbors].median())
            entry_loss = max(0.0, float(lap_times.iloc[idx] - baseline)) if pd.notna(lap_times.iloc[idx]) else 0.0
            out_loss = (
                max(0.0, float(lap_times.iloc[idx + 1] - baseline))
                if len(affected) == 2 else 0.0
            )
            if entry_loss + out_loss > 0:
                sample = (entry_loss, out_loss)
                estimates.setdefault(str(event_name), []).append(sample)
                global_samples.append(sample)

    def summarize(samples: list[tuple[float, float]]) -> dict[str, float | int]:
        entry = pd.Series([sample[0] for sample in samples], dtype=float)
        out_lap = pd.Series([sample[1] for sample in samples], dtype=float)
        entry_seconds = float(entry.median()) if not entry.empty else 0.0
        out_seconds = float(out_lap.median()) if not out_lap.empty else 0.0
        return {
            "seconds": round(entry_seconds + out_seconds, 3),
            "entry_lap_seconds": round(entry_seconds, 3),
            "out_lap_seconds": round(out_seconds, 3),
            "samples": len(samples),
        }

    result = {event: summarize(samples) for event, samples in estimates.items()}
    result["default"] = summarize(global_samples)
    return result
