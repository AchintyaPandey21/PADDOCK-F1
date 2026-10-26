"""Repository layer for reading and writing F1 session data in PostgreSQL."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.analytics import (
    build_driver_summary,
    build_heatmap_matrix,
    build_position_data,
    build_sector_summary,
    build_stint_table,
    get_pace_laps,
    timedelta_to_seconds,
)
from utils.constants import FALLBACK_DRIVER_COLORS
from .connection import get_db_connection


def is_session_stored(year: int, grand_prix: str, session_name: str) -> bool:
    """Check if the requested race session is already stored and loaded in PostgreSQL."""
    query = """
        SELECT s.is_loaded
        FROM sessions s
        JOIN races r ON s.race_id = r.race_id
        WHERE r.season_year = %s
          AND LOWER(r.event_name) = LOWER(%s)
          AND LOWER(s.session_name) = LOWER(%s)
        LIMIT 1;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (year, grand_prix, session_name))
                row = cur.fetchone()
                return bool(row and row[0])
    except Exception:
        return False


def save_session_data(
    year: int,
    grand_prix: str,
    session_name: str,
    session_obj: Any,
    dataset: dict[str, Any],
    metadata: dict[str, str],
    roster: pd.DataFrame,
) -> None:
    """Store complete session data into PostgreSQL in a single ACID transaction.

    Inserts seasons, circuits, teams, drivers, races, sessions, race_results,
    laps, tyre_stints, pit_stops, and weather_observations.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. Season
            cur.execute(
                "INSERT INTO seasons (year) VALUES (%s) ON CONFLICT (year) DO NOTHING;",
                (year,),
            )

            # 2. Circuit
            circuit_short = metadata.get("track") or grand_prix.replace(" Grand Prix", "")
            circuit_key = circuit_short.lower().replace(" ", "_")
            cur.execute(
                """
                INSERT INTO circuits (circuit_key, circuit_name, location, country)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (circuit_key) DO UPDATE
                SET circuit_name = EXCLUDED.circuit_name,
                    location = EXCLUDED.location,
                    country = EXCLUDED.country
                RETURNING circuit_id;
                """,
                (circuit_key, circuit_short, metadata.get("location", ""), metadata.get("country", "")),
            )
            circuit_id = cur.fetchone()[0]

            # 3. Teams
            team_map: dict[str, int] = {}
            unique_teams = set()
            if not roster.empty and "Team" in roster.columns:
                unique_teams.update(roster["Team"].dropna().unique())
            if "Team" in dataset.get("laps", pd.DataFrame()).columns:
                unique_teams.update(dataset["laps"]["Team"].dropna().unique())

            for team_name in unique_teams:
                team_name_str = str(team_name).strip()
                if not team_name_str:
                    continue
                cur.execute(
                    """
                    INSERT INTO teams (team_name)
                    VALUES (%s)
                    ON CONFLICT (team_name) DO UPDATE
                    SET team_name = EXCLUDED.team_name
                    RETURNING team_id;
                    """,
                    (team_name_str,),
                )
                team_map[team_name_str] = cur.fetchone()[0]

            # 4. Drivers
            driver_map: dict[str, int] = {}
            if not roster.empty:
                for _, row in roster.iterrows():
                    abbr = str(row.get("Driver", "")).strip()
                    if not abbr:
                        continue
                    full_name = str(row.get("FullName", abbr)).strip() or abbr
                    cur.execute(
                        """
                        INSERT INTO drivers (abbreviation, full_name)
                        VALUES (%s, %s)
                        ON CONFLICT (abbreviation) DO UPDATE
                        SET full_name = EXCLUDED.full_name
                        RETURNING driver_id;
                        """,
                        (abbr, full_name),
                    )
                    driver_map[abbr] = cur.fetchone()[0]

            # 5. Race
            round_num = int(metadata.get("round", 1)) if str(metadata.get("round", "")).isdigit() else 1
            cur.execute(
                """
                INSERT INTO races (season_year, round_number, event_name, official_event_name, circuit_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (season_year, event_name) DO UPDATE
                SET round_number = EXCLUDED.round_number,
                    circuit_id = EXCLUDED.circuit_id
                RETURNING race_id;
                """,
                (year, round_num, grand_prix, metadata.get("event_name", grand_prix), circuit_id),
            )
            race_id = cur.fetchone()[0]

            # 6. Session
            cur.execute(
                """
                INSERT INTO sessions (race_id, session_name, session_type, is_loaded)
                VALUES (%s, %s, %s, TRUE)
                ON CONFLICT (race_id, session_name) DO UPDATE
                SET is_loaded = TRUE
                RETURNING session_id;
                """,
                (race_id, session_name, session_name),
            )
            session_id = cur.fetchone()[0]

            # 7. Race Results (if available)
            results_df = getattr(session_obj, "results", None)
            if results_df is not None and not pd.DataFrame(results_df).empty:
                results_data = pd.DataFrame(results_df).copy()
                for _, res in results_data.iterrows():
                    abbr = str(res.get("Abbreviation", "")).strip()
                    driver_id = driver_map.get(abbr)
                    if not driver_id:
                        continue
                    team_str = str(res.get("TeamName", "")).strip()
                    team_id = team_map.get(team_str)
                    pos = int(res["Position"]) if pd.notna(res.get("Position")) else None
                    grid = int(res["GridPosition"]) if pd.notna(res.get("GridPosition")) else None
                    points = float(res["Points"]) if pd.notna(res.get("Points")) else 0.0
                    laps_comp = int(res["Laps"]) if pd.notna(res.get("Laps")) else None
                    status_str = str(res.get("Status", "Finished"))
                    class_pos = str(res.get("ClassifiedPosition", pos or ""))

                    cur.execute(
                        """
                        INSERT INTO race_results (
                            session_id, driver_id, team_id, position, classified_position,
                            grid_position, status, points, laps_completed
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id, driver_id) DO UPDATE
                        SET position = EXCLUDED.position,
                            classified_position = EXCLUDED.classified_position,
                            status = EXCLUDED.status,
                            points = EXCLUDED.points;
                        """,
                        (session_id, driver_id, team_id, pos, class_pos, grid, status_str, points, laps_comp),
                    )

            # 8. Laps
            laps_df = dataset.get("laps", pd.DataFrame())
            if not laps_df.empty:
                for _, lap in laps_df.iterrows():
                    abbr = str(lap.get("Driver", "")).strip()
                    driver_id = driver_map.get(abbr)
                    if not driver_id:
                        continue

                    team_str = str(lap.get("Team", "")).strip()
                    team_id = team_map.get(team_str)

                    lap_num = int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else 0
                    if lap_num <= 0:
                        continue

                    pos = int(lap["Position"]) if pd.notna(lap.get("Position")) else None
                    lap_time_s = float(lap["LapTimeSeconds"]) if pd.notna(lap.get("LapTimeSeconds")) else None
                    s1 = float(lap["Sector1TimeSeconds"]) if pd.notna(lap.get("Sector1TimeSeconds")) else None
                    s2 = float(lap["Sector2TimeSeconds"]) if pd.notna(lap.get("Sector2TimeSeconds")) else None
                    s3 = float(lap["Sector3TimeSeconds"]) if pd.notna(lap.get("Sector3TimeSeconds")) else None
                    compound = str(lap.get("Compound", "UNKNOWN")).upper()
                    stint = int(lap["Stint"]) if pd.notna(lap.get("Stint")) else 1
                    tyre_life = int(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else None
                    fresh_tyre = bool(lap.get("FreshTyre", False))
                    track_status = str(lap.get("TrackStatus", ""))
                    is_accurate = bool(lap.get("IsAccurate", True))
                    is_timed = bool(lap.get("IsTimedLap", False))
                    is_push = bool(lap.get("IsPushLap", False))
                    pit_in_s = timedelta_to_seconds(lap.get("PitInTime"))
                    pit_out_s = timedelta_to_seconds(lap.get("PitOutTime"))
                    pit_in_val = float(pit_in_s) if pd.notna(pit_in_s) else None
                    pit_out_val = float(pit_out_s) if pd.notna(pit_out_s) else None

                    cur.execute(
                        """
                        INSERT INTO laps (
                            session_id, driver_id, team_id, lap_number, position,
                            lap_time_seconds, sector1_time_seconds, sector2_time_seconds, sector3_time_seconds,
                            compound, stint, tyre_life, fresh_tyre, track_status,
                            is_accurate, is_timed_lap, is_push_lap, pit_in_time_seconds, pit_out_time_seconds
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id, driver_id, lap_number) DO UPDATE
                        SET lap_time_seconds = EXCLUDED.lap_time_seconds,
                            position = EXCLUDED.position,
                            compound = EXCLUDED.compound,
                            is_accurate = EXCLUDED.is_accurate,
                            is_push_lap = EXCLUDED.is_push_lap;
                        """,
                        (
                            session_id, driver_id, team_id, lap_num, pos,
                            lap_time_s, s1, s2, s3,
                            compound, stint, tyre_life, fresh_tyre, track_status,
                            is_accurate, is_timed, is_push, pit_in_val, pit_out_val,
                        ),
                    )

            # 9. Tyre Stints
            stints_df = dataset.get("stints", pd.DataFrame())
            if not stints_df.empty:
                for _, stint in stints_df.iterrows():
                    abbr = str(stint.get("Driver", "")).strip()
                    driver_id = driver_map.get(abbr)
                    if not driver_id:
                        continue
                    team_str = str(stint.get("Team", "")).strip()
                    team_id = team_map.get(team_str)
                    stint_num = int(stint["Stint"]) if pd.notna(stint.get("Stint")) else 1
                    compound = str(stint.get("Compound", "UNKNOWN")).upper()
                    start_lap = int(stint["StartLap"]) if pd.notna(stint.get("StartLap")) else 1
                    end_lap = int(stint["EndLap"]) if pd.notna(stint.get("EndLap")) else start_lap
                    length = int(stint["StintLength"]) if pd.notna(stint.get("StintLength")) else 1
                    mean_lap = float(stint["MeanLapSeconds"]) if pd.notna(stint.get("MeanLapSeconds")) else None
                    fresh = bool(stint.get("FreshTyre", False))
                    mean_life = float(stint["MeanTyreLife"]) if pd.notna(stint.get("MeanTyreLife")) else None

                    cur.execute(
                        """
                        INSERT INTO tyre_stints (
                            session_id, driver_id, team_id, stint_number, compound,
                            start_lap, end_lap, stint_length, mean_lap_seconds, fresh_tyre, mean_tyre_life
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id, driver_id, stint_number) DO UPDATE
                        SET start_lap = EXCLUDED.start_lap,
                            end_lap = EXCLUDED.end_lap,
                            stint_length = EXCLUDED.stint_length,
                            mean_lap_seconds = EXCLUDED.mean_lap_seconds;
                        """,
                        (
                            session_id, driver_id, team_id, stint_num, compound,
                            start_lap, end_lap, length, mean_lap, fresh, mean_life,
                        ),
                    )

            # 10. Weather Observations
            weather_data = getattr(session_obj, "weather_data", None)
            if weather_data is not None and not pd.DataFrame(weather_data).empty:
                w_df = pd.DataFrame(weather_data).copy()
                for _, w_row in w_df.iterrows():
                    air_t = float(w_row["AirTemp"]) if pd.notna(w_row.get("AirTemp")) else None
                    track_t = float(w_row["TrackTemp"]) if pd.notna(w_row.get("TrackTemp")) else None
                    hum = float(w_row["Humidity"]) if pd.notna(w_row.get("Humidity")) else None
                    press = float(w_row["Pressure"]) if pd.notna(w_row.get("Pressure")) else None
                    wind_s = float(w_row["WindSpeed"]) if pd.notna(w_row.get("WindSpeed")) else None
                    wind_d = int(w_row["WindDirection"]) if pd.notna(w_row.get("WindDirection")) else None
                    rain = bool(w_row.get("Rainfall", False))

                    cur.execute(
                        """
                        INSERT INTO weather_observations (
                            session_id, air_temp, track_temp, humidity, pressure,
                            wind_speed, wind_direction, rainfall
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (session_id, air_t, track_t, hum, press, wind_s, wind_d, rain),
                    )

        conn.commit()


def load_session_data(year: int, grand_prix: str, session_name: str) -> dict[str, Any]:
    """Retrieve complete race session dataset from PostgreSQL without calling FastF1.

    Returns the exact context structures expected by the analytics layer and dashboard views.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. Resolve Session, Race, and Circuit
            cur.execute(
                """
                SELECT s.session_id, r.round_number, r.event_name, c.circuit_name, c.location, c.country
                FROM sessions s
                JOIN races r ON s.race_id = r.race_id
                LEFT JOIN circuits c ON r.circuit_id = c.circuit_id
                WHERE r.season_year = %s
                  AND LOWER(r.event_name) = LOWER(%s)
                  AND LOWER(s.session_name) = LOWER(%s)
                LIMIT 1;
                """,
                (year, grand_prix, session_name),
            )
            sess_row = cur.fetchone()
            if not sess_row:
                raise ValueError(f"Session '{year} {grand_prix} {session_name}' not found in PostgreSQL.")

            session_id, round_num, event_name, track_name, location, country = sess_row

            metadata = {
                "event_name": event_name or grand_prix,
                "country": country or "",
                "location": location or "",
                "round": str(round_num or ""),
                "date": f"{year}",
                "session_name": session_name,
                "track": track_name or location or "",
            }

            # 2. Drivers and Teams (Roster)
            cur.execute(
                """
                SELECT DISTINCT d.abbreviation, d.full_name, COALESCE(t.team_name, '') as team_name, t.team_color
                FROM laps l
                JOIN drivers d ON l.driver_id = d.driver_id
                LEFT JOIN teams t ON l.team_id = t.team_id
                WHERE l.session_id = %s
                ORDER BY d.abbreviation;
                """,
                (session_id,),
            )
            roster_rows = cur.fetchall()

            roster_list = []
            driver_colors: dict[str, str] = {}
            for idx, r_row in enumerate(roster_rows):
                abbr, full_name, team_name, team_color = r_row
                color = team_color or FALLBACK_DRIVER_COLORS[idx % len(FALLBACK_DRIVER_COLORS)]
                driver_colors[abbr] = color
                roster_list.append(
                    {
                        "Driver": abbr,
                        "FullName": full_name or abbr,
                        "Team": team_name,
                        "Label": f"{abbr} | {full_name} ({team_name})".strip(),
                    }
                )

            roster = pd.DataFrame(roster_list)
            driver_labels = dict(zip(roster["Driver"], roster["Label"])) if not roster.empty else {}

            # 3. Laps
            cur.execute(
                """
                SELECT
                    d.abbreviation AS "Driver",
                    l.lap_number AS "LapNumber",
                    l.position AS "Position",
                    l.lap_time_seconds AS "LapTimeSeconds",
                    l.sector1_time_seconds AS "Sector1TimeSeconds",
                    l.sector2_time_seconds AS "Sector2TimeSeconds",
                    l.sector3_time_seconds AS "Sector3TimeSeconds",
                    l.compound AS "Compound",
                    l.stint AS "Stint",
                    l.tyre_life AS "TyreLife",
                    l.fresh_tyre AS "FreshTyre",
                    COALESCE(t.team_name, '') AS "Team",
                    l.track_status AS "TrackStatus",
                    l.is_accurate AS "IsAccurate",
                    l.is_timed_lap AS "IsTimedLap",
                    l.is_push_lap AS "IsPushLap",
                    l.pit_in_time_seconds,
                    l.pit_out_time_seconds
                FROM laps l
                JOIN drivers d ON l.driver_id = d.driver_id
                LEFT JOIN teams t ON l.team_id = t.team_id
                WHERE l.session_id = %s
                ORDER BY d.abbreviation, l.lap_number;
                """,
                (session_id,),
            )
            columns = [desc[0] for desc in cur.description]
            lap_records = cur.fetchall()
            laps = pd.DataFrame(lap_records, columns=columns)

            if not laps.empty:
                # Reconstruct timedelta columns expected by analytics & telemetry
                laps["LapTime"] = pd.to_timedelta(laps["LapTimeSeconds"], unit="s")
                laps["Sector1Time"] = pd.to_timedelta(laps["Sector1TimeSeconds"], unit="s")
                laps["Sector2Time"] = pd.to_timedelta(laps["Sector2TimeSeconds"], unit="s")
                laps["Sector3Time"] = pd.to_timedelta(laps["Sector3TimeSeconds"], unit="s")
                laps["PitInTime"] = pd.to_timedelta(laps["pit_in_time_seconds"], unit="s")
                laps["PitOutTime"] = pd.to_timedelta(laps["pit_out_time_seconds"], unit="s")
            else:
                for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "PitInTime", "PitOutTime"]:
                    laps[col] = pd.NaT

            # 4. Weather Summary
            cur.execute(
                """
                SELECT
                    AVG(air_temp) as air_temp,
                    AVG(track_temp) as track_temp,
                    AVG(humidity) as humidity,
                    AVG(wind_speed) as wind_speed,
                    AVG(CASE WHEN rainfall THEN 1.0 ELSE 0.0 END) * 100 as rainfall_pct
                FROM weather_observations
                WHERE session_id = %s;
                """,
                (session_id,),
            )
            weather_row = cur.fetchone()
            weather_summary: dict[str, str] = {}
            if weather_row and weather_row[0] is not None:
                weather_summary = {
                    "air_temp": f"{weather_row[0]:.1f}",
                    "track_temp": f"{weather_row[1]:.1f}" if weather_row[1] is not None else "N/A",
                    "humidity": f"{weather_row[2]:.1f}" if weather_row[2] is not None else "N/A",
                    "wind_speed": f"{weather_row[3]:.1f}" if weather_row[3] is not None else "N/A",
                    "rainfall": f"{weather_row[4]:.0f}%" if weather_row[4] is not None else "0%",
                }

            # 5. Build Reusable Analytics Dataset
            driver_summary = build_driver_summary(laps)
            drivers = driver_summary["Driver"].tolist() if not driver_summary.empty else sorted(laps["Driver"].unique())

            dataset = {
                "laps": laps,
                "pace_laps": get_pace_laps(laps),
                "driver_summary": driver_summary,
                "sector_summary": build_sector_summary(laps),
                "stints": build_stint_table(laps),
                "positions": build_position_data(laps),
                "heatmap": build_heatmap_matrix(laps),
                "weather_summary": weather_summary,
                "drivers": drivers,
            }

            return {
                "metadata": metadata,
                "dataset": dataset,
                "roster": roster,
                "driver_colors": driver_colors,
                "driver_labels": driver_labels,
            }
