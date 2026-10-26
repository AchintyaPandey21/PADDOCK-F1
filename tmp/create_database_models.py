"""Generate verified Paddock F1 database-model artifacts using Matplotlib."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "database_models"
OUT.mkdir(parents=True, exist_ok=True)

TABLES = {
    "SEASONS": ["PK year : INT"],
    "CIRCUITS": ["PK circuit_id : SERIAL", "UQ circuit_key : VARCHAR(50)", "circuit_name : VARCHAR(150) NN", "location : VARCHAR(100)", "country : VARCHAR(100)"],
    "TEAMS": ["PK team_id : SERIAL", "UQ team_name : VARCHAR(100) NN", "team_color : VARCHAR(10)"],
    "DRIVERS": ["PK driver_id : SERIAL", "UQ abbreviation : VARCHAR(10) NN", "driver_number : INT", "full_name : VARCHAR(150) NN", "broadcast_name : VARCHAR(100)", "country_code : VARCHAR(10)"],
    "RACES": ["PK race_id : SERIAL", "FK season_year : INT NN", "round_number : INT NN", "event_name : VARCHAR(150) NN", "official_event_name : VARCHAR(255)", "event_date : DATE", "event_format : VARCHAR(50)", "FK circuit_id : INT", "UQ(season_year, round_number)", "UQ(season_year, event_name)"],
    "SESSIONS": ["PK session_id : SERIAL", "FK race_id : INT NN", "session_name : VARCHAR(50) NN", "session_type : VARCHAR(50)", "session_date : TIMESTAMPTZ", "is_loaded : BOOLEAN DEFAULT FALSE", "UQ(race_id, session_name)"],
    "RACE_RESULTS": ["PK result_id : SERIAL", "FK session_id : INT NN", "FK driver_id : INT NN", "FK team_id : INT", "position : INT", "classified_position : VARCHAR(10)", "grid_position : INT", "time_seconds : DOUBLE PRECISION", "status : VARCHAR(100)", "points : DOUBLE PRECISION", "laps_completed : INT", "q1_seconds / q2_seconds / q3_seconds", "UQ(session_id, driver_id)"],
    "LAPS": ["PK lap_id : BIGSERIAL", "FK session_id : INT NN", "FK driver_id : INT NN", "FK team_id : INT", "lap_number : INT NN", "position : INT", "lap_time_seconds : DOUBLE PRECISION", "sector1/2/3_time_seconds : DOUBLE PRECISION", "compound : VARCHAR(20)", "stint : INT", "tyre_life : INT", "fresh_tyre : BOOLEAN", "track_status : VARCHAR(20)", "is_accurate / is_timed_lap / is_push_lap", "pit_in_time_seconds / pit_out_time_seconds", "UQ(session_id, driver_id, lap_number)"],
    "TYRE_STINTS": ["PK stint_id : SERIAL", "FK session_id : INT NN", "FK driver_id : INT NN", "FK team_id : INT", "stint_number : INT NN", "compound : VARCHAR(20) NN", "start_lap : INT NN", "end_lap : INT NN", "stint_length : INT NN", "mean_lap_seconds : DOUBLE PRECISION", "fresh_tyre : BOOLEAN", "mean_tyre_life : DOUBLE PRECISION", "UQ(session_id, driver_id, stint_number)"],
    "PIT_STOPS": ["PK pit_stop_id : SERIAL", "FK session_id : INT NN", "FK driver_id : INT NN", "lap_number : INT NN", "pit_in_time_seconds : DOUBLE PRECISION", "pit_out_time_seconds : DOUBLE PRECISION", "duration_seconds : DOUBLE PRECISION", "UQ(session_id, driver_id, lap_number)"],
    "WEATHER_OBSERVATIONS": ["PK weather_id : SERIAL", "FK session_id : INT NN", "sample_time_seconds : DOUBLE PRECISION", "air_temp : DOUBLE PRECISION", "track_temp : DOUBLE PRECISION", "humidity : DOUBLE PRECISION", "pressure : DOUBLE PRECISION", "wind_speed : DOUBLE PRECISION", "wind_direction : INT", "rainfall : BOOLEAN"],
}

POSITIONS = {
    "SEASONS": (0.04, 0.80), "CIRCUITS": (0.37, 0.76), "TEAMS": (0.71, 0.80), "DRIVERS": (0.04, 0.53),
    "RACES": (0.37, 0.51), "SESSIONS": (0.71, 0.55), "RACE_RESULTS": (0.04, 0.06), "LAPS": (0.28, 0.04),
    "TYRE_STINTS": (0.52, 0.04), "PIT_STOPS": (0.76, 0.04), "WEATHER_OBSERVATIONS": (0.76, 0.23),
}

SIZES = {
    "SEASONS": (0.20, 0.09), "CIRCUITS": (0.26, 0.16), "TEAMS": (0.23, 0.10), "DRIVERS": (0.27, 0.20),
    "RACES": (0.28, 0.27), "SESSIONS": (0.25, 0.21), "RACE_RESULTS": (0.28, 0.29), "LAPS": (0.28, 0.39),
    "TYRE_STINTS": (0.28, 0.31), "PIT_STOPS": (0.22, 0.23), "WEATHER_OBSERVATIONS": (0.22, 0.28),
}

RELATIONSHIPS = [
    ("SEASONS", "RACES", "1", "0..*", "season_year", "CASCADE"),
    ("CIRCUITS", "RACES", "0..1", "0..*", "circuit_id", "NO ACTION"),
    ("RACES", "SESSIONS", "1", "0..*", "race_id", "CASCADE"),
    ("SESSIONS", "RACE_RESULTS", "1", "0..*", "session_id", "CASCADE"),
    ("DRIVERS", "RACE_RESULTS", "1", "0..*", "driver_id", "NO ACTION"),
    ("TEAMS", "RACE_RESULTS", "0..1", "0..*", "team_id", "NO ACTION"),
    ("SESSIONS", "LAPS", "1", "0..*", "session_id", "CASCADE"),
    ("DRIVERS", "LAPS", "1", "0..*", "driver_id", "NO ACTION"),
    ("TEAMS", "LAPS", "0..1", "0..*", "team_id", "NO ACTION"),
    ("SESSIONS", "TYRE_STINTS", "1", "0..*", "session_id", "CASCADE"),
    ("DRIVERS", "TYRE_STINTS", "1", "0..*", "driver_id", "NO ACTION"),
    ("TEAMS", "TYRE_STINTS", "0..1", "0..*", "team_id", "NO ACTION"),
    ("SESSIONS", "PIT_STOPS", "1", "0..*", "session_id", "CASCADE"),
    ("DRIVERS", "PIT_STOPS", "1", "0..*", "driver_id", "NO ACTION"),
    ("SESSIONS", "WEATHER_OBSERVATIONS", "1", "0..*", "session_id", "CASCADE"),
]


def center(name: str) -> tuple[float, float]:
    x, y = POSITIONS[name]
    w, h = SIZES[name]
    return x + w / 2, y + h / 2


def make_er_figure():
    fig, ax = plt.subplots(figsize=(20, 14), facecolor="#121212")
    ax.set_facecolor("#121212")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.035, 0.992, "Paddock F1 - Entity Relationship Model", color="#f5f5f5", fontsize=23, fontweight="bold", va="top")
    ax.text(0.035, 0.965, "Verified PostgreSQL public schema | Crow's Foot-style cardinality labels", color="#b3b3b3", fontsize=10.5, va="top")

    for parent, child, parent_card, child_card, fk, action in RELATIONSHIPS:
        x1, y1 = center(parent)
        x2, y2 = center(child)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-", color="#6b7280", lw=1.15, connectionstyle="arc3,rad=0.0"), zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(x1 * 0.72 + x2 * 0.28, y1 * 0.72 + y2 * 0.28, parent_card, color="#d1d5db", fontsize=7.5, ha="center", va="center", bbox=dict(facecolor="#121212", edgecolor="none", pad=0.25), zorder=2)
        ax.text(x1 * 0.28 + x2 * 0.72, y1 * 0.28 + y2 * 0.72, child_card, color="#d1d5db", fontsize=7.5, ha="center", va="center", bbox=dict(facecolor="#121212", edgecolor="none", pad=0.25), zorder=2)
        ax.text(mx, my, fk, color="#94a3b8", fontsize=6.3, ha="center", va="center", bbox=dict(facecolor="#121212", edgecolor="none", pad=0.2), zorder=2)

    for name, fields in TABLES.items():
        x, y = POSITIONS[name]
        w, h = SIZES[name]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor="#202020", edgecolor="#5b6472", linewidth=1.0, zorder=3))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.032), w, 0.032, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor="#e10600", edgecolor="#e10600", linewidth=0, zorder=4))
        ax.text(x + 0.009, y + h - 0.016, name, color="white", fontsize=8.4, fontweight="bold", va="center", zorder=5)
        text = "\n".join(fields)
        ax.text(x + 0.009, y + h - 0.04, text, color="#e5e7eb", fontsize=5.65, family="DejaVu Sans Mono", va="top", linespacing=1.22, zorder=5)

    ax.text(0.035, 0.015, "Legend: PK primary key | FK foreign key | NN NOT NULL | UQ UNIQUE | parent and child endpoint labels express cardinality. Lines represent enforced FK constraints only.", color="#b3b3b3", fontsize=8)
    return fig


RELATION_TEXT = [
    "SEASONS(year INT PK NN)",
    "CIRCUITS(circuit_id SERIAL PK NN, circuit_key VARCHAR(50) UQ, circuit_name VARCHAR(150) NN, location VARCHAR(100), country VARCHAR(100))",
    "TEAMS(team_id SERIAL PK NN, team_name VARCHAR(100) NN UQ, team_color VARCHAR(10))",
    "DRIVERS(driver_id SERIAL PK NN, abbreviation VARCHAR(10) NN UQ, driver_number INT, full_name VARCHAR(150) NN, broadcast_name VARCHAR(100), country_code VARCHAR(10))",
    "RACES(race_id SERIAL PK NN, season_year INT NN FK -> SEASONS(year) ON DELETE CASCADE, round_number INT NN, event_name VARCHAR(150) NN, official_event_name VARCHAR(255), event_date DATE, event_format VARCHAR(50), circuit_id INT FK -> CIRCUITS(circuit_id); UQ(season_year, round_number); UQ(season_year, event_name))",
    "SESSIONS(session_id SERIAL PK NN, race_id INT NN FK -> RACES(race_id) ON DELETE CASCADE, session_name VARCHAR(50) NN, session_type VARCHAR(50), session_date TIMESTAMPTZ, is_loaded BOOLEAN DEFAULT FALSE; UQ(race_id, session_name))",
    "RACE_RESULTS(result_id SERIAL PK NN, session_id INT NN FK -> SESSIONS(session_id) ON DELETE CASCADE, driver_id INT NN FK -> DRIVERS(driver_id), team_id INT FK -> TEAMS(team_id), position INT, classified_position VARCHAR(10), grid_position INT, time_seconds DOUBLE PRECISION, status VARCHAR(100), points DOUBLE PRECISION, laps_completed INT, q1_seconds DOUBLE PRECISION, q2_seconds DOUBLE PRECISION, q3_seconds DOUBLE PRECISION; UQ(session_id, driver_id))",
    "LAPS(lap_id BIGSERIAL PK NN, session_id INT NN FK -> SESSIONS(session_id) ON DELETE CASCADE, driver_id INT NN FK -> DRIVERS(driver_id), team_id INT FK -> TEAMS(team_id), lap_number INT NN, position INT, lap_time_seconds DOUBLE PRECISION, sector1_time_seconds DOUBLE PRECISION, sector2_time_seconds DOUBLE PRECISION, sector3_time_seconds DOUBLE PRECISION, compound VARCHAR(20), stint INT, tyre_life INT, fresh_tyre BOOLEAN, track_status VARCHAR(20), is_accurate BOOLEAN, is_timed_lap BOOLEAN, is_push_lap BOOLEAN, pit_in_time_seconds DOUBLE PRECISION, pit_out_time_seconds DOUBLE PRECISION; UQ(session_id, driver_id, lap_number))",
    "TYRE_STINTS(stint_id SERIAL PK NN, session_id INT NN FK -> SESSIONS(session_id) ON DELETE CASCADE, driver_id INT NN FK -> DRIVERS(driver_id), team_id INT FK -> TEAMS(team_id), stint_number INT NN, compound VARCHAR(20) NN, start_lap INT NN, end_lap INT NN, stint_length INT NN, mean_lap_seconds DOUBLE PRECISION, fresh_tyre BOOLEAN, mean_tyre_life DOUBLE PRECISION; UQ(session_id, driver_id, stint_number))",
    "PIT_STOPS(pit_stop_id SERIAL PK NN, session_id INT NN FK -> SESSIONS(session_id) ON DELETE CASCADE, driver_id INT NN FK -> DRIVERS(driver_id), lap_number INT NN, pit_in_time_seconds DOUBLE PRECISION, pit_out_time_seconds DOUBLE PRECISION, duration_seconds DOUBLE PRECISION; UQ(session_id, driver_id, lap_number))",
    "WEATHER_OBSERVATIONS(weather_id SERIAL PK NN, session_id INT NN FK -> SESSIONS(session_id) ON DELETE CASCADE, sample_time_seconds DOUBLE PRECISION, air_temp DOUBLE PRECISION, track_temp DOUBLE PRECISION, humidity DOUBLE PRECISION, pressure DOUBLE PRECISION, wind_speed DOUBLE PRECISION, wind_direction INT, rainfall BOOLEAN)",
]


def add_page_header(fig, title: str, subtitle: str):
    fig.text(0.055, 0.955, title, color="#e10600", fontsize=20, fontweight="bold", va="top")
    fig.text(0.055, 0.925, subtitle, color="#4b5563", fontsize=9, va="top")


def add_text_page(
    pdf: PdfPages,
    title: str,
    lines: list[str],
    font_size: float = 9.0,
    wrap_width: int = 133,
    line_height: float = 0.026,
):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    add_page_header(fig, title, "Paddock F1 - verified PostgreSQL relational model")
    y = 0.875
    for line in lines:
        wrapped = wrap(line, width=wrap_width, break_long_words=False, break_on_hyphens=False) or [""]
        for part in wrapped:
            fig.text(0.06, y, part, color="#111827", fontsize=font_size, family="DejaVu Sans Mono", va="top")
            y -= line_height
        y -= line_height * 0.45
        if y < 0.07:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            add_page_header(fig, title + " (continued)", "Paddock F1 - verified PostgreSQL relational model")
            y = 0.875
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_pdf(er_fig):
    pdf_path = OUT / "database_models.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(er_fig, facecolor=er_fig.get_facecolor(), bbox_inches="tight")
        intro = [
            "Paddock F1 - Formula 1 Race Data Management and Analytics System",
            "Database Models Report",
            "",
            "Verification source: live PostgreSQL public catalog queried on 2026-10-22 using the project's configured database connection. The live catalog agrees with db/schema.sql and init_db.py's expected-table list.",
            "",
            "Relationship interpretation:",
            "- SEASONS contains race events. Each RACES row belongs to exactly one season; one season can have zero or many races.",
            "- RACES can optionally reference a CIRCUITS row and contain zero or many SESSIONS rows.",
            "- RACE_RESULTS, LAPS, TYRE_STINTS, and PIT_STOPS are session-driver fact relations. Their unique constraints provide session-scoped business identifiers.",
            "- TEAM references on results, laps, and tyre stints are optional. There is no direct DRIVERS-to-TEAMS foreign key in the implemented database.",
            "- WEATHER_OBSERVATIONS belongs to exactly one session. It is not related to individual laps by a foreign key.",
            "",
            "Delete actions: deleting a season cascades to its races; deleting a race cascades to sessions; deleting a session cascades to results, laps, tyre stints, pit stops, and weather observations. Other verified foreign keys use NO ACTION.",
            "",
            "Important implementation note: PIT_STOPS is an enforced table and relationship in the schema, but save_session_data() currently does not insert into it. This report documents the database design, not an inferred application relationship.",
        ]
        add_text_page(pdf, "Database Model Overview", intro, font_size=10)
        add_text_page(
            pdf,
            "Relational Model - Core Relations",
            ["Reference and event hierarchy relations", ""] + RELATION_TEXT[:6],
            font_size=7.8,
            wrap_width=140,
            line_height=0.024,
        )
        add_text_page(
            pdf,
            "Relational Model - Session Facts",
            ["Session-level analytical and operational relations", ""] + RELATION_TEXT[6:],
            font_size=7.1,
            wrap_width=145,
            line_height=0.022,
        )
        constraints = [
            "Foreign-key inventory",
            "",
        ]
        for parent, child, parent_card, child_card, fk, action in RELATIONSHIPS:
            constraints.append(f"{child}.{fk} -> {parent} | Parent endpoint: {parent_card}; child endpoint: {child_card}; ON DELETE {action}")
        constraints.extend([
            "",
            "Schema notes",
            "- All 11 requested tables are implemented and were verified in the live PostgreSQL catalog.",
            "- No composite primary keys exist. Composite uniqueness constraints exist on races, sessions, race_results, laps, tyre_stints, and pit_stops.",
            "- No check constraints validate domain values such as tyre compound or lap-number ranges.",
            "- PIT_STOPS has no FK to LAPS; a lap association must not be assumed from matching attribute names.",
        ])
        add_text_page(pdf, "Relationships and Constraints", constraints, font_size=7.4, wrap_width=150, line_height=0.021)


def main():
    er_fig = make_er_figure()
    er_fig.savefig(OUT / "er_model.svg", format="svg", facecolor=er_fig.get_facecolor(), bbox_inches="tight")
    er_fig.savefig(OUT / "er_model.png", format="png", dpi=320, facecolor=er_fig.get_facecolor(), bbox_inches="tight")
    make_pdf(er_fig)
    plt.close(er_fig)


if __name__ == "__main__":
    main()
