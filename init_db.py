"""Database initialization script for Paddock F1.

Creates the configured PostgreSQL database when needed, then executes schema.sql
without dropping existing tables or data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg import sql

from db.config import validate_db_config
from db.connection import get_db_connection

EXPECTED_TABLES = [
    "seasons",
    "circuits",
    "teams",
    "drivers",
    "races",
    "sessions",
    "race_results",
    "laps",
    "tyre_stints",
    "pit_stops",
    "weather_observations",
]


def ensure_database_exists(config: dict[str, object]) -> None:
    """Create the configured database when it does not already exist."""
    database_name = str(config["dbname"])
    maintenance_config = {
        "host": config["host"],
        "port": config["port"],
        "dbname": "postgres",
        "user": config["user"],
        "password": config["password"],
        "connect_timeout": 10,
        "autocommit": True,
    }

    with psycopg.connect(**maintenance_config) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (database_name,))
            if cur.fetchone():
                print(f"Database '{database_name}' already exists.")
                return

            # Identifiers cannot be query parameters; psycopg quotes the name safely.
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            print(f"Created database '{database_name}'.")


def initialize_database() -> bool:
    """Initialize PostgreSQL tables from db/schema.sql."""
    print("Validating database configuration...")
    try:
        config = validate_db_config()
        print(f"Target Database: {config['dbname']} on {config['host']}:{config['port']} (user: {config['user']})")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False

    try:
        print("Ensuring target database exists...")
        ensure_database_exists(config)

        schema_file = Path(__file__).resolve().parent / "db" / "schema.sql"
        if not schema_file.exists():
            print(f"ERROR: Schema file not found at {schema_file}", file=sys.stderr)
            return False

        schema_sql = schema_file.read_text(encoding="utf-8")

        print("Connecting to target database...")
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                print("Executing schema DDL (CREATE TABLE IF NOT EXISTS)...")
                cur.execute(schema_sql)
            conn.commit()

            # Verify table creation
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    """
                )
                existing_tables = {row[0] for row in cur.fetchall()}

            missing = [t for t in EXPECTED_TABLES if t not in existing_tables]
            if missing:
                print(f"WARNING: The following tables were not found: {missing}", file=sys.stderr)
                return False

            print("Database initialized successfully!")
            print(f"Verified {len(existing_tables.intersection(EXPECTED_TABLES))} tables:")
            for table in sorted(existing_tables.intersection(EXPECTED_TABLES)):
                print(f"  - {table}")
            return True

    except Exception as exc:
        print(f"Database initialization failed: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = initialize_database()
    sys.exit(0 if success else 1)
