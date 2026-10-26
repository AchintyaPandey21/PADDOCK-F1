"""Database configuration and environment variable validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_environment() -> None:
    """Load project configuration without overriding externally supplied values."""
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH)
    else:
        load_dotenv()


_load_environment()

REQUIRED_DB_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]


def validate_db_config() -> dict[str, Any]:
    """Validate that all required database environment variables are set.

    Raises:
        ValueError: If any required database variable is missing or empty.
    """
    # Re-read the project file so a running Streamlit process can pick up a
    # newly added .env file on its next script execution.
    _load_environment()
    missing = [var for var in REQUIRED_DB_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(
            "Database configuration is incomplete. "
            "Please configure DB_HOST, DB_PORT, DB_NAME, DB_USER and DB_PASSWORD."
        )

    try:
        port = int(os.environ["DB_PORT"])
    except ValueError as exc:
        raise ValueError("DB_PORT must be a valid integer port number.") from exc

    return {
        "host": os.environ["DB_HOST"].strip(),
        "port": port,
        "dbname": os.environ["DB_NAME"].strip(),
        "user": os.environ["DB_USER"].strip(),
        "password": os.environ["DB_PASSWORD"],
    }


def is_db_configured() -> bool:
    """Return True if all required database environment variables are present."""
    return all(bool(os.getenv(var)) for var in REQUIRED_DB_VARS)


def get_db_config() -> dict[str, Any]:
    """Return the validated database configuration dictionary."""
    return validate_db_config()
