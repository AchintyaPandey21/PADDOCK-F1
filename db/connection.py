"""PostgreSQL connection management using psycopg."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg

from .config import get_db_config


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """Provide a transactional database connection context manager."""
    config = get_db_config()
    conn = psycopg.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        connect_timeout=10,
    )
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """Test whether the database connection can be established."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            result = cur.fetchone()
            return bool(result and result[0] == 1)
