"""Consistent display formatting for race timing values stored as seconds."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP


def _format_total_minutes(seconds: float | int) -> str:
    """Render rounded elapsed seconds as MM:SS:mmm with total elapsed minutes."""
    milliseconds = int((Decimal(str(abs(float(seconds)))) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    total_minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{total_minutes:02d}:{whole_seconds:02d}:{milliseconds:03d}"


def format_lap_time(seconds: float | int | None) -> str:
    """Format a lap timing value as MM:SS:mmm without changing its numeric source."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "N/A"
    return _format_total_minutes(seconds)


def format_duration(seconds: float | int | None) -> str:
    """Format elapsed race time as MM:SS:mmm with minutes beyond 59 retained."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "N/A"
    return _format_total_minutes(seconds)


def format_delta(seconds: float | int | None) -> str:
    """Format a signed strategy delta as +/-MM:SS:mmm."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "N/A"
    sign = "-" if float(seconds) < 0 else "+"
    return sign + _format_total_minutes(seconds)
