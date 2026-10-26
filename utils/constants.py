"""Project-wide constants and lightweight configuration values."""

from __future__ import annotations

from collections import OrderedDict

MIN_SEASON_YEAR = 2018
CACHE_DIR = "cache"
APP_TITLE = "Paddock F1"
APP_SUBTITLE = (
    "Professional-grade Formula 1 session analytics powered by FastF1, "
    "designed for fast comparisons across pace, strategy, telemetry, and race flow."
)

SESSION_OPTIONS = OrderedDict(
    [
        ("FP1", "Free Practice 1"),
        ("FP2", "Free Practice 2"),
        ("FP3", "Free Practice 3"),
        ("Q", "Qualifying"),
        ("Sprint", "Sprint"),
        ("Race", "Race"),
    ]
)

PAGE_OPTIONS = OrderedDict(
    [
        ("Race Centre", "overview"),
        ("Driver Analytics", "driver_analytics"),
        ("Tyre Strategy", "tyre_strategy"),
        ("Telemetry Lab", "telemetry"),
    ]
)

DASH_PATTERNS = [
    "solid",
    "dash",
    "dot",
    "longdash",
    "dashdot",
    "longdashdot",
]

MARKER_SYMBOLS = [
    "circle",
    "diamond",
    "square",
    "x",
    "triangle-up",
    "hexagon",
]

FALLBACK_DRIVER_COLORS = [
    "#E10600",
    "#3B82F6",
    "#FFC857",
    "#10B981",
    "#EC4899",
    "#8B5CF6",
    "#06B6D4",
    "#F97316",
]

COMPOUND_FALLBACKS = {
    "SOFT": "#ff4d4f",
    "MEDIUM": "#ffd84d",
    "HARD": "#fafafa",
    "INTERMEDIATE": "#43d17a",
    "WET": "#2d76ff",
    "UNKNOWN": "#8d99ae",
}
