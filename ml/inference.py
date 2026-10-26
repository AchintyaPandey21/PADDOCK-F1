"""Model loading and inference utilities for Paddock F1 Strategy Simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.pipeline import ALL_FEATURES

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODEL_DIR / "strategy_model.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

_CACHED_PIPELINE: Pipeline | None = None
_CACHED_METADATA: dict[str, Any] | None = None


def is_model_available() -> bool:
    """Check if the trained model artifact and metadata exist on disk."""
    return MODEL_FILE.exists() and METADATA_FILE.exists()


def load_metadata() -> dict[str, Any]:
    """Load model training metadata and validation metrics."""
    global _CACHED_METADATA
    if _CACHED_METADATA is not None:
        return _CACHED_METADATA

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Model metadata not found at {METADATA_FILE}. Run `python -m ml.train` first.")

    _CACHED_METADATA = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return _CACHED_METADATA


def load_trained_pipeline() -> Pipeline:
    """Load the trained Scikit-learn strategy pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None:
        return _CACHED_PIPELINE

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_FILE}. Run `python -m ml.train` first.")

    _CACHED_PIPELINE = joblib.load(MODEL_FILE)
    return _CACHED_PIPELINE


def get_circuit_benchmark(event_name: str) -> float:
    """Retrieve the median reference lap time for a given Grand Prix event."""
    metadata = load_metadata()
    benchmarks = metadata.get("circuit_benchmarks", {})
    return float(benchmarks.get(event_name, 90.0))


def get_circuit_pit_loss(event_name: str) -> float:
    """Retrieve a data-derived median pit-lap penalty for an event."""
    metadata = load_metadata()
    estimates = metadata.get("pit_loss_estimates", {})
    event_estimate = estimates.get(event_name) or estimates.get("default", {})
    return float(event_estimate.get("seconds", 0.0))


def get_pit_loss_estimate(event_name: str) -> dict[str, float | int]:
    """Return the saved data-derived pit-loss estimate and supporting sample count."""
    metadata = load_metadata()
    estimates = metadata.get("pit_loss_estimates", {})
    estimate = estimates.get(event_name) or estimates.get("default", {})
    return {
        "seconds": float(estimate.get("seconds", 0.0)),
        "samples": int(estimate.get("samples", 0)),
        "source": "event" if event_name in estimates else "historical default",
    }


def predict_laps(pipeline: Pipeline, df_laps: pd.DataFrame) -> np.ndarray:
    """Predict lap times (in seconds) for a dataframe of lap configurations.

    Requires columns:
      - event_name, driver, team, compound, circuit_ref_time,
        lap_number, tyre_life, stint_length, track_temp, air_temp, humidity, rainfall
    """
    for col in ALL_FEATURES:
        if col not in df_laps.columns:
            raise ValueError(f"Missing required feature '{col}' for model prediction.")

    X = df_laps[ALL_FEATURES].copy()
    predictions = pipeline.predict(X)
    return np.array(predictions, dtype=float)
