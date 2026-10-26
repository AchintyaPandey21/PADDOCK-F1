"""Scikit-learn preprocessing and regressor pipelines for lap-time estimation."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["event_name", "driver", "team", "compound"]
NUMERIC_FEATURES = [
    "circuit_ref_time",
    "lap_number",
    "tyre_life",
    "stint_length",
    "track_temp",
    "air_temp",
    "humidity",
    "rainfall",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_FEATURE = "lap_time_seconds"


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer encoding categoricals and scaling numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_linear_regression_pipeline() -> Pipeline:
    """Baseline linear regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def build_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Primary candidate Random Forest regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    # Keep fitting reproducible in constrained local deployments where
                    # process-pool creation is unavailable (for example Streamlit hosts).
                    n_jobs=1,
                ),
            ),
        ]
    )
