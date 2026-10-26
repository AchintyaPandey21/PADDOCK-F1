"""Model training and race-separated validation pipeline for Paddock F1."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

from ml.data import load_clean_training_data
from ml.pipeline import (
    ALL_FEATURES,
    TARGET_FEATURE,
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODEL_DIR / "strategy_model.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


def train_and_evaluate(save_artifacts: bool = True) -> dict[str, Any]:
    """Execute race-separated validation, model selection, and artifact serialization.

    Validation Strategy:
      Uses LeaveOneGroupOut grouped by 'event_name' so that every fold tests on
      an entire race event that was never seen during training, preventing data leakage.
    """
    print("=" * 60)
    print("PADDOCK F1 — STRATEGY SIMULATOR MODEL TRAINING")
    print("=" * 60)

    # 1. Load clean lap dataset from PostgreSQL
    print("\n1. Querying and filtering clean training data from PostgreSQL...")
    df, audit = load_clean_training_data()
    print(f"   Raw laps in database: {audit['raw_laps_count']}")
    print(f"   After valid lap times: {audit['filter_steps']['valid_lap_time']}")
    print(f"   After is_accurate flag: {audit['filter_steps']['accurate_laps']}")
    print(f"   After green-flag filter (status='1'): {audit['filter_steps']['green_flag_laps']}")
    print(f"   After non-pit filter: {audit['filter_steps']['non_pit_laps']}")
    print(f"   After known compounds: {audit['filter_steps']['known_compounds']}")
    print(f"   Final usable sample size: {audit['final_usable_sample_size']} laps")

    X = df[ALL_FEATURES]
    y = df[TARGET_FEATURE].values
    groups = df["event_name"].values
    unique_events = sorted(np.unique(groups))
    print(f"\n   Events in dataset ({len(unique_events)}):")
    for ev in unique_events:
        print(f"     - {ev}: {np.sum(groups == ev)} laps")

    # 2. Race-separated validation
    print("\n2. Running Race-Separated Cross-Validation (Leave-One-Race-Out)...")
    logo = LeaveOneGroupOut()

    pipe_lr = build_linear_regression_pipeline()
    pipe_rf = build_random_forest_pipeline(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42)

    lr_fold_metrics = []
    rf_fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        test_event = groups[test_idx[0]]
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # Train Linear Regression baseline
        pipe_lr.fit(X_train, y_train)
        preds_lr = pipe_lr.predict(X_test)
        mae_lr = float(mean_absolute_error(y_test, preds_lr))
        rmse_lr = float(root_mean_squared_error(y_test, preds_lr))
        lr_fold_metrics.append({"event": test_event, "mae": mae_lr, "rmse": rmse_lr, "samples": len(y_test)})

        # Train Random Forest candidate
        pipe_rf.fit(X_train, y_train)
        preds_rf = pipe_rf.predict(X_test)
        mae_rf = float(mean_absolute_error(y_test, preds_rf))
        rmse_rf = float(root_mean_squared_error(y_test, preds_rf))
        rf_fold_metrics.append({"event": test_event, "mae": mae_rf, "rmse": rmse_rf, "samples": len(y_test)})

        print(
            f"   Fold {fold_idx}/6 [{test_event}]: "
            f"LR MAE={mae_lr:.3f}s RMSE={rmse_lr:.3f}s | "
            f"RF MAE={mae_rf:.3f}s RMSE={rmse_rf:.3f}s"
        )

    # 3. Aggregate metrics
    mean_lr_mae = float(np.mean([m["mae"] for m in lr_fold_metrics]))
    mean_lr_rmse = float(np.mean([m["rmse"] for m in lr_fold_metrics]))
    mean_rf_mae = float(np.mean([m["mae"] for m in rf_fold_metrics]))
    mean_rf_rmse = float(np.mean([m["rmse"] for m in rf_fold_metrics]))

    print("\n3. Validation Performance Comparison:")
    print(f"   Linear Regression (Baseline): Mean MAE = {mean_lr_mae:.3f}s | Mean RMSE = {mean_lr_rmse:.3f}s")
    print(f"   Random Forest (Candidate):    Mean MAE = {mean_rf_mae:.3f}s | Mean RMSE = {mean_rf_rmse:.3f}s")

    selected_model_name = "RandomForestRegressor" if mean_rf_mae < mean_lr_mae else "RidgeRegression"
    print(f"\n   -> Selected Model: {selected_model_name} (superior validation performance)")

    # 4. Fit final model on full dataset
    print(f"\n4. Fitting final {selected_model_name} pipeline on all {len(X)} training samples...")
    final_pipeline = (
        build_random_forest_pipeline(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42)
        if selected_model_name == "RandomForestRegressor"
        else build_linear_regression_pipeline()
    )
    final_pipeline.fit(X, y)

    # Calculate training set fit metrics
    train_preds = final_pipeline.predict(X)
    train_mae = float(mean_absolute_error(y, train_preds))
    train_rmse = float(root_mean_squared_error(y, train_preds))
    print(f"   Full-dataset Fit: MAE = {train_mae:.3f}s | RMSE = {train_rmse:.3f}s")

    # 5. Save only data-derived context for inference.
    metadata = {
        "model_name": selected_model_name,
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(X),
        "target": TARGET_FEATURE,
        "features": {
            "all": ALL_FEATURES,
            "categorical": ["event_name", "driver", "team", "compound"],
            "numeric": ["circuit_ref_time", "lap_number", "tyre_life", "stint_length", "track_temp", "air_temp", "humidity", "rainfall"],
        },
        "race_separated_validation": {
            "validation_type": "LeaveOneGroupOut (Race-Separated)",
            "num_folds": len(unique_events),
            "linear_regression": {
                "mean_mae": mean_lr_mae,
                "mean_rmse": mean_lr_rmse,
                "folds": lr_fold_metrics,
            },
            "random_forest": {
                "mean_mae": mean_rf_mae,
                "mean_rmse": mean_rf_rmse,
                "folds": rf_fold_metrics,
            },
        },
        "full_training_fit": {
            "mae": train_mae,
            "rmse": train_rmse,
        },
        "circuit_benchmarks": audit["circuit_benchmarks"],
        "pit_loss_estimates": audit["pit_loss_estimates"],
        "dataset_audit": audit,
    }

    if save_artifacts:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n5. Saving model artifact to {MODEL_FILE}...")
        joblib.dump(final_pipeline, MODEL_FILE)
        print(f"   Saving metadata to {METADATA_FILE}...")
        METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("   Artifacts successfully persisted!")

    return metadata


if __name__ == "__main__":
    train_and_evaluate(save_artifacts=True)
