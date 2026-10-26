"""Unit and integration tests for Paddock F1 ML Strategy Simulator pipeline."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ml.data import VALID_COMPOUNDS, load_clean_training_data
from ml.inference import (
    get_circuit_benchmark,
    get_circuit_pit_loss,
    is_model_available,
    load_metadata,
    load_trained_pipeline,
    predict_laps,
)
from ml.pipeline import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    build_linear_regression_pipeline,
    build_preprocessor,
    build_random_forest_pipeline,
)


class TestMLPipeline(unittest.TestCase):
    """Test suite for data loading, pipeline construction, training, and inference."""

    @classmethod
    def setUpClass(cls):
        """Ensure model artifacts exist before testing inference."""
        if not is_model_available():
            from ml.train import train_and_evaluate
            train_and_evaluate(save_artifacts=True)

    def test_clean_data_loading(self):
        """Verify that training data loaded from PostgreSQL meets all quality criteria."""
        df, audit = load_clean_training_data()
        self.assertGreater(len(df), 4000, "Clean dataset should contain >4,000 laps")
        self.assertEqual(len(df), audit["final_usable_sample_size"])

        # No missing values in target or key features
        self.assertFalse(df["lap_time_seconds"].isna().any(), "Lap times must not contain NaNs")
        self.assertFalse(df["tyre_life"].isna().any(), "Tyre life must not contain NaNs")
        self.assertFalse(df["stint_length"].isna().any(), "Stint length must not contain NaNs")
        self.assertFalse(df["compound"].isna().any(), "Compound must not contain NaNs")
        self.assertTrue((df["lap_time_seconds"] > 0).all(), "Lap times must be positive")
        self.assertTrue((df["tyre_life"] > 0).all(), "Tyre life must be > 0")
        self.assertTrue((df["stint_length"] > 0).all(), "Stint length must be > 0")

        # Compounds must all be valid F1 compounds
        unique_compounds = set(df["compound"].unique())
        self.assertTrue(unique_compounds.issubset(set(VALID_COMPOUNDS)))

        # Must include all 6 circuits
        self.assertEqual(len(audit["events"]), 6)

    def test_pipeline_construction(self):
        """Verify pipeline instantiation and ColumnTransformer schema."""
        preprocessor = build_preprocessor()
        self.assertIsNotNone(preprocessor)

        lr_pipe = build_linear_regression_pipeline()
        self.assertIsNotNone(lr_pipe)
        self.assertEqual(len(lr_pipe.steps), 2)

        rf_pipe = build_random_forest_pipeline(n_estimators=10)
        self.assertIsNotNone(rf_pipe)
        self.assertEqual(len(rf_pipe.steps), 2)

    def test_model_artifacts_and_metadata(self):
        """Verify that saved model artifact and metadata adhere to conventions."""
        self.assertTrue(is_model_available(), "Trained model and metadata files must exist")

        metadata = load_metadata()
        self.assertIn("model_name", metadata)
        self.assertIn("training_samples", metadata)
        self.assertIn("race_separated_validation", metadata)
        self.assertIn("circuit_benchmarks", metadata)
        self.assertIn("pit_loss_estimates", metadata)
        self.assertIn("observed_tyre_age_by_compound", metadata["dataset_audit"])

        # Verify validation metrics structure
        val_info = metadata["race_separated_validation"]
        self.assertEqual(val_info["num_folds"], 6)
        self.assertGreater(val_info["linear_regression"]["mean_mae"], 0.0)
        self.assertGreater(val_info["random_forest"]["mean_mae"], 0.0)
        self.assertGreater(val_info["linear_regression"]["mean_rmse"], 0.0)
        self.assertGreater(val_info["random_forest"]["mean_rmse"], 0.0)

    def test_circuit_benchmarks_and_pit_losses(self):
        """Verify circuit benchmark and pit loss lookup helpers."""
        benchmark = get_circuit_benchmark("Italian Grand Prix")
        self.assertGreater(benchmark, 70.0)
        self.assertLess(benchmark, 120.0)

        pit_loss = get_circuit_pit_loss("Italian Grand Prix")
        self.assertGreaterEqual(pit_loss, 0.0)

    def test_inference_predictions(self):
        """Verify that model generates realistic lap times on synthetic test data."""
        pipeline = load_trained_pipeline()

        test_laps = pd.DataFrame(
            [
                {
                    "event_name": "Italian Grand Prix",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "compound": "SOFT",
                    "circuit_ref_time": 87.0,
                    "lap_number": 10,
                    "tyre_life": 5.0,
                    "stint_length": 15.0,
                    "track_temp": 45.0,
                    "air_temp": 30.0,
                    "humidity": 40.0,
                    "rainfall": 0,
                },
                {
                    "event_name": "Italian Grand Prix",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "compound": "HARD",
                    "circuit_ref_time": 87.0,
                    "lap_number": 10,
                    "tyre_life": 5.0,
                    "stint_length": 15.0,
                    "track_temp": 45.0,
                    "air_temp": 30.0,
                    "humidity": 40.0,
                    "rainfall": 0,
                },
                {
                    "event_name": "Italian Grand Prix",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "compound": "SOFT",
                    "circuit_ref_time": 87.0,
                    "lap_number": 10,
                    "tyre_life": 25.0,
                    "stint_length": 30.0,
                    "track_temp": 45.0,
                    "air_temp": 30.0,
                    "humidity": 40.0,
                    "rainfall": 0,
                },
            ]
        )

        preds = predict_laps(pipeline, test_laps)
        self.assertEqual(len(preds), 3)

        soft_fresh = preds[0]
        hard_fresh = preds[1]
        soft_worn = preds[2]

        # Predicted times must be realistic for Monza (~75-105s)
        for p in preds:
            self.assertGreater(p, 75.0, f"Lap time {p} too low for Monza")
            self.assertLess(p, 105.0, f"Lap time {p} too high for Monza")

        # Test fuel burn sensitivity: later lap in race should be faster than early lap for same tyre life
        later_lap = pd.DataFrame(
            [
                {
                    "event_name": "Italian Grand Prix",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "compound": "SOFT",
                    "circuit_ref_time": 87.0,
                    "lap_number": 45,
                    "tyre_life": 5.0,
                    "stint_length": 15.0,
                    "track_temp": 45.0,
                    "air_temp": 30.0,
                    "humidity": 40.0,
                    "rainfall": 0,
                }
            ]
        )
        late_pred = predict_laps(pipeline, later_lap)[0]
        self.assertLess(
            late_pred,
            soft_fresh,
            f"Later lap with lighter fuel ({late_pred:.2f}s) should be faster than early lap ({soft_fresh:.2f}s)",
        )

    def test_missing_features_error(self):
        """Verify that inference raises ValueError if required features are omitted."""
        pipeline = load_trained_pipeline()
        incomplete_df = pd.DataFrame([{"event_name": "Italian Grand Prix", "driver": "VER"}])
        with self.assertRaises(ValueError):
            predict_laps(pipeline, incomplete_df)


if __name__ == "__main__":
    unittest.main()
