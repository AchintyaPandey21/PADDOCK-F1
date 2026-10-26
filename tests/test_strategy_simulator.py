"""Focused behavioural tests for the what-if strategy simulation engine."""

import unittest

import math

from db.repository import load_session_data
from ml.inference import load_trained_pipeline
from ml.simulator import build_stints_from_pit_laps, get_recorded_baseline, run_strategy_simulation
from pages.strategy_simulator import _comparison_table
from utils.time_format import format_delta, format_duration, format_lap_time


class TestStrategySimulator(unittest.TestCase):
    """Validate scenario construction and a real PostgreSQL-backed simulation."""

    def test_stint_plan_covers_the_complete_race(self):
        stints = build_stints_from_pit_laps([18, 38], ["MEDIUM", "HARD", "SOFT"], 57)
        self.assertEqual([(s.start_lap, s.end_lap) for s in stints], [(1, 18), (19, 38), (39, 57)])
        self.assertEqual(sum(s.stint_length for s in stints), 57)

    def test_invalid_pit_plan_is_rejected(self):
        with self.assertRaises(ValueError):
            build_stints_from_pit_laps([20, 20], ["MEDIUM", "HARD", "SOFT"], 57)

    def test_postgres_backed_counterfactual_keeps_recorded_baseline(self):
        context = load_session_data(2026, "Bahrain Grand Prix", "Race")
        dataset = context["dataset"]
        driver = dataset["drivers"][0]
        result = run_strategy_simulation(
            dataset,
            context["metadata"],
            driver,
            [20],
            ["MEDIUM", "HARD"],
            load_trained_pipeline(),
        )
        self.assertEqual(len(result.simulated_laps), result.total_laps)
        self.assertEqual(result.simulated_laps["LapNumber"].iloc[0], 1)
        self.assertEqual(result.simulated_laps["LapNumber"].iloc[-1], result.total_laps)
        valid = result.simulated_laps.dropna(subset=["ActualLapTime", "SimulatedLapTime"])
        self.assertGreater(len(valid), 0)
        self.assertTrue(valid["ActualLapTime"].gt(0).all())
        self.assertTrue(valid["SimulatedLapTime"].gt(0).all())
        self.assertEqual(len(valid), result.summary["comparison_lap_count"])
        self.assertEqual(int(result.simulated_laps.loc[result.simulated_laps["LapNumber"] == 21, "TyreLife"].iloc[0]), 1)
        self.assertAlmostEqual(
            float(result.simulated_laps["AssumedPitLoss"].sum()),
            float(result.summary["simulated_pit_loss"]),
            places=3,
        )
        self.assertAlmostEqual(
            float(result.simulated_laps["ModelledActualPitLoss"].sum()),
            float(result.summary["modelled_actual_pit_loss"]),
            places=3,
        )
        self.assertNotIn("SimulatedPosition", result.simulated_laps.columns)

    def test_recorded_baseline_uses_stored_lap_data(self):
        context = load_session_data(2026, "Singapore Grand Prix", "Race")
        baseline = get_recorded_baseline(context["dataset"], "LEC")
        self.assertEqual(baseline["total_laps"], 62)
        self.assertGreater(baseline["valid_laps"], 0)
        self.assertGreater(baseline["total_time"], 0)
        self.assertGreater(baseline["average_lap"], 0)
        self.assertEqual(baseline["pit_stops"], 2)

    def test_recorded_strategy_replay_is_a_zero_delta_sanity_check(self):
        context = load_session_data(2026, "Singapore Grand Prix", "Race")
        result = run_strategy_simulation(
            context["dataset"],
            context["metadata"],
            "LEC",
            [24, 39],
            ["SOFT", "MEDIUM", "SOFT"],
            load_trained_pipeline(),
        )
        self.assertTrue(result.summary["matches_recorded_plan"])
        self.assertAlmostEqual(result.summary["net_time_delta"], 0.0, places=3)

    def test_comparison_table_aligns_recorded_and_simulated_values(self):
        context = load_session_data(2026, "Singapore Grand Prix", "Race")
        result = run_strategy_simulation(
            context["dataset"],
            context["metadata"],
            "LEC",
            [24, 39],
            ["SOFT", "MEDIUM", "SOFT"],
            load_trained_pipeline(),
        )
        comparison = _comparison_table(result)
        self.assertEqual(list(comparison.columns), ["Metric", "Actual (Recorded)", "What-if (Simulated)", "Difference"])
        self.assertEqual(comparison.loc[0, "Actual (Recorded)"], "102:08:120")
        self.assertEqual(comparison.loc[0, "Difference"], "+00:00:000")
        self.assertEqual(comparison.loc[4, "Difference"], "Not directly comparable")

    def test_timing_formatters(self):
        self.assertEqual(format_lap_time(0), "00:00:000")
        self.assertEqual(format_lap_time(60), "01:00:000")
        self.assertEqual(format_lap_time(98.45), "01:38:450")
        self.assertEqual(format_lap_time(102.135), "01:42:135")
        self.assertEqual(format_lap_time(59.9996), "01:00:000")
        self.assertEqual(format_duration(3723.12), "62:03:120")
        self.assertEqual(format_duration(3600), "60:00:000")
        self.assertEqual(format_delta(3.685), "+00:03:685")
        self.assertEqual(format_delta(-2.15), "-00:02:150")
        self.assertEqual(format_lap_time(None), "N/A")
        self.assertEqual(format_duration(math.nan), "N/A")
        self.assertEqual(format_delta(math.inf), "N/A")

    def test_rejects_wet_tyres_for_a_dry_recorded_race(self):
        context = load_session_data(2026, "Singapore Grand Prix", "Race")
        with self.assertRaisesRegex(ValueError, "no recorded rainfall"):
            run_strategy_simulation(
                context["dataset"],
                context["metadata"],
                "LEC",
                [31],
                ["MEDIUM", "WET"],
                load_trained_pipeline(),
            )


if __name__ == "__main__":
    unittest.main()
