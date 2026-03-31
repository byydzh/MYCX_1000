import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG
from tuner.offline_evaluator import OfflineEvaluator
from tuner.train import build_snapshot_plan, run_training, split_event_ids


def _make_cached_event(event_id: int, speed_offset: float = 0.0) -> dict:
    hours = np.arange(0.0, 126.0, 6.0)
    times = (hours * 3600 * 1000).astype(np.int64)
    speed_pt_min = np.full(hours.shape, 18.0 + speed_offset)
    values = 100000.0 + np.cumsum(speed_pt_min * 60.0 * 6.0)
    scale = 1000.0

    dataframe = pd.DataFrame(
        {
            "time": times,
            "value": values,
            "speed": speed_pt_min,
            "norm_speed": speed_pt_min / scale,
        }
    )

    return {
        "event_id": event_id,
        "meta": {
            "event_id": event_id,
            "event_type": "medley",
            "start_at": 0,
            "end_at": int(192 * 3600 * 1000),
            "aggregate_at": int(192 * 3600 * 1000),
        },
        "event_type": "medley",
        "scale": scale,
        "actual_final_score": float(values[-1]),
        "dataframe": dataframe,
        "df_records": dataframe.to_dict(orient="records"),
        "record_count": len(dataframe),
    }


class TestOfflineEvaluator(unittest.TestCase):
    def setUp(self):
        self.cached_events = {
            event_id: _make_cached_event(event_id, speed_offset=(event_id - 100) * 0.3)
            for event_id in range(100, 107)
        }
        self.evaluator = OfflineEvaluator(self.cached_events)

    def test_history_selection_uses_only_earlier_events(self):
        history_ids = self.evaluator._select_history_event_ids(106, "medley", 5)

        self.assertEqual(len(history_ids), 5)
        self.assertTrue(all(event_id < 106 for event_id in history_ids))
        self.assertEqual(history_ids, [105, 104, 103, 102, 101])

    def test_predict_event_returns_offline_metrics(self):
        result = self.evaluator.predict_event(
            106,
            config=DEFAULT_CONFIG,
            debug_hours=24.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["event_id"], 106)
        self.assertEqual(result["debug_hours"], 24.0)
        self.assertTrue(all(event_id < 106 for event_id in result["history_ids"]))
        self.assertIn("predicted_final", result)
        self.assertIn("actual_final", result)

    def test_eligible_event_ids_require_enough_prior_history(self):
        eligible_ids = self.evaluator.eligible_event_ids(similar_count=5)

        self.assertEqual(eligible_ids, [105, 106])

    def test_evaluate_accepts_per_event_snapshot_plan(self):
        snapshot_plan = {
            105: [24.0],
            106: [24.0, 48.0],
        }

        metrics = self.evaluator.evaluate(
            DEFAULT_CONFIG,
            event_ids=[105, 106],
            snapshot_plan=snapshot_plan,
        )

        self.assertEqual(metrics["count"], 3)
        self.assertEqual(
            sorted((row["event_id"], row["debug_hours"]) for row in metrics["rows"]),
            [(105, 24.0), (106, 24.0), (106, 48.0)],
        )

    def test_curve_loss_metrics_are_reported_when_enabled(self):
        metrics = self.evaluator.evaluate(
            DEFAULT_CONFIG,
            event_ids=[105, 106],
            snapshot_plan={105: [24.0], 106: [24.0]},
            curve_sample_count=2,
            curve_loss_weight=0.5,
        )

        self.assertIsNotNone(metrics["curve_relative_mse"])
        self.assertGreaterEqual(metrics["objective_loss"], metrics["relative_mse"])
        self.assertTrue(all("curve_relative_mse" in row for row in metrics["rows"]))

    def test_build_snapshot_plan_is_deterministic_and_bucketed(self):
        snapshot_plan_a = build_snapshot_plan(
            self.evaluator,
            [105, 106],
            seed=42,
        )
        snapshot_plan_b = build_snapshot_plan(
            self.evaluator,
            [105, 106],
            seed=42,
        )

        self.assertEqual(snapshot_plan_a, snapshot_plan_b)
        self.assertEqual(len(snapshot_plan_a[105]), 5)
        self.assertTrue(all(hour >= 18.0 for hour in snapshot_plan_a[105]))
        self.assertTrue(all(hour <= 192.0 * 0.90 for hour in snapshot_plan_a[105]))

    def test_split_event_ids_prefers_tail_with_better_type_coverage(self):
        typed_events = {}
        event_types = {
            201: "type_a",
            202: "type_b",
            203: "type_a",
            204: "type_b",
            205: "type_c",
            206: "type_a",
            207: "type_b",
            208: "type_c",
        }
        for event_id, event_type in event_types.items():
            payload = _make_cached_event(event_id)
            payload["event_type"] = event_type
            payload["meta"]["event_type"] = event_type
            typed_events[event_id] = payload

        typed_evaluator = OfflineEvaluator(typed_events)
        train_ids, test_ids = split_event_ids(
            typed_events.keys(),
            train_ratio=0.75,
            evaluator=typed_evaluator,
            min_test_events=2,
        )

        self.assertEqual(train_ids, [201, 202, 203, 204, 205])
        self.assertEqual(test_ids, [206, 207, 208])

    def test_run_training_works_from_existing_cache_without_network(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events_dir = Path(tmpdir) / "events"
            events_dir.mkdir(parents=True, exist_ok=True)

            for event_id, payload in self.cached_events.items():
                write_payload = payload.copy()
                write_payload.pop("dataframe", None)
                with (events_dir / f"{event_id}.json").open("w", encoding="utf-8") as f:
                    json.dump(write_payload, f, ensure_ascii=False, indent=2)

            result = run_training(
                cache_root=Path(tmpdir),
                prepare_cache=False,
                train_ids=[105],
                test_ids=[106],
                debug_hours_list=[24.0],
                maxiter=1,
                popsize=2,
                seed=123,
                save_report=False,
                output_preset_path=Path(tmpdir) / "learned_test.json",
            )

            preset_exists = Path(result["preset_path"]).exists()

        self.assertIn("best_config", result)
        self.assertTrue(result["objective_history"])
        self.assertIsNotNone(result["test_metrics"])
        self.assertTrue(preset_exists)


if __name__ == "__main__":
    unittest.main()
