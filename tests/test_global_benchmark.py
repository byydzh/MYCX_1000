import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tuner.global_benchmark import run_global_benchmark, select_benchmark_event_ids
from tuner.offline_evaluator import OfflineEvaluator
from tuner.train import split_formal_event_ids, split_formal_training_event_ids


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


class TestGlobalBenchmark(unittest.TestCase):
    def test_split_formal_event_ids_respects_300_and_301_boundary(self):
        train_ids, holdout_ids = split_formal_event_ids(
            [198, 199, 200, 250, 300, 301, 312],
        )

        self.assertEqual(train_ids, [200, 250, 300])
        self.assertEqual(holdout_ids, [301, 312])

    def test_split_formal_training_event_ids_keeps_external_holdout_outside_validation(self):
        train_ids, validation_ids, external_holdout_ids = split_formal_training_event_ids(
            [219, 223, 227, 228, 229, 230, 231, 299, 300, 301, 312],
            train_ratio=0.75,
            evaluator=None,
            min_test_events=2,
        )

        self.assertTrue(all(event_id <= 300 for event_id in train_ids))
        self.assertTrue(all(event_id <= 300 for event_id in validation_ids))
        self.assertEqual(external_holdout_ids, [301, 312])

    def test_select_benchmark_event_ids_splits_train_and_holdout_ranges(self):
        cached_events = {
            event_id: _make_cached_event(event_id, speed_offset=(event_id - 290) * 0.1)
            for event_id in range(295, 305)
        }
        evaluator = OfflineEvaluator(cached_events)

        eligible_ids, train_ids, holdout_ids = select_benchmark_event_ids(
            evaluator,
            similar_count=5,
            ignore_ids=[],
            min_event_id=200,
            train_max_event_id=300,
            holdout_min_event_id=301,
        )

        self.assertEqual(eligible_ids, [300, 301, 302, 303, 304])
        self.assertEqual(train_ids, [300])
        self.assertEqual(holdout_ids, [301, 302, 303, 304])

    def test_run_global_benchmark_evaluates_requested_presets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events_dir = Path(tmpdir) / "events"
            events_dir.mkdir(parents=True, exist_ok=True)

            for event_id in range(295, 305):
                payload = _make_cached_event(event_id, speed_offset=(event_id - 295) * 0.2)
                write_payload = payload.copy()
                write_payload.pop("dataframe", None)
                with (events_dir / f"{event_id}.json").open("w", encoding="utf-8") as f:
                    json.dump(write_payload, f, ensure_ascii=False, indent=2)

            with patch("tuner.global_benchmark.list_models", return_value=[{"id": "demo_model"}]), patch(
                "tuner.global_benchmark.list_presets",
                return_value=[{"id": "baseline"}, {"id": "alt"}],
            ), patch(
                "tuner.global_benchmark.load_preset",
                side_effect=lambda model_id, preset_id: {
                    "similar_count": 5,
                    "ignore_event_ids": [],
                    "weekend_multiplier": 1.1 if preset_id == "baseline" else 1.2,
                },
            ):
                result = run_global_benchmark(
                    cache_root=Path(tmpdir),
                    prepare_cache=False,
                    model_ids=["demo_model"],
                    preset_ids_by_model={"demo_model": ["baseline", "alt"]},
                    benchmark_ignore_ids=[],
                    save_report=False,
                    curve_sample_count=0,
                )

        self.assertEqual(result["train_ids"], [300])
        self.assertEqual(result["holdout_ids"], [301, 302, 303, 304])
        self.assertEqual(len(result["leaderboard"]), 2)
        self.assertEqual([item["preset_id"] for item in result["leaderboard"]], ["baseline", "alt"])
        self.assertTrue(all(item["holdout_relative_mse"] is not None for item in result["leaderboard"]))


if __name__ == "__main__":
    unittest.main()
