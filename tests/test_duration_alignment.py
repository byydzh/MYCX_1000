import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_models import EventData, EventMeta
from prediction_engine import PredictionEngine


class DummySeasonality:
    tz_offset = 8

    def remove_seasonality(self, df):
        cleaned = df.copy()
        if "skeleton_speed" not in cleaned.columns:
            cleaned["skeleton_speed"] = cleaned["norm_speed"]
        return cleaned

    def apply_seasonality(self, future_t, skeleton_pred, start_at, total_hours=None, t_panic=None):
        return skeleton_pred, np.ones_like(skeleton_pred)


class DummyModeler:
    def fit(self, t, v, total_hours):
        return np.array([0.02, 0.01, 0.001, 0.1, 24.0])

    def shape_function(self, t, Base, A, B, B_end, T_panic, T_total):
        t = np.asarray(t, dtype=float)
        return Base + A * t + B * (t ** 2)


def _build_event(event_id: int, total_hours: float, norm_speed: float = 0.02) -> EventData:
    hours = np.arange(0.0, min(total_hours, 120.0) + 6.0, 6.0)
    times = (hours * 3600 * 1000).astype(np.int64)
    speed_pt_min = np.full(hours.shape, norm_speed * 1000.0)
    values = 100000.0 + np.cumsum(speed_pt_min * 60.0 * 6.0)

    df = pd.DataFrame(
        {
            "time": times,
            "value": values,
            "hours_elapsed": hours,
            "speed": speed_pt_min,
            "norm_speed": np.full(hours.shape, norm_speed),
            "skeleton_speed": np.full(hours.shape, norm_speed),
        }
    )
    meta = EventMeta(
        event_id=event_id,
        event_type="medley",
        start_at=0,
        end_at=int(total_hours * 3600 * 1000),
        aggregate_at=int(total_hours * 3600 * 1000),
    )
    return EventData(meta=meta, df=df, scale=1000.0)


class TestDurationAlignment(unittest.TestCase):
    def setUp(self):
        self.engine = PredictionEngine(
            DummySeasonality(),
            DummyModeler(),
            config={
                "t_start_cmp": 6.0,
                "t_end_cap": 72.0,
                "duration_align_log_full_weight": 0.18,
                "duration_window_align_max_weight": 0.75,
                "duration_param_align_max_weight": 0.85,
                "refit_min_points": 9999,
            },
        )

    def test_duration_mismatch_weight_is_symmetric_and_zero_at_equal(self):
        equal_weight = self.engine._get_duration_mismatch_weight(192.0, 192.0, "duration_param_align_max_weight")
        long_weight = self.engine._get_duration_mismatch_weight(226.0, 192.0, "duration_param_align_max_weight")
        short_weight = self.engine._get_duration_mismatch_weight(192.0, 226.0, "duration_param_align_max_weight")

        self.assertEqual(equal_weight, 0.0)
        self.assertGreater(long_weight, 0.0)
        self.assertAlmostEqual(long_weight, short_weight, places=9)

    def test_progress_window_mapping_scales_with_duration(self):
        mapped_start, mapped_end = self.engine._map_window_by_progress(
            35.0,
            109.75,
            source_total_hours=226.0,
            target_total_hours=178.0,
        )

        self.assertAlmostEqual(mapped_start, 27.5664, places=3)
        self.assertAlmostEqual(mapped_end, 86.4403, places=3)

    def test_predict_applies_continuous_duration_alignment_below_old_threshold(self):
        target = _build_event(401, total_hours=210.0, norm_speed=0.02)
        history = [
            _build_event(400, total_hours=192.0, norm_speed=0.02),
            _build_event(399, total_hours=192.0, norm_speed=0.02),
        ]

        result = self.engine.predict(target, history)

        self.assertLess(result.used_params[1], 0.01)
        self.assertLess(result.used_params[2], 0.001)


if __name__ == "__main__":
    unittest.main()
