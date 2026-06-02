import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_source import BandoriDataSource
from domain_models import EventData, EventMeta
from prediction_engine import PredictionEngine


class DummySeasonality:
    def remove_seasonality(self, df):
        cleaned = df.copy()
        cleaned["skeleton_speed"] = cleaned["norm_speed"]
        return cleaned

    def apply_seasonality(self, future_t, skeleton_pred, start_at, total_hours=None, t_panic=None):
        return skeleton_pred, np.ones_like(skeleton_pred)


class DummyModeler:
    def fit(self, t, v, total_hours):
        return np.array([0.02, 0.0, 0.0, 0.1, 24.0])

    def shape_function(self, t, Base, A, B, B_end, T_panic, T_total):
        return np.full_like(np.asarray(t, dtype=float), Base)


def _target_event() -> EventData:
    hours = np.arange(0.0, 24.0, 6.0)
    df = pd.DataFrame(
        {
            "time": (hours * 3600 * 1000).astype(np.int64),
            "value": np.arange(len(hours), dtype=float) * 1000.0,
            "hours_elapsed": hours,
            "speed": np.full(len(hours), 20.0),
            "norm_speed": np.full(len(hours), 0.02),
        }
    )
    meta = EventMeta(
        event_id=1,
        event_type="challenge",
        start_at=0,
        end_at=int(100 * 3600 * 1000),
        aggregate_at=int(100 * 3600 * 1000),
    )
    return EventData(meta=meta, df=df, scale=1000.0, tier=1500)


class TestBaselineFallback(unittest.TestCase):
    def test_predict_rejects_missing_history_prior(self):
        engine = PredictionEngine(DummySeasonality(), DummyModeler(), config={"refit_min_points": 9999})

        with self.assertRaisesRegex(ValueError, "缺少有效历史先验"):
            engine.predict(_target_event(), [])

    def test_interpolated_tier_dataframe_uses_adjacent_rank_curves(self):
        ds = BandoriDataSource(api_source="hhwx", server_index=3)
        lower = pd.DataFrame({"time": [0, 3600000], "ep": [10_000_000, 12_000_000]})
        upper = pd.DataFrame({"time": [0, 3600000], "ep": [4_000_000, 6_000_000]})

        interpolated = ds._interpolate_tier_dataframe(lower, upper, 1000, 1500, 2000)

        self.assertIsNotNone(interpolated)
        self.assertEqual(list(interpolated.columns), ["time", "value"])
        self.assertEqual(len(interpolated), 2)
        self.assertGreater(float(interpolated["value"].iloc[0]), 4_000_000)
        self.assertLess(float(interpolated["value"].iloc[0]), 10_000_000)


if __name__ == "__main__":
    unittest.main()
