import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG, load_preset
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
        return np.array([0.02, 0.0, 0.0, 0.1, 24.0])

    def shape_function(self, t, Base, A, B, B_end, T_panic, T_total):
        t = np.asarray(t, dtype=float)
        return Base + A * t + B * (t ** 2)


def _build_event(event_id: int, offset: float = 0.0) -> EventData:
    hours = np.arange(0.0, 102.0, 6.0)
    times = (hours * 3600 * 1000).astype(np.int64)

    speed_pt_min = np.full(hours.shape, 20.0 + offset)
    values = 100000.0 + np.cumsum(speed_pt_min * 60.0 * 6.0)
    norm_speed = speed_pt_min / 1000.0

    df = pd.DataFrame(
        {
            "time": times,
            "value": values,
            "hours_elapsed": hours,
            "speed": speed_pt_min,
            "norm_speed": norm_speed,
            "skeleton_speed": norm_speed,
        }
    )
    meta = EventMeta(
        event_id=event_id,
        event_type="medley",
        start_at=0,
        end_at=int(192 * 3600 * 1000),
        aggregate_at=int(192 * 3600 * 1000),
    )
    return EventData(meta=meta, df=df, scale=1000.0)


class TestBackwardCompatibility(unittest.TestCase):
    def test_default_preset_matches_default_config_dict(self):
        self.assertEqual(load_preset("skeleton_kf", "default"), DEFAULT_CONFIG)

    def test_default_preset_prediction_matches_legacy_default_prediction(self):
        seasonality = DummySeasonality()
        modeler = DummyModeler()

        legacy_engine = PredictionEngine(seasonality, modeler, config=DEFAULT_CONFIG)
        preset_engine = PredictionEngine(
            seasonality,
            modeler,
            config=load_preset("skeleton_kf", "default"),
        )

        legacy_result = legacy_engine.predict(
            _build_event(400),
            [_build_event(399, offset=-1.0), _build_event(398, offset=1.0)],
        )
        preset_result = preset_engine.predict(
            _build_event(400),
            [_build_event(399, offset=-1.0), _build_event(398, offset=1.0)],
        )

        self.assertAlmostEqual(legacy_result.final_score, preset_result.final_score, places=6)
        self.assertAlmostEqual(legacy_result.ratio, preset_result.ratio, places=9)
        self.assertTrue(np.allclose(legacy_result.used_params, preset_result.used_params))
        self.assertTrue(np.allclose(legacy_result.future_speed, preset_result.future_speed))


if __name__ == "__main__":
    unittest.main()
