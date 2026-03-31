import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_models import EventData, EventMeta
from prediction_engine import PredictionEngine


class DummyModeler:
    def shape_function(self, t, Base, A, B, B_end, T_total):
        return Base + A * t + B * (t ** 2)


class TestRefitGuardrails(unittest.TestCase):
    def setUp(self):
        self.engine = PredictionEngine(
            seasonality_handler=None,
            modeler=DummyModeler(),
            config={
                'refit_start_hours': 6.0,
                'refit_recent_hours': 48.0,
                'refit_conf_norm_hours': 72.0,
                'refit_conf_max': 0.35,
                'refit_base_min_ratio': 0.6,
                'refit_base_max_ratio': 1.6,
                'refit_linear_bound_scale': 2.0,
                'refit_linear_zero_ratio': 0.25,
                'refit_quad_min_ratio': 0.1,
                'refit_quad_max_ratio': 2.0,
            }
        )

    def _build_target(self):
        meta = EventMeta(event_id=1, event_type='live_try', start_at=0, end_at=3600 * 1000 * 200, aggregate_at=0)
        df = pd.DataFrame({
            'time': np.arange(0, 101) * 3600 * 1000,
            'value': np.arange(0, 101),
            'hours_elapsed': np.arange(0, 101, dtype=float),
            'norm_speed': np.full(101, 0.02),
            'skeleton_speed': np.full(101, 0.02),
        })
        return EventData(meta=meta, df=df, scale=1000.0)

    def test_refit_bounds_preserve_negative_linear_trend(self):
        bounds = self.engine._get_refit_bounds(np.array([0.03, -0.0003, 0.000002, 0.1, 24.0]))

        self.assertGreaterEqual(bounds[0][0], 0.0)
        self.assertLess(bounds[1][1], 0.0)
        self.assertLess(bounds[1][0], bounds[1][1])
        self.assertGreater(bounds[2][0], 0.0)
        self.assertLess(bounds[2][0], bounds[2][1])

    def test_refit_frame_uses_post_warmup_recent_window(self):
        target = self._build_target()

        refit_df = self.engine._get_refit_frame(target)

        self.assertFalse(refit_df.empty)
        self.assertGreaterEqual(refit_df['hours_elapsed'].min(), 52.0)
        self.assertLessEqual(refit_df['hours_elapsed'].max(), 100.0)

    def test_refit_confidence_is_capped(self):
        target = self._build_target()

        conf = self.engine._get_refit_confidence(target)

        self.assertEqual(conf, 0.35)


if __name__ == '__main__':
    unittest.main()
