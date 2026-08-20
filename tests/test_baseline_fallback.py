import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_source
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

    def test_special_reward_targets_keep_distinct_reward_ids(self):
        entries = [
            {"toRank": 500, "rewardType": "voice_stamp", "rewardId": 10118},
            {"toRank": 1500, "rewardType": "voice_stamp", "rewardId": 519},
            {"toRank": 2000, "rewardType": "deco_pins", "rewardId": 103210},
            {"toRank": 1000, "rewardType": "degree", "rewardId": 8633},
            {
                "toRank": 1000,
                "rewardType": "bili_degree_effect",
                "rewardId": 8633,
            },
            {"toRank": 3000, "rewardType": "star", "rewardId": None},
        ]

        target_tiers, last_appearance = BandoriDataSource._extract_special_reward_targets(entries)

        self.assertEqual(target_tiers, [500, 1500, 2000])
        self.assertEqual(last_appearance["voice_stamp:10118"], 500)
        self.assertEqual(last_appearance["voice_stamp:519"], 1500)
        self.assertEqual(last_appearance["deco_pins:103210"], 2000)

    def test_hhwx_master_events_contract_drives_current_id_and_meta(self):
        payload = {
            "success": True,
            "data": {
                "318": {
                    "eventType": "mission_live",
                    "startAt": [1000, None, None, "2000"],
                    "endAt": [2000, None, None, "3000"],
                    "aggregateEndAt": [2100, None, None, "3100"],
                },
                "319": {
                    "eventType": "live_try",
                    "startAt": [3000, None, None, "4000"],
                    "endAt": [5000, None, None, "6000"],
                    "aggregateEndAt": [5100, None, None, "6100"],
                },
            },
        }
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = BandoriDataSource(api_source="hhwx", server_index=3)
            try:
                with patch.object(ds, "_get_json", return_value=payload) as get_json:
                    with patch("data_source.time.time", return_value=5.0):
                        self.assertEqual(ds.get_current_event_id(), 319)
                    meta = ds.fetch_event_meta(319)

                get_json.assert_called_once_with(
                    "https://hhwx.org/api/bandori/master/events", timeout=8
                )
                self.assertEqual(meta["event_id"], 319)
                self.assertEqual(meta["event_type"], "live_try")
                self.assertEqual(meta["start_at"], 4000)
                self.assertEqual(meta["end_at"], 6000)
                self.assertEqual(meta["aggregate_at"], 6100)
            finally:
                ds.close()

    def test_hhwx_detail_wrapper_drives_special_reward_targets(self):
        entries = [
            {"toRank": 500, "rewardType": "voice_stamp", "rewardId": 10118},
            {"toRank": 1500, "rewardType": "voice_stamp", "rewardId": 519},
            {"toRank": 2000, "rewardType": "deco_pins", "rewardId": 103210},
            {"toRank": 1000, "rewardType": "degree", "rewardId": 8633},
            {
                "toRank": 1000,
                "rewardType": "bili_degree_effect",
                "rewardId": 8633,
            },
        ]
        payload = {
            "success": True,
            "data": {"rankingRewards": [[], None, [], entries]},
        }
        ds = BandoriDataSource(api_source="hhwx", server_index=3)
        try:
            with patch.object(ds, "_get_json", return_value=payload) as get_json:
                rewards = ds.fetch_event_rewards(318)

            get_json.assert_called_once_with(
                "https://hhwx.org/api/bandori/master/events/318", timeout=8
            )
            self.assertEqual(rewards["target_tiers"], [500, 1500, 2000])
            self.assertNotIn("bili_degree_effect:8633", rewards["last_appearance"])
        finally:
            ds.close()

    def test_hhwx_primary_request_failures_are_explicit(self):
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = BandoriDataSource(api_source="hhwx", server_index=3)
            try:
                with patch.object(ds, "_get_json", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "HHWX events request failed"):
                        ds.fetch_events_index()
                    with self.assertRaisesRegex(RuntimeError, "detail request failed"):
                        ds.fetch_event_rewards(318)
            finally:
                ds.close()


if __name__ == "__main__":
    unittest.main()
