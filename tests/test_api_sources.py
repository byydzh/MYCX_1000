import os
import sys
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_source
from config import (
    DEFAULT_API_SOURCE,
    DEFAULT_CONFIG,
    EVENTS_INDEX_CACHE_TTL_SECONDS,
    LIVE_SCALE_CACHE_TTL_SECONDS,
)
from data_source import BandoriDataSource, create_data_source


class TestApiSources(unittest.TestCase):
    def test_default_config_uses_hhwx(self):
        self.assertEqual(DEFAULT_API_SOURCE, "hhwx")
        self.assertEqual(DEFAULT_CONFIG.get("api_source"), "hhwx")

    def test_create_data_source_uses_requested_profile(self):
        ds = create_data_source("hhwx")
        try:
            self.assertIsInstance(ds, BandoriDataSource)
            self.assertEqual(ds.api_source, "hhwx")
            self.assertIn("type=event", ds.api_config["tracker_url"])
            self.assertIn("hhwx.org/api/bandori/tracker/data", ds.api_config["top10_url"])
            self.assertIn("tier=10", ds.api_config["top10_url"])
        finally:
            ds.close()

    def test_fetch_event_meta_extracts_from_hhwx_events_index(self):
        ds = create_data_source("hhwx")
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "data": {
                    "events": [
                        {
                            "eventId": 312,
                            "eventType": "mission_live",
                            "timeline": {
                                "jp": {"startAt": 1000, "endAt": 3000},
                                "cn": {"startAt": 2000, "endAt": 4000},
                            },
                        }
                    ]
                },
            }
            ds.session.get = MagicMock(return_value=mock_response)

            meta = ds.fetch_event_meta(312)

            self.assertEqual(meta["event_id"], 312)
            self.assertEqual(meta["event_type"], "mission_live")
            self.assertEqual(meta["start_at"], 2000)
            self.assertEqual(meta["end_at"], 4000)
            self.assertEqual(meta["aggregate_at"], 4000)
        finally:
            ds.close()

    def test_fetch_tier_1000_data_uses_profile_tracker_url(self):
        ds = create_data_source("hhwx")
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "result": True,
                "cutoffs": [{"time": 1, "ep": 100}, {"time": 2, "ep": 120}],
            }
            ds.session.get = MagicMock(return_value=mock_response)

            df = ds.fetch_tier_1000_data(312)

            called_url = ds.session.get.call_args.args[0]
            self.assertIn("https://hhwx.org/api/bandori/tracker/data", called_url)
            self.assertIn("event=312", called_url)
            self.assertIn("type=event", called_url)
            self.assertIn("tier=1000", called_url)
            self.assertListEqual(df["ep"].tolist(), [100, 120])
        finally:
            ds.close()

    def test_fetch_top10_max_speed_supports_tracker_cutoffs_payload(self):
        ds = create_data_source("hhwx")
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "result": True,
                "cutoffs": [
                    {"time": 0, "ep": 0},
                    {"time": 3600000, "ep": 6000},
                    {"time": 7200000, "ep": 15000},
                    {"time": 10800000, "ep": 21000},
                ],
            }
            ds.session.get = MagicMock(return_value=mock_response)

            scale = ds.fetch_top10_max_speed(304)

            called_url = ds.session.get.call_args.args[0]
            self.assertIn("https://hhwx.org/api/bandori/tracker/data", called_url)
            self.assertIn("tier=10", called_url)
            self.assertGreater(scale, 0)
        finally:
            ds.close()

    def test_fetch_top10_max_speed_uses_bestdori_only_when_explicitly_enabled(self):
        ds = create_data_source("hhwx")
        try:
            responses = [
                {"result": True, "cutoffs": []},
                {
                    "points": [
                        {"time": 0, "uid": 1, "value": 0},
                        {"time": 3600000, "uid": 1, "value": 6000},
                        {"time": 0, "uid": 2, "value": 0},
                        {"time": 3600000, "uid": 2, "value": 6200},
                        {"time": 0, "uid": 3, "value": 0},
                        {"time": 3600000, "uid": 3, "value": 6100},
                    ],
                    "users": [],
                },
            ]

            with patch.object(ds, "_get_json", side_effect=responses) as mock_get_json:
                scale = ds.fetch_top10_max_speed(301, allow_fallback=True)

            self.assertAlmostEqual(scale, ((6000 + 6200 + 6100) / 3) / 60.0)
            self.assertEqual(mock_get_json.call_count, 2)
        finally:
            ds.close()

    def test_live_scale_failure_is_not_cached_and_can_retry_hhwx(self):
        with patch.dict(data_source._GLOBAL_SCALE_CACHE, {}, clear=True):
            ds = create_data_source("hhwx")
            try:
                with patch.object(
                    ds,
                    "_fetch_top10_max_speed_from_config",
                    side_effect=[None, 321.0],
                ) as fetch_scale:
                    failed = ds.fetch_top10_max_speed_observation(
                        987_640, allow_fallback=False
                    )
                    self.assertIsNone(failed.value)
                    self.assertFalse(failed.cache_hit)
                    self.assertFalse(failed.fallback_used)
                    self.assertEqual(fetch_scale.call_count, 1)

                    retried = ds.fetch_top10_max_speed_observation(
                        987_640, allow_fallback=False
                    )

                self.assertEqual(retried.value, 321.0)
                self.assertFalse(retried.cache_hit)
                self.assertEqual(retried.source, "hhwx")
                self.assertEqual(fetch_scale.call_count, 2)
                self.assertTrue(
                    all(
                        call.args[0] is ds.api_config
                        for call in fetch_scale.call_args_list
                    )
                )
            finally:
                ds.close()

    def test_live_scale_cache_expires_at_tracker_refresh_boundary(self):
        with patch.dict(data_source._GLOBAL_SCALE_CACHE, {}, clear=True):
            ds = create_data_source("hhwx")
            try:
                with patch.object(
                    ds,
                    "_fetch_top10_max_speed_from_config",
                    side_effect=[100.0, 200.0],
                ) as fetch_scale:
                    with patch("data_source.time.monotonic", return_value=100.0) as clock:
                        first = ds.fetch_top10_max_speed_observation(987_641)
                        clock.return_value = (
                            100.0 + LIVE_SCALE_CACHE_TTL_SECONDS - 1.0
                        )
                        cached = ds.fetch_top10_max_speed_observation(987_641)
                        clock.return_value = (
                            100.0 + LIVE_SCALE_CACHE_TTL_SECONDS + 1.0
                        )
                        refreshed = ds.fetch_top10_max_speed_observation(987_641)

                self.assertEqual(first.value, 100.0)
                self.assertFalse(first.cache_hit)
                self.assertIsNotNone(first.cache_expires_at)
                self.assertEqual(cached.value, 100.0)
                self.assertTrue(cached.cache_hit)
                self.assertEqual(refreshed.value, 200.0)
                self.assertFalse(refreshed.cache_hit)
                self.assertEqual(fetch_scale.call_count, 2)
            finally:
                ds.close()

    def test_frozen_scale_cache_is_immutable_but_never_caches_failure(self):
        with patch.dict(data_source._GLOBAL_SCALE_CACHE, {}, clear=True):
            ds = create_data_source("hhwx")
            try:
                with patch.object(
                    ds,
                    "_fetch_top10_max_speed_from_config",
                    return_value=123.0,
                ) as fetch_scale:
                    with patch("data_source.time.monotonic", return_value=100.0) as clock:
                        first = ds.fetch_top10_max_speed_observation(
                            987_642, origin_as_of=3_600_000
                        )
                        clock.return_value = 1_000_000_000.0
                        cached = ds.fetch_top10_max_speed_observation(
                            987_642, origin_as_of=3_600_000
                        )

                self.assertEqual(first.value, 123.0)
                self.assertIsNone(first.cache_expires_at)
                self.assertTrue(cached.cache_hit)
                self.assertEqual(fetch_scale.call_count, 1)
            finally:
                ds.close()

    def test_events_index_cache_expires_and_does_not_reuse_stale_on_failure(self):
        first_payload = {
            "success": True,
            "data": {
                "1": {
                    "eventType": "challenge",
                    "startAt": [1, None, None, 1],
                    "endAt": [2, None, None, 2],
                }
            },
        }
        second_payload = {
            "success": True,
            "data": {
                "2": {
                    "eventType": "versus",
                    "startAt": [3, None, None, 3],
                    "endAt": [4, None, None, 4],
                }
            },
        }

        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = create_data_source("hhwx")
            try:
                with patch.object(
                    ds,
                    "_get_json",
                    side_effect=[first_payload, second_payload, None],
                ) as get_json:
                    with patch("data_source.time.monotonic", return_value=100.0) as clock:
                        first = ds.fetch_events_index()
                        clock.return_value = (
                            100.0 + EVENTS_INDEX_CACHE_TTL_SECONDS - 1.0
                        )
                        cached = ds.fetch_events_index()
                        clock.return_value = (
                            100.0 + EVENTS_INDEX_CACHE_TTL_SECONDS + 1.0
                        )
                        refreshed = ds.fetch_events_index()
                        clock.return_value = (
                            100.0 + 2 * EVENTS_INDEX_CACHE_TTL_SECONDS + 2.0
                        )
                        with self.assertRaisesRegex(
                            RuntimeError, "HHWX events request failed"
                        ):
                            ds.fetch_events_index()

                self.assertEqual(set(first), {"1"})
                self.assertIs(cached, first)
                self.assertEqual(set(refreshed), {"2"})
                self.assertEqual(get_json.call_count, 3)
            finally:
                ds.close()

    def test_online_events_index_falls_back_with_explicit_provenance(self):
        bestdori_payload = {
            "320": {
                "eventType": "versus",
                "startAt": [None, None, None, 1_000],
                "endAt": [None, None, None, 2_000],
            }
        }
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(
                    ds,
                    "_get_json",
                    side_effect=[{"success": False}, bestdori_payload],
                ) as get_json:
                    result = ds.fetch_events_index()

                self.assertEqual(result, bestdori_payload)
                self.assertEqual(get_json.call_count, 2)
                provenance = ds.get_provenance("events_index")
                self.assertEqual(provenance["source"], "bestdori")
                self.assertTrue(provenance["fallback_used"])
                self.assertIn("success=true", provenance["primary_error"])
            finally:
                ds.close()

    def test_current_event_falls_back_when_hhwx_cn_timestamps_are_unusable(self):
        hhwx_payload = {
            "success": True,
            "data": {
                "319": {
                    "eventType": "versus",
                    "startAt": [1_000, None, None, None],
                    "endAt": [2_000, None, None, None],
                }
            },
        }
        bestdori_payload = {
            "320": {
                "eventType": "versus",
                "startAt": [None, None, None, 1_000],
                "endAt": [None, None, None, 2_000],
            }
        }
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(
                    ds,
                    "_get_json",
                    side_effect=[hhwx_payload, bestdori_payload],
                ) as get_json, patch("data_source.time.time", return_value=1.5):
                    event_id = ds.get_current_event_id()

                self.assertEqual(event_id, 320)
                self.assertEqual(get_json.call_count, 2)
                provenance = ds.get_provenance("current_event")
                self.assertEqual(provenance["source"], "bestdori")
                self.assertTrue(provenance["fallback_used"])
                self.assertIn("no timestamps for server 3", provenance["primary_error"])
            finally:
                ds.close()

    def test_current_event_hhwx_success_never_requests_bestdori(self):
        hhwx_payload = {
            "success": True,
            "data": {
                "320": {
                    "eventType": "versus",
                    "startAt": [None, None, None, 1_000],
                    "endAt": [None, None, None, 2_000],
                }
            },
        }
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(ds, "_get_json", return_value=hhwx_payload) as get_json, \
                        patch("data_source.time.time", return_value=1.5):
                    event_id = ds.get_current_event_id()

                self.assertEqual(event_id, 320)
                get_json.assert_called_once()
                provenance = ds.get_provenance("current_event")
                self.assertEqual(provenance["source"], "hhwx")
                self.assertFalse(provenance["fallback_used"])
            finally:
                ds.close()

    def test_current_event_double_semantic_failure_names_both_sources(self):
        unusable_hhwx = {
            "success": True,
            "data": {
                "319": {
                    "eventType": "versus",
                    "startAt": [1_000, None, None, None],
                    "endAt": [2_000, None, None, None],
                }
            },
        }
        unusable_bestdori = {
            "320": {
                "eventType": "versus",
                "startAt": [1_000, None, None, None],
                "endAt": [2_000, None, None, None],
            }
        }
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(
                    ds,
                    "_get_json",
                    side_effect=[unusable_hhwx, unusable_bestdori],
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        ds.get_current_event_id()

                message = str(caught.exception)
                self.assertIn("HHWX", message)
                self.assertIn("BESTDORI", message)
                self.assertEqual(message.count("no timestamps for server 3"), 2)
            finally:
                ds.close()

    def test_online_tracker_falls_back_only_after_hhwx_empty(self):
        ds = create_data_source("hhwx", allow_fallback=True)
        try:
            with patch.object(
                ds,
                "_get_json",
                side_effect=[
                    {"result": True, "cutoffs": []},
                    {
                        "result": True,
                        "cutoffs": [
                            {"time": 1, "ep": 100},
                            {"time": 2, "ep": 130},
                        ],
                    },
                ],
            ) as get_json:
                frame = ds.fetch_tier_data(320, 500)

            self.assertEqual(get_json.call_count, 2)
            self.assertEqual(frame.attrs["source"], "bestdori")
            self.assertTrue(frame.attrs["fallback_used"])
            self.assertIn("cutoffs are empty", frame.attrs["primary_error"])
        finally:
            ds.close()

    def test_online_meta_falls_back_after_hhwx_schema_failure(self):
        bestdori_meta = {
            "eventType": "versus",
            "startAt": [None, None, None, 1_000],
            "endAt": [None, None, None, 2_000],
            "aggregateAt": [None, None, None, 2_100],
        }
        with patch.dict(data_source._GLOBAL_EVENTS_INDEX_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(
                    ds,
                    "_get_json",
                    side_effect=[{"success": True, "data": {}}, bestdori_meta],
                ):
                    meta = ds.fetch_event_meta(320)

                self.assertEqual(meta["source"], "bestdori")
                self.assertTrue(meta["fallback_used"])
                self.assertEqual(meta["aggregate_at"], 2_100)
                provenance = ds.get_provenance("event_meta")
                self.assertIn("HHWX", provenance["primary_error"])
            finally:
                ds.close()

    def test_online_rewards_fall_back_after_hhwx_empty(self):
        bestdori_rewards = {
            "rankingRewards": [
                [],
                [],
                [],
                [
                    {
                        "toRank": 500,
                        "rewardType": "voice_stamp",
                        "rewardId": 10118,
                    }
                ],
            ]
        }
        ds = create_data_source("hhwx", allow_fallback=True)
        try:
            with patch.object(
                ds,
                "_get_json",
                side_effect=[
                    {"success": True, "data": {"rankingRewards": [[], [], [], []]}},
                    bestdori_rewards,
                ],
            ):
                rewards = ds.fetch_event_rewards(320)

            self.assertEqual(rewards["target_tiers"], [500])
            self.assertEqual(rewards["source"], "bestdori")
            self.assertTrue(rewards["fallback_used"])
        finally:
            ds.close()

    def test_online_tracker_hhwx_success_never_calls_fallback(self):
        ds = create_data_source("hhwx", allow_fallback=True)
        try:
            payload = {
                "result": True,
                "cutoffs": [
                    {"time": 1, "ep": 100},
                    {"time": 2, "ep": 120},
                ],
            }
            with patch.object(ds, "_get_json", return_value=payload) as get_json:
                frame = ds.fetch_tier_data(320, 500)

            get_json.assert_called_once()
            self.assertEqual(frame.attrs["source"], "hhwx")
            self.assertFalse(frame.attrs["fallback_used"])
        finally:
            ds.close()

    def test_online_tracker_double_failure_names_both_sources(self):
        ds = create_data_source("hhwx", allow_fallback=True)
        try:
            with patch.object(ds, "_get_json", side_effect=[None, None]):
                with self.assertRaises(RuntimeError) as caught:
                    ds.fetch_tier_data(320, 500)

            message = str(caught.exception)
            self.assertIn("HHWX", message)
            self.assertIn("BESTDORI", message)
            self.assertIn("tracker unavailable", message)
        finally:
            ds.close()

    def test_scale_fallback_cache_never_leaks_into_strict_primary_route(self):
        with patch.dict(data_source._GLOBAL_SCALE_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(
                    ds,
                    "_fetch_top10_max_speed_from_config",
                    side_effect=[None, 123.0, 456.0],
                ) as fetch_scale:
                    fallback = ds.fetch_top10_max_speed_observation(
                        987_643,
                        allow_fallback=True,
                    )
                    strict = ds.fetch_top10_max_speed_observation(
                        987_643,
                        allow_fallback=False,
                    )

                self.assertEqual(fallback.source, "bestdori")
                self.assertTrue(fallback.fallback_used)
                self.assertEqual(strict.source, "hhwx")
                self.assertFalse(strict.fallback_used)
                self.assertFalse(strict.cache_hit)
                self.assertEqual(fetch_scale.call_count, 3)
            finally:
                ds.close()

    def test_online_scale_schema_failure_falls_back_to_bestdori(self):
        with patch.dict(data_source._GLOBAL_SCALE_CACHE, {}, clear=True):
            ds = create_data_source("hhwx", allow_fallback=True)
            try:
                with patch.object(
                    ds,
                    "_get_json",
                    side_effect=[
                        {"cutoffs": [{"ep": 10}]},
                        {
                            "points": [
                                {"time": 0, "uid": 1, "value": 0},
                                {"time": 3_600_000, "uid": 1, "value": 6_000},
                            ]
                        },
                    ],
                ):
                    observation = ds.fetch_top10_max_speed_observation(987_644)

                self.assertEqual(observation.source, "bestdori")
                self.assertTrue(observation.fallback_used)
                self.assertGreater(observation.value, 0)
                self.assertIn("HHWX", observation.primary_error)
            finally:
                ds.close()


if __name__ == "__main__":
    unittest.main()
