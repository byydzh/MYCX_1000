import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "collect_tier_surface_cache", PROJECT_ROOT / "scripts" / "collect_tier_surface_cache.py"
)
collector = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(collector)


_DEFAULT_REWARD_RESULT = object()


class FakeDataSource:
    def __init__(self, records_by_tier, reward_result=_DEFAULT_REWARD_RESULT):
        self.records_by_tier = records_by_tier
        self.reward_result = (
            {
                "target_tiers": [1000],
                "last_appearance": {"voice_stamp:test": 1000},
            }
            if reward_result is _DEFAULT_REWARD_RESULT
            else reward_result
        )
        self.calls = []
        self.reward_calls = []
        self.closed = False

    def fetch_event_meta(self, event_id):
        return {"event_id": int(event_id), "start_at": 0, "end_at": 10, "aggregate_at": 10}

    def fetch_tier_data(self, event_id, tier):
        self.calls.append((int(event_id), int(tier)))
        records = self.records_by_tier.get(int(tier))
        return None if records is None else pd.DataFrame(records)

    def fetch_event_rewards(self, event_id):
        self.reward_calls.append(int(event_id))
        return self.reward_result

    def close(self):
        self.closed = True


def _args(cache_dir, **overrides):
    values = {
        "api_source": "bestdori",
        "server": 3,
        "event_ids": [312],
        "min_event_id": None,
        "max_event_id": None,
        "tiers": [10, 1000],
        "wait_ms": 0,
        "workers": 1,
        "cache_dir": cache_dir,
        "dry_run": False,
        "refresh_existing": False,
        "replace_legacy_reward_metadata": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestCollectTierSurfaceCache(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path("mock-tier-surface-cache")

    def test_incrementally_adds_missing_tiers_and_keeps_raw_rows(self):
        old_t10 = [{"time": 1, "ep": 100}]
        source = FakeDataSource({1000: [{"time": 2, "ep": 50}]})

        with patch.object(collector, "read_cache", return_value={"event_id": 312, "tier_records": {"10": old_t10}}), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(_args(self.cache_dir), lambda *args, **kwargs: source)

        self.assertEqual(result[0]["status"], "updated")
        self.assertEqual(source.calls, [(312, 1000)])
        payload = write_cache.call_args.args[1]
        self.assertEqual(payload["tier_records"]["10"], old_t10)
        self.assertEqual(payload["tier_records"]["1000"], [{"time": 2, "ep": 50}])
        metadata = payload["collection_metadata"]
        self.assertEqual(metadata["source"], "bestdori")
        self.assertIsNone(metadata["available_at"])
        self.assertEqual(
            metadata["availability_status"],
            collector.VERSION_AVAILABILITY_STATUS,
        )
        self.assertIn("fetched_at", metadata)
        self.assertNotIn("available_at", payload["tier_records"]["1000"][0])
        self.assertEqual(payload["reward_tiers"], [1000])
        reward_provenance = payload["reward_tier_provenance"]
        self.assertEqual(reward_provenance["source"], "bestdori")
        self.assertEqual(reward_provenance["server"], 3)
        self.assertEqual(
            reward_provenance["last_appearance"],
            {"voice_stamp:test": 1000},
        )
        self.assertIsInstance(reward_provenance["observed_at"], int)
        self.assertEqual(source.reward_calls, [312])
        version = payload["tier_record_versions"]["1000"][0]
        self.assertEqual(version["time"], 2)
        self.assertEqual(version["ep"], 50)
        self.assertIsInstance(version["available_at"], int)
        self.assertEqual(version["source"], "bestdori")
        self.assertTrue(source.closed)

    def test_terminal_refresh_window_extends_past_declared_end(self):
        meta = {"end_at": 1_000, "aggregate_at": 1_000}
        self.assertTrue(
            collector._event_is_live(
                meta,
                now_ms=1_000 + collector.FINAL_REFRESH_GRACE_MS,
            )
        )
        self.assertFalse(
            collector._event_is_live(
                meta,
                now_ms=1_001 + collector.FINAL_REFRESH_GRACE_MS,
            )
        )

    def test_failed_fetch_does_not_rewrite_existing_cache(self):
        source = FakeDataSource({1000: None})

        with patch.object(collector, "read_cache", return_value={"event_id": 312, "tier_records": {"10": [{"time": 1, "ep": 100}]}}), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(_args(self.cache_dir), lambda *args, **kwargs: source)

        self.assertEqual(result[0]["status"], "failed_tiers")
        write_cache.assert_not_called()

    def test_misfiled_cache_identity_fails_closed(self):
        source = FakeDataSource({})
        with patch.object(
            collector,
            "read_cache",
            return_value={"event_id": 999, "tier_records": {}},
        ), patch.object(collector, "atomic_write_json") as write_cache:
            with self.assertRaisesRegex(ValueError, "999 != requested 312"):
                collector.collect_events(
                    _args(self.cache_dir),
                    lambda *args, **kwargs: source,
                )
        write_cache.assert_not_called()
        self.assertTrue(source.closed)

    def test_live_event_refreshes_existing_tier_and_appends_new_rows(self):
        source = FakeDataSource({
            10: [{"time": 2, "ep": 125}, {"time": 3, "ep": 150}],
            1000: [{"time": 2, "ep": 55}],
        })
        existing = {
            "event_id": 312,
            "meta": {
                "start_at": 0,
                "end_at": 9_999_999_999_998,
                "aggregate_at": 9_999_999_999_999,
            },
            "tier_records": {
                "10": [{"time": 1, "ep": 100}, {"time": 2, "ep": 120}],
                "1000": [{"time": 1, "ep": 50}],
            },
        }

        with patch.object(collector, "read_cache", return_value=existing), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(
                _args(self.cache_dir), lambda *args, **kwargs: source
            )

        self.assertEqual(source.calls, [(312, 10), (312, 1000)])
        self.assertEqual(result[0]["status"], "updated")
        self.assertEqual(result[0]["refreshed_tiers"], [10, 1000])
        payload = write_cache.call_args.args[1]
        self.assertEqual(
            payload["tier_records"]["10"],
            [
                {"time": 1, "ep": 100},
                {"time": 2, "ep": 125},
                {"time": 3, "ep": 150},
            ],
        )
        self.assertEqual(
            payload["tier_records"]["1000"],
            [{"time": 1, "ep": 50}, {"time": 2, "ep": 55}],
        )
        self.assertEqual(
            payload["collection_metadata"]["changed_tiers"], [10, 1000]
        )
        t10_versions = payload["tier_record_versions"]["10"]
        self.assertEqual(
            [
                row["ep"]
                for row in t10_versions
                if row["time"] == 2
            ],
            [120, 125],
        )
        self.assertTrue(
            all("available_at" in row for row in t10_versions)
        )

    def test_completed_cache_missing_rewards_gets_reward_only_update(self):
        rewards = {
            "target_tiers": [2000, 500, 1500],
            "last_appearance": {
                "voice_stamp:519": 1500,
                "deco_pins:103210": 2000,
                "voice_stamp:10118": 500,
            },
        }
        source = FakeDataSource({}, reward_result=rewards)
        existing = {
            "event_id": 312,
            "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
            "tier_records": {
                "500": [{"time": 1, "ep": 100}],
                "1500": [{"time": 1, "ep": 50}],
                "2000": [{"time": 1, "ep": 25}],
            },
        }

        with patch.object(collector, "read_cache", return_value=existing), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(
                _args(
                    self.cache_dir,
                    api_source="hhwx",
                    tiers=[500, 1500, 2000],
                ),
                lambda *args, **kwargs: source,
            )

        self.assertEqual(result[0]["status"], "updated")
        self.assertTrue(result[0]["reward_metadata_changed"])
        self.assertEqual(source.calls, [])
        self.assertEqual(source.reward_calls, [312])
        payload = write_cache.call_args.args[1]
        self.assertEqual(payload["reward_tiers"], [500, 1500, 2000])
        self.assertEqual(
            payload["reward_tier_provenance"]["last_appearance"],
            {
                "deco_pins:103210": 2000,
                "voice_stamp:10118": 500,
                "voice_stamp:519": 1500,
            },
        )
        self.assertEqual(payload["reward_tier_provenance"]["source"], "hhwx")
        self.assertEqual(payload["reward_tier_provenance"]["server"], 3)
        self.assertIsInstance(
            payload["reward_tier_provenance"]["observed_at"], int
        )

    def test_completed_cache_still_refetches_and_validates_rewards(self):
        rewards = {
            "target_tiers": [1500],
            "last_appearance": {"voice_stamp:519": 1500},
        }
        source = FakeDataSource({}, reward_result=rewards)
        existing = {
            "event_id": 312,
            "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
            "tier_records": {"1500": [{"time": 1, "ep": 50}]},
            "reward_tiers": [1500],
            "reward_tier_provenance": {
                "source": "hhwx",
                "server": 3,
                "last_appearance": {"voice_stamp:519": 1500},
                "observed_at": 123,
            },
        }

        with patch.object(collector, "read_cache", return_value=existing), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(
                _args(
                    self.cache_dir,
                    api_source="hhwx",
                    tiers=[1500],
                ),
                lambda *args, **kwargs: source,
            )

        self.assertEqual(result[0]["status"], "already_complete")
        self.assertFalse(result[0]["reward_metadata_changed"])
        self.assertEqual(source.reward_calls, [312])
        self.assertEqual(source.calls, [])
        write_cache.assert_not_called()

    def test_explicit_switch_replaces_legacy_reward_metadata_only(self):
        rewards = {
            "target_tiers": [500, 1500, 2000],
            "last_appearance": {
                "voice_stamp:1": 500,
                "voice_stamp:2": 1500,
                "deco_pins:3": 2000,
            },
        }
        source = FakeDataSource(
            {2000: [{"time": 2, "ep": 30}]},
            reward_result=rewards,
        )
        existing = {
            "event_id": 312,
            "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
            "tier_records": {
                "500": [{"time": 1, "ep": 100}],
                "1500": [{"time": 1, "ep": 50}],
            },
            "reward_tiers": [1000],
        }

        with patch.object(collector, "read_cache", return_value=existing), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(
                _args(
                    self.cache_dir,
                    api_source="hhwx",
                    tiers=[500, 1500, 2000],
                    replace_legacy_reward_metadata=True,
                ),
                lambda *args, **kwargs: source,
            )

        self.assertEqual(result[0]["status"], "updated")
        self.assertTrue(result[0]["reward_metadata_changed"])
        self.assertTrue(result[0]["reward_metadata_replaced_legacy"])
        self.assertEqual(source.calls, [(312, 2000)])
        payload = write_cache.call_args.args[1]
        self.assertEqual(payload["reward_tiers"], [500, 1500, 2000])
        self.assertEqual(
            payload["tier_records"]["2000"],
            [{"time": 2, "ep": 30}],
        )
        self.assertEqual(
            payload["reward_tier_provenance"]["last_appearance"],
            {
                "deco_pins:3": 2000,
                "voice_stamp:1": 500,
                "voice_stamp:2": 1500,
            },
        )

    def test_invalid_or_unrequested_reward_metadata_fails_explicitly(self):
        cases = [
            (None, [1000], "response is missing"),
            ({}, [1000], "needs target_tiers and last_appearance"),
            (
                {
                    "target_tiers": [1500],
                    "last_appearance": {"voice_stamp:1": 1000},
                },
                [1000, 1500],
                "disagree",
            ),
            (
                {
                    "target_tiers": [1500],
                    "last_appearance": {"voice_stamp:1": 1500},
                },
                [1000],
                "were not requested",
            ),
        ]
        for reward_result, tiers, message in cases:
            with self.subTest(message=message):
                source = FakeDataSource({}, reward_result=reward_result)
                existing = {
                    "event_id": 312,
                    "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
                    "tier_records": {
                        str(tier): [{"time": 1, "ep": 1}] for tier in tiers
                    },
                }
                with patch.object(collector, "read_cache", return_value=existing), \
                        patch.object(collector, "atomic_write_json") as write_cache:
                    with self.assertRaisesRegex(ValueError, message):
                        collector.collect_events(
                            _args(
                                self.cache_dir,
                                api_source="hhwx",
                                tiers=tiers,
                            ),
                            lambda *args, **kwargs: source,
                        )
                write_cache.assert_not_called()
                self.assertEqual(source.reward_calls, [312])
                self.assertTrue(source.closed)

    def test_explicit_empty_reward_set_is_valid_metadata(self):
        source = FakeDataSource(
            {},
            reward_result={"target_tiers": [], "last_appearance": {}},
        )
        existing = {
            "event_id": 312,
            "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
            "tier_records": {"1000": [{"time": 1, "ep": 1}]},
        }
        with patch.object(collector, "read_cache", return_value=existing), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(
                _args(self.cache_dir, api_source="hhwx", tiers=[1000]),
                lambda *args, **kwargs: source,
            )

        self.assertEqual(result[0]["status"], "updated")
        payload = write_cache.call_args.args[1]
        self.assertEqual(payload["reward_tiers"], [])
        self.assertEqual(
            payload["reward_tier_provenance"]["last_appearance"], {}
        )

    def test_explicit_legacy_replacement_repairs_obsolete_nontracker_tiers(self):
        source = FakeDataSource(
            {},
            reward_result={
                "target_tiers": [500, 1500, 2000],
                "last_appearance": {
                    "voice_stamp:540": 1500,
                    "deco_pins:103300": 2000,
                    "voice_stamp:10134": 500,
                },
            },
        )
        existing = {
            "event_id": 316,
            "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
            # Historical Graph extraction incorrectly treated these ranks as
            # reward targets.  There is deliberately no provenance object.
            "reward_tiers": [1, 2, 3, 10, 20, 30, 40, 50, 100, 200, 300,
                             400, 500, 1000, 1500, 2000],
            "tier_records": {
                "500": [{"time": 1, "ep": 3}],
                "1500": [{"time": 1, "ep": 2}],
                "2000": [{"time": 1, "ep": 1}],
            },
        }
        with patch.object(collector, "read_cache", return_value=existing), \
                patch.object(collector, "atomic_write_json") as write_cache:
            result = collector.collect_events(
                _args(
                    self.cache_dir,
                    api_source="hhwx",
                    event_ids=[316],
                    tiers=[500, 1500, 2000],
                    replace_legacy_reward_metadata=True,
                ),
                lambda *args, **kwargs: source,
            )

        self.assertEqual(result[0]["status"], "updated")
        payload = write_cache.call_args.args[1]
        self.assertEqual(payload["reward_tiers"], [500, 1500, 2000])
        self.assertEqual(payload["reward_tier_provenance"]["source"], "hhwx")

    def test_partial_or_conflicting_cached_rewards_fail_explicitly(self):
        fetched_rewards = {
            "target_tiers": [1500],
            "last_appearance": {"voice_stamp:2": 1500},
        }
        cases = [
            (
                {"reward_tiers": [1500]},
                "cached reward metadata is partial",
                False,
            ),
            (
                {
                    "reward_tiers": [500],
                    "reward_tier_provenance": {
                        "source": "hhwx",
                        "server": 3,
                        "last_appearance": {"voice_stamp:1": 500},
                        "observed_at": 1,
                    },
                },
                "cached reward metadata conflicts",
                True,
            ),
        ]
        for cached_rewards, message, replace_legacy in cases:
            with self.subTest(message=message):
                source = FakeDataSource({}, reward_result=fetched_rewards)
                existing = {
                    "event_id": 312,
                    "meta": {"start_at": 0, "end_at": 10, "aggregate_at": 10},
                    "tier_records": {
                        "500": [{"time": 1, "ep": 2}],
                        "1500": [{"time": 1, "ep": 1}],
                    },
                    **cached_rewards,
                }
                with patch.object(collector, "read_cache", return_value=existing), \
                        patch.object(collector, "atomic_write_json") as write_cache:
                    with self.assertRaisesRegex(ValueError, message):
                        collector.collect_events(
                            _args(
                                self.cache_dir,
                                api_source="hhwx",
                                tiers=[500, 1500],
                                replace_legacy_reward_metadata=replace_legacy,
                            ),
                            lambda *args, **kwargs: source,
                        )
                write_cache.assert_not_called()
                self.assertEqual(source.reward_calls, [312])
                self.assertTrue(source.closed)

    def test_dry_run_has_no_network_or_filesystem_mutation(self):
        source_factory = lambda *args, **kwargs: self.fail("dry-run must not create a data source")

        with patch.object(collector, "read_cache", return_value={}):
            result = collector.collect_events(_args(self.cache_dir, dry_run=True), source_factory)

        self.assertEqual(result[0]["status"], "dry_run")
        self.assertEqual(result[0]["would_fetch_tiers"], [10, 1000])


if __name__ == "__main__":
    unittest.main()
