import os
import sys
import unittest
from unittest.mock import MagicMock

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_TRACKER_TIERS, canonicalize_tracker_tiers
from data_source import BandoriDataSource
from tier_surface import build_origin_as_of_tier_snapshot


class TestTierSurfaceAsOfContract(unittest.TestCase):
    def test_missing_start_and_negative_scores_fail_closed(self):
        frame = pd.DataFrame({"time": [0, 1], "ep": [10.0, -5.0]})
        with self.assertRaisesRegex(ValueError, "start_at is required"):
            build_origin_as_of_tier_snapshot({10: frame}, {}, 1, tiers=[10])

        snapshot = build_origin_as_of_tier_snapshot(
            {10: frame},
            {"start_at": 0},
            1,
            tiers=[10],
        )
        self.assertEqual(
            snapshot.quality_report["quality_summary"]["invalid_value_rows"],
            1,
        )
        self.assertEqual(snapshot.surface[10].dropna().tolist(), [10.0])

    def test_public_tier_contract_is_ordered_and_rejects_arbitrary_rank(self):
        self.assertEqual(ALL_TRACKER_TIERS[0], 1)
        self.assertEqual(ALL_TRACKER_TIERS[-1], 100000)
        self.assertEqual(canonicalize_tracker_tiers([1000, 10, 5000]), [10, 1000, 5000])
        with self.assertRaisesRegex(ValueError, "Unsupported tracker tier T999"):
            canonicalize_tracker_tiers([999])
        for invalid in (True, 10.9, "010", "10.0"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    canonicalize_tracker_tiers([invalid])

    def test_future_mutation_never_enters_as_of_surface(self):
        base = pd.DataFrame({
            "time": [0, 3_600_000, 7_200_000],
            "ep": [0, 100, 250],
        })
        mutated = base.copy()
        mutated.loc[2, "ep"] = 99_999_999
        kwargs = {
            "meta": {"start_at": 0},
            "origin_as_of": 3_600_000,
            "tiers": [1000],
        }
        original = build_origin_as_of_tier_snapshot({1000: base}, **kwargs)
        after_mutation = build_origin_as_of_tier_snapshot({1000: mutated}, **kwargs)

        pd.testing.assert_frame_equal(original.surface, after_mutation.surface)
        self.assertEqual(after_mutation.surface.loc[1.0, 1000], 100)
        report = after_mutation.quality_report["tiers"]["1000"]
        self.assertEqual(report["post_origin_rows_excluded"], 1)
        self.assertEqual(after_mutation.quality_report["availability"]["status"], "degraded_timestamp_only")

    def test_surface_has_canonical_tier_order_and_missing_tier_report(self):
        t10 = pd.DataFrame({"time": [0], "ep": [100]})
        t1000 = pd.DataFrame({"time": [0], "ep": [10]})
        snapshot = build_origin_as_of_tier_snapshot(
            {1000: t1000, 10: t10},
            {"start_at": 0},
            0,
            tiers=[1000, 20, 10],
        )

        self.assertEqual(list(snapshot.surface.columns), [10, 1000])
        self.assertEqual(snapshot.quality_report["missing_tiers"], [20])
        self.assertAlmostEqual(snapshot.quality_report["completeness"]["fraction"], 2 / 3)

    def test_quality_report_counts_duplicates_monotonicity_and_available_at(self):
        df = pd.DataFrame({
            "time": [0, 3_600_000, 3_600_000, 5_400_000, 7_200_000],
            "ep": [100, 130, 120, 110, 140],
            "available_at": [0, 3_600_000, 3_600_000, 5_400_000, 10_800_000],
        })
        snapshot = build_origin_as_of_tier_snapshot(
            {10: df}, {"start_at": 0}, 7_200_000, tiers=[10]
        )
        quality = snapshot.quality_report["tiers"]["10"]
        self.assertEqual(quality["duplicate_timestamps"], 2)
        self.assertEqual(quality["non_monotonic_steps"], 1)
        self.assertEqual(quality["future_available_rows_excluded"], 1)
        self.assertEqual(snapshot.provenance[10]["availability_status"], "explicit_row_level")

    def test_partial_row_level_available_at_is_excluded_and_degraded(self):
        df = pd.DataFrame({
            "time": [0, 3_600_000, 7_200_000],
            "ep": [100, 130, 160],
            "available_at": [0, None, 7_200_000],
        })
        snapshot = build_origin_as_of_tier_snapshot(
            {10: df}, {"start_at": 0}, 7_200_000, tiers=[10]
        )

        self.assertNotIn(1.0, snapshot.surface.index)
        quality = snapshot.quality_report["tiers"]["10"]
        self.assertEqual(quality["unknown_available_rows_excluded"], 1)
        self.assertEqual(
            snapshot.provenance[10]["availability_status"],
            "degraded_partial_row_level_available_at",
        )
        self.assertEqual(
            snapshot.quality_report["availability"]["status"],
            "degraded_timestamp_only",
        )

    def test_latest_visible_revision_wins_independent_of_input_order(self):
        revisions = pd.DataFrame({
            "time": [0, 50, 50],
            "ep": [0, 200, 100],
            "available_at": [0, 90, 80],
        })
        snapshot = build_origin_as_of_tier_snapshot(
            {10: revisions}, {"start_at": 0}, 100, tiers=[10]
        )

        self.assertEqual(float(snapshot.surface.iloc[-1, 0]), 200.0)

    def test_missing_requested_tier_never_claims_explicit_availability(self):
        t10 = pd.DataFrame({
            "time": [0],
            "ep": [100],
            "available_at": [0],
        })
        snapshot = build_origin_as_of_tier_snapshot(
            {10: t10}, {"start_at": 0}, 0, tiers=[10, 1000]
        )

        self.assertEqual(
            snapshot.quality_report["availability"]["status"],
            "degraded_timestamp_only",
        )
        self.assertEqual(
            snapshot.provenance[1000]["availability_status"],
            "unknown_degraded_missing_tier",
        )

    def test_unsupported_tier_fails_before_network_request(self):
        ds = BandoriDataSource(api_source="hhwx", server_index=3)
        try:
            ds.session.get = MagicMock()
            with self.assertRaisesRegex(ValueError, "Unsupported tracker tier T999"):
                ds.fetch_tier_data(312, 999)
            ds.session.get.assert_not_called()
        finally:
            ds.close()

    def test_scale_origin_as_of_excludes_future_top10_change(self):
        ds = BandoriDataSource(api_source="bestdori", server_index=3)
        try:
            ds._get_json = MagicMock(return_value={
                "points": [
                    {"uid": 1, "time": 0, "value": 0},
                    {"uid": 1, "time": 3_600_000, "value": 6_000},
                    {"uid": 1, "time": 7_200_000, "value": 606_000},
                ]
            })
            scale = ds.fetch_top10_max_speed(987654, origin_as_of=3_600_000, allow_fallback=False)
            self.assertAlmostEqual(scale, 100.0)
        finally:
            ds.close()

    def test_scale_observation_preserves_fallback_and_cache_provenance(self):
        ds = BandoriDataSource(api_source="hhwx", server_index=3)
        try:
            ds._fetch_top10_max_speed_from_config = MagicMock(
                side_effect=[None, 321.0]
            )
            first = ds.fetch_top10_max_speed_observation(
                987_651,
                origin_as_of=123_456,
                allow_fallback=True,
                primary_retry=False,
                fallback_retry=False,
            )
            second = ds.fetch_top10_max_speed_observation(
                987_651,
                origin_as_of=123_456,
                allow_fallback=True,
                primary_retry=False,
                fallback_retry=False,
            )

            self.assertEqual(first.value, 321.0)
            self.assertEqual(first.source, "bestdori")
            self.assertTrue(first.fallback_used)
            self.assertFalse(first.cache_hit)
            self.assertEqual(first.origin_as_of, 123_456)
            self.assertEqual(second.fetched_at, first.fetched_at)
            self.assertEqual(second.source, first.source)
            self.assertTrue(second.cache_hit)
            self.assertEqual(
                ds._fetch_top10_max_speed_from_config.call_count, 2
            )
        finally:
            ds.close()


if __name__ == "__main__":
    unittest.main()
