import os
import sys
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_API_SOURCE, DEFAULT_CONFIG
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

    def test_fetch_top10_max_speed_falls_back_to_bestdori_when_hhwx_is_empty(self):
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
                scale = ds.fetch_top10_max_speed(301)

            self.assertAlmostEqual(scale, ((6000 + 6200 + 6100) / 3) / 60.0)
            self.assertEqual(mock_get_json.call_count, 2)
        finally:
            ds.close()


if __name__ == "__main__":
    unittest.main()
