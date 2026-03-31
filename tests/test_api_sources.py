import os
import sys
import unittest
from unittest.mock import MagicMock

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
            self.assertIn("hhwx.org/api/bestdori/eventtop/data", ds.api_config["top10_url"])
        finally:
            ds.close()

    def test_fetch_event_meta_normalizes_hhwx_proxy_payload(self):
        ds = create_data_source("hhwx")
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "eventType": "mission_live",
                "startAt": ["1000", None, None, "2000", None],
                "endAt": ["3000", None, None, "4000", None],
                "aggregateEndAt": ["5000", None, None, "6000", None],
            }
            ds.session.get = MagicMock(return_value=mock_response)

            meta = ds.fetch_event_meta(312)

            self.assertEqual(meta["event_id"], 312)
            self.assertEqual(meta["event_type"], "mission_live")
            self.assertEqual(meta["start_at"], 2000)
            self.assertEqual(meta["end_at"], 4000)
            self.assertEqual(meta["aggregate_at"], 6000)
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
            self.assertIn("https://hhwx.org/api/tracker/data", called_url)
            self.assertIn("event=312", called_url)
            self.assertIn("type=event", called_url)
            self.assertIn("tier=1000", called_url)
            self.assertListEqual(df["ep"].tolist(), [100, 120])
        finally:
            ds.close()


if __name__ == "__main__":
    unittest.main()
