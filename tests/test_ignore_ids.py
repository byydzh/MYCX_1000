import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_source import BestdoriDataSource

class TestIgnoreEventIds(unittest.TestCase):
    def setUp(self):
        self.ds = BestdoriDataSource()
        # Mock the session to prevent actual network calls
        self.ds.session = MagicMock()

    def test_find_similar_events_ignores_specified_ids(self):
        """Test that find_similar_events correctly filters out ignored IDs."""
        
        # 1. Mock the fast index response (all.3.json)
        # We simulate a list of events, including the ones we want to ignore
        mock_index = {
            "300": {"eventType": "medley", "eventName": ["Event 300"]},
            "299": {"eventType": "medley", "eventName": ["Event 299"]},
            "298": {"eventType": "medley", "eventName": ["Event 298"]}, # Should be ignored
            "297": {"eventType": "medley", "eventName": ["Event 297"]}, # Should be ignored
            "296": {"eventType": "medley", "eventName": ["Event 296"]},
        }
        
        # Mock the response for fetch_recent_json/all.3.json
        # Since find_similar_events calls session.get for the index
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = mock_index
        self.ds.session.get.return_value = mock_response

        # 2. Mock fetch_event_data_pack to return valid dummy data for any ID
        # This ensures that if an ID is NOT ignored, it would be returned as a valid result
        def side_effect_fetch(event_id, tier=1000):
            return {
                'event_id': event_id,
                'meta': {'event_type': 'medley'},
                'dataframe': MagicMock(),
                'scale': 1000.0,
                'tier': tier,
            }
        
        # We need to patch the method on the instance or class. 
        # Since find_similar_events calls self.fetch_event_data_pack, we patch it on the instance.
        self.ds.fetch_event_data_pack = MagicMock(side_effect=side_effect_fetch)

        # 3. Define inputs
        target_event_id = 301
        event_type = "medley"
        ignore_ids = [297, 298]
        count = 5

        # 4. Run the method
        results = self.ds.find_similar_events(target_event_id, event_type, count, ignore_ids)
        
        # 5. Assertions
        result_ids = [r['event_id'] for r in results]
        print(f"Found events: {result_ids}")

        # Check that ignored IDs are NOT in the results
        self.assertNotIn(297, result_ids, "Event 297 should be ignored")
        self.assertNotIn(298, result_ids, "Event 298 should be ignored")
        
        # Check that other valid IDs ARE in the results
        self.assertIn(300, result_ids)
        self.assertIn(299, result_ids)
        self.assertIn(296, result_ids)

    def tearDown(self):
        self.ds.close()

if __name__ == '__main__':
    unittest.main()
