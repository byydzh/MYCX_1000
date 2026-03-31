import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG, DEFAULT_MODEL_ID, list_models, list_presets, load_preset


class TestConfigLoading(unittest.TestCase):
    def test_list_models_includes_default_model(self):
        models = list_models()
        model_ids = {model["id"] for model in models}

        self.assertIn(DEFAULT_MODEL_ID, model_ids)

    def test_list_presets_returns_expected_files(self):
        presets = list_presets(DEFAULT_MODEL_ID)
        preset_ids = {preset["id"] for preset in presets}
        preset_names = {preset["id"]: preset["name"] for preset in presets}

        self.assertIn("default", preset_ids)
        self.assertIn("learned_notebook", preset_ids)
        self.assertNotIn("conservative", preset_ids)
        self.assertNotIn("aggressive", preset_ids)
        self.assertEqual(preset_names["default"], "早期手动参数配置")
        self.assertEqual(preset_names["learned_notebook"], "最新基于学习的参数配置")

    def test_load_preset_missing_file_falls_back_to_default(self):
        loaded = load_preset(DEFAULT_MODEL_ID, "does_not_exist")

        self.assertEqual(loaded, DEFAULT_CONFIG)

    def test_load_preset_merges_partial_params_with_default_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preset_path = Path(tmpdir) / "partial.json"
            preset_path.write_text(
                json.dumps(
                    {
                        "_meta": {"name": "Partial"},
                        "params": {
                            "ratio_max": 9.9,
                            "refit_lambda": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "config._resolve_model_entry",
                return_value={"id": "unit_test_model", "config_dir": Path(tmpdir)},
            ):
                loaded = load_preset("unit_test_model", "partial")

        self.assertEqual(loaded["ratio_max"], 9.9)
        self.assertEqual(loaded["refit_lambda"], 0.8)
        self.assertEqual(loaded["smooth_hard_cap"], DEFAULT_CONFIG["smooth_hard_cap"])
        self.assertEqual(loaded["ignore_event_ids"], DEFAULT_CONFIG["ignore_event_ids"])


if __name__ == "__main__":
    unittest.main()
