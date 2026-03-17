import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aigc.settings import get_runs_root, load_runtime_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_load_runtime_settings_uses_runs_root_from_yaml(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "local_runtime.yaml"
        config_path.write_text(
            json.dumps({"paths": {"runs_root": str(tmpdir / "shared_runs")}}),
            encoding="utf-8",
        )

        settings = load_runtime_settings(config_path)

        self.assertEqual(settings.runs_root, tmpdir / "shared_runs")
        self.assertEqual(settings.config_path, config_path)

    def test_get_runs_root_honors_env_override(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "local_runtime.yaml"
        config_path.write_text(
            "paths:\n  runs_root: /tmp/aigc_runs_env\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(config_path)}, clear=False):
            self.assertEqual(get_runs_root(), Path("/tmp/aigc_runs_env"))
