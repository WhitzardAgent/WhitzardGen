import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whitzard.settings import (
    get_benchmarks_root,
    get_default_seed,
    get_experiments_root,
    get_runs_root,
    load_runtime_settings,
)


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
        self.assertIsNone(settings.default_seed)
        self.assertEqual(settings.config_path, config_path)

    def test_load_runtime_settings_reads_default_seed(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "local_runtime.yaml"
        config_path.write_text(
            json.dumps(
                {
                    "paths": {"runs_root": str(tmpdir / "shared_runs")},
                    "generation": {"default_seed": 321},
                }
            ),
            encoding="utf-8",
        )

        settings = load_runtime_settings(config_path)

        self.assertEqual(settings.default_seed, 321)
        self.assertEqual(get_default_seed(config_path), 321)

    def test_get_runs_root_honors_env_override(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "local_runtime.yaml"
        config_path.write_text(
            "paths:\n  runs_root: /tmp/whitzard_runs_env\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(config_path)}, clear=False):
            self.assertEqual(get_runs_root(), Path("/tmp/whitzard_runs_env"))

    def test_benchmark_and_experiment_roots_default_under_runs_root(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "local_runtime.yaml"
        config_path.write_text(
            json.dumps({"paths": {"runs_root": str(tmpdir / "shared_runs")}}),
            encoding="utf-8",
        )

        settings = load_runtime_settings(config_path)

        self.assertEqual(settings.benchmarks_root, tmpdir / "shared_runs" / "benchmarks")
        self.assertEqual(settings.experiments_root, tmpdir / "shared_runs" / "experiments")
        self.assertEqual(get_benchmarks_root(config_path), tmpdir / "shared_runs" / "benchmarks")
        self.assertEqual(get_experiments_root(config_path), tmpdir / "shared_runs" / "experiments")
