import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aigc.env import EnvManager


ROOT = Path(__file__).resolve().parents[1]


class EnvManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.manager = EnvManager(metadata_path=self.tmpdir / "env_metadata.json")

    def test_resolve_spec_for_model(self) -> None:
        spec = self.manager.resolve_spec_for_model("Z-Image")
        self.assertEqual(spec.spec_name, "zimage")
        self.assertTrue(spec.environment_file.exists())
        self.assertIn("diffusers", spec.validation_imports)

    def test_compute_env_id_is_stable(self) -> None:
        spec = self.manager.resolve_spec_for_model("Z-Image")
        env_id_a = self.manager.compute_env_id(spec)
        env_id_b = self.manager.compute_env_id(spec)
        self.assertEqual(env_id_a, env_id_b)
        self.assertTrue(env_id_a.startswith("env_zimage_"))

    def test_wrap_command(self) -> None:
        command = self.manager.wrap_command("env_demo", ["python", "-V"])
        self.assertEqual(command[:3], ["conda", "run", "--prefix"])
        self.assertEqual(command[-2:], ["python", "-V"])

    def test_doctor_json_output(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "aigc", "doctor", "--model", "Z-Image", "--output", "json"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(len(payload["records"]), 1)
        self.assertEqual(payload["records"][0]["model_name"], "Z-Image")
        self.assertIn("path_checks", payload["records"][0])
