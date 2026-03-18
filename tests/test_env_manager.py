import json
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from aigc.env import EnvManager
from aigc.env import manager as env_manager_module
from aigc.env.manager import EnvManagerError, MissingEnvironmentError


ROOT = Path(__file__).resolve().parents[1]


class EnvManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.runtime_root = self.tmpdir / "runtime"
        self.patchers = [
            patch.object(env_manager_module, "RUNTIME_ROOT", self.runtime_root),
        ]
        for patcher in self.patchers:
            patcher.start()
        self.manager = EnvManager(metadata_path=self.tmpdir / "env_metadata.json")

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()

    def test_resolve_conda_env_name_from_model(self) -> None:
        env_name = self.manager.resolve_conda_env_name("Z-Image")
        self.assertEqual(env_name, "zimage")

    def test_resolve_conda_env_name_for_cogvideox(self) -> None:
        env_name = self.manager.resolve_conda_env_name("CogVideoX-5B")
        self.assertEqual(env_name, "cogvideox")

    def test_resolve_conda_env_name_for_hunyuan_video(self) -> None:
        env_name = self.manager.resolve_conda_env_name("HunyuanVideo-1.5")
        self.assertEqual(env_name, "hunyuan_video")

    def test_resolve_spec_for_model(self) -> None:
        spec = self.manager.resolve_spec_for_model("Z-Image")
        self.assertEqual(spec.spec_name, "zimage")
        self.assertTrue(spec.python_version_file.exists())
        self.assertEqual(spec.python_version, "3.10")
        self.assertTrue(spec.requirements_file is not None and spec.requirements_file.exists())
        self.assertIn("diffusers", spec.validation_imports)

    def test_resolve_spec_for_cogvideox_includes_tokenizer_dependencies(self) -> None:
        spec = self.manager.resolve_spec_for_model("CogVideoX-5B")
        self.assertEqual(spec.spec_name, "cogvideox_5b")
        self.assertTrue(spec.requirements_file is not None and spec.requirements_file.exists())
        requirements_text = spec.requirements_file.read_text(encoding="utf-8")
        self.assertIn("tiktoken", requirements_text)
        self.assertIn("sentencepiece", requirements_text)

    def test_resolve_spec_for_wan_repo_runtime_includes_easydict(self) -> None:
        spec = self.manager.resolve_spec_for_model("Wan2.2-T2V-A14B-Diffusers")
        self.assertEqual(spec.spec_name, "wan_t2v_diffusers")
        self.assertTrue(spec.requirements_file is not None and spec.requirements_file.exists())
        requirements_text = spec.requirements_file.read_text(encoding="utf-8")
        self.assertIn("easydict", requirements_text)

    def test_wrap_command_uses_conda_run_with_env_name(self) -> None:
        command = self.manager.wrap_command("zimage", ["python", "-V"])
        self.assertEqual(command[:4], ["conda", "run", "-n", "zimage"])
        self.assertEqual(command[-2:], ["python", "-V"])

    def test_wrap_command_with_foreground(self) -> None:
        command = self.manager.wrap_command("zimage", ["python", "-V"], foreground=True)
        self.assertEqual(command[:5], ["conda", "run", "--no-capture-output", "-n", "zimage"])
        self.assertEqual(command[-2:], ["python", "-V"])

    def test_local_env_override_rewrites_github_requirement(self) -> None:
        local_envs_path = self.tmpdir / "local_envs.yaml"
        local_envs_path.write_text(
            """
envs:
  zimage:
    pip_requirement_overrides:
      diffusers: /wheelhouse/diffusers-0.35.0-py3-none-any.whl
      git+https://github.com/huggingface/diffusers: /wheelhouse/diffusers-0.35.0-py3-none-any.whl
    pip_install_args:
      - --no-index
      - --find-links
      - /wheelhouse
""".strip()
            + "\n",
            encoding="utf-8",
        )
        manager = EnvManager(
            metadata_path=self.tmpdir / "env_metadata_override.json",
            local_envs_path=local_envs_path,
        )

        spec = manager.resolve_spec_for_model("Z-Image")

        self.assertEqual(spec.local_override_source, str(local_envs_path))
        self.assertEqual(spec.pip_install_args, ["--no-index", "--find-links", "/wheelhouse"])

    def test_local_env_override_can_replace_requirements_file(self) -> None:
        alternate_requirements = self.tmpdir / "zimage_alt_requirements.txt"
        alternate_requirements.write_text("torch\n/local/wheels/diffusers.whl\n", encoding="utf-8")
        local_envs_path = self.tmpdir / "local_envs_requirements.yaml"
        local_envs_path.write_text(
            f"""
envs:
  zimage:
    requirements_file: {alternate_requirements}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        manager = EnvManager(
            metadata_path=self.tmpdir / "env_metadata_override_file.json",
            local_envs_path=local_envs_path,
        )

        spec = manager.resolve_spec_for_model("Z-Image")

        self.assertEqual(spec.requirements_file, alternate_requirements)

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
        self.assertIn("conda_env_name", payload["records"][0])
        self.assertEqual(payload["records"][0]["conda_env_name"], "zimage")
        self.assertIn("exists", payload["records"][0])

    def test_ensure_ready_raises_missing_environment_error_when_env_missing(self) -> None:
        manager = FakeEnvManager(
            metadata_path=self.tmpdir / "env_metadata_missing.json",
            env_exists=False,
        )

        with self.assertRaises(MissingEnvironmentError) as context:
            manager.ensure_ready("Z-Image", foreground=True)

        self.assertIn("Z-Image", str(context.exception))
        self.assertIn("zimage", str(context.exception))
        self.assertIn("Please create this environment manually", str(context.exception))

    def test_ensure_ready_returns_ready_when_env_exists(self) -> None:
        manager = FakeEnvManager(
            metadata_path=self.tmpdir / "env_metadata_ready.json",
            env_exists=True,
            validation_passes=True,
        )

        record = manager.ensure_ready("Z-Image", foreground=True)

        self.assertEqual(record.state, "ready")
        self.assertTrue(record.exists)

    def test_inspect_model_environment_shows_conda_env_name(self) -> None:
        manager = FakeEnvManager(
            metadata_path=self.tmpdir / "env_metadata_inspect.json",
            env_exists=True,
            validation_passes=True,
        )

        record = manager.inspect_model_environment("Z-Image")

        self.assertEqual(record.conda_env_name, "zimage")
        self.assertEqual(record.state, "ready")
        self.assertTrue(record.exists)

    def test_inspect_model_environment_reuses_cached_validation(self) -> None:
        manager = FakeEnvManager(
            metadata_path=self.tmpdir / "env_metadata_cached.json",
            env_exists=True,
            validation_passes=True,
            validation_ttl_sec=3600,
        )
        metadata_key = "env_zimage"
        manager.metadata_path.write_text(
            json.dumps(
                {
                    metadata_key: {
                        "conda_env_name": "zimage",
                        "env_spec": "zimage",
                        "models": ["Z-Image"],
                        "state": "ready",
                        "path": "/path/to/zimage",
                        "last_validation": {
                            "passed": True,
                            "error": None,
                            "checked_at": datetime.now(UTC).isoformat(),
                        },
                        "error": None,
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        record = manager.inspect_model_environment("Z-Image")

        self.assertEqual(record.state, "ready")
        self.assertEqual(manager.validate_calls, 0)

    def test_doctor_forces_revalidation(self) -> None:
        manager = FakeEnvManager(
            metadata_path=self.tmpdir / "env_metadata_doctor.json",
            env_exists=True,
            validation_passes=True,
            validation_ttl_sec=3600,
        )
        metadata_key = "env_zimage"
        manager.metadata_path.write_text(
            json.dumps(
                {
                    metadata_key: {
                        "conda_env_name": "zimage",
                        "env_spec": "zimage",
                        "models": ["Z-Image"],
                        "state": "ready",
                        "path": "/path/to/zimage",
                        "last_validation": {
                            "passed": True,
                            "error": None,
                            "checked_at": datetime.now(UTC).isoformat(),
                        },
                        "error": None,
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        records = manager.doctor(model_name="Z-Image")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, "ready")
        self.assertEqual(manager.validate_calls, 1)

    def test_inspect_model_environment_revalidates_stale_validation(self) -> None:
        manager = FakeEnvManager(
            metadata_path=self.tmpdir / "env_metadata_stale.json",
            env_exists=True,
            validation_passes=True,
            validation_ttl_sec=60,
        )
        metadata_key = "env_zimage"
        manager.metadata_path.write_text(
            json.dumps(
                {
                    metadata_key: {
                        "conda_env_name": "zimage",
                        "env_spec": "zimage",
                        "models": ["Z-Image"],
                        "state": "ready",
                        "path": "/path/to/zimage",
                        "last_validation": {
                            "passed": True,
                            "error": None,
                            "checked_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
                        },
                        "error": None,
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        record = manager.inspect_model_environment("Z-Image")

        self.assertEqual(record.state, "ready")
        self.assertEqual(manager.validate_calls, 1)


class FakeEnvManager(EnvManager):
    def __init__(
        self,
        metadata_path: Path,
        env_exists: bool = True,
        validation_passes: bool = True,
        validation_ttl_sec: int | None = None,
    ) -> None:
        super().__init__(
            metadata_path=metadata_path,
            validation_ttl_sec=validation_ttl_sec,
        )
        self._env_exists = env_exists
        self._validation_passes = validation_passes
        self.validate_calls = 0

    def conda_available(self) -> bool:
        return True

    def conda_env_exists(self, env_name: str) -> tuple[bool, str | None]:
        if self._env_exists:
            return True, f"/path/to/{env_name}"
        return False, None

    def validate_environment(self, conda_env_name: str, spec) -> tuple[bool, str | None]:
        self.validate_calls += 1
        if self._validation_passes:
            return True, None
        return False, "validation failed"


if __name__ == "__main__":
    unittest.main()
