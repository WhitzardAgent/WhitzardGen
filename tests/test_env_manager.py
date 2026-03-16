import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aigc.env import EnvManager
from aigc.env import manager as env_manager_module
from aigc.env.manager import EnvManagerError


ROOT = Path(__file__).resolve().parents[1]


class EnvManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.runtime_root = self.tmpdir / "runtime"
        self.locks_root = self.runtime_root / "locks"
        self.conda_envs_root = self.runtime_root / "conda_envs"
        self.conda_pkgs_root = self.runtime_root / "conda_pkgs"
        self.conda_cache_root = self.runtime_root / "cache"
        self.patchers = [
            patch.object(env_manager_module, "RUNTIME_ROOT", self.runtime_root),
            patch.object(env_manager_module, "LOCKS_ROOT", self.locks_root),
            patch.object(env_manager_module, "CONDA_ENVS_ROOT", self.conda_envs_root),
            patch.object(env_manager_module, "CONDA_PKGS_ROOT", self.conda_pkgs_root),
            patch.object(env_manager_module, "CONDA_CACHE_ROOT", self.conda_cache_root),
        ]
        for patcher in self.patchers:
            patcher.start()
        self.manager = EnvManager(metadata_path=self.tmpdir / "env_metadata.json")

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()

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

    def test_ensure_ready_transitions_missing_to_ready(self) -> None:
        manager = FakeForegroundEnvManager(metadata_path=self.tmpdir / "env_metadata_ready.json")
        progress_messages: list[str] = []

        record = manager.ensure_ready(
            "Z-Image-Turbo",
            foreground=True,
            progress=progress_messages.append,
        )

        metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
        env_state = metadata[record.env_id]["state"]
        self.assertEqual(record.state, "ready")
        self.assertEqual(env_state, "ready")
        self.assertIn("Creating Conda environment", "\n".join(progress_messages))
        self.assertIn("Validating environment...", progress_messages)
        self.assertIn(f"Environment ready: {record.env_id}", progress_messages)

    def test_ensure_ready_transitions_missing_to_failed_on_creation_error(self) -> None:
        manager = FakeForegroundEnvManager(
            metadata_path=self.tmpdir / "env_metadata_failed.json",
            fail_stage="create",
        )
        progress_messages: list[str] = []

        with self.assertRaises(EnvManagerError):
            manager.ensure_ready(
                "Z-Image-Turbo",
                foreground=True,
                progress=progress_messages.append,
            )

        metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
        env_id = next(iter(metadata))
        self.assertEqual(metadata[env_id]["state"], "failed")
        self.assertIn("Creating Conda environment", "\n".join(progress_messages))

    def test_ensure_ready_recovers_stale_creating_state(self) -> None:
        manager = FakeForegroundEnvManager(metadata_path=self.tmpdir / "env_metadata_stale.json")
        spec = manager.resolve_spec_for_model("Z-Image-Turbo")
        env_id = manager.compute_env_id(spec)
        manager.metadata_path.write_text(
            json.dumps(
                {
                    env_id: {
                        "env_id": env_id,
                        "env_spec": spec.spec_name,
                        "models": ["Z-Image-Turbo"],
                        "state": "creating",
                        "path": None,
                        "last_validation": {},
                        "error": None,
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        progress_messages: list[str] = []

        record = manager.ensure_ready(
            "Z-Image-Turbo",
            foreground=True,
            progress=progress_messages.append,
            wait_timeout_sec=0.01,
            poll_interval_sec=0.01,
        )

        self.assertEqual(record.state, "ready")
        self.assertIn("Recovering stale creating state", "\n".join(progress_messages))


class FakeForegroundEnvManager(EnvManager):
    def __init__(self, metadata_path: Path, fail_stage: str | None = None) -> None:
        super().__init__(metadata_path=metadata_path)
        self.fail_stage = fail_stage

    def conda_available(self) -> bool:
        return True

    def validate_environment(self, env_id: str, spec) -> tuple[bool, str | None]:
        if self.fail_stage == "validate":
            return False, "validation failed"
        exists, _env_path = self.environment_exists(env_id)
        return (exists, None) if exists else (False, "environment missing")

    def _run_command(self, command, *, env, foreground, cwd=None):
        env_text = " ".join(str(item) for item in command)
        if self.fail_stage == "create" and command[:3] == ["conda", "env", "create"]:
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="conda env create failed", stderr="")
        if self.fail_stage == "pip" and "pip install" in env_text:
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="pip install failed", stderr="")

        if command[:3] == ["conda", "env", "create"]:
            prefix = Path(command[command.index("--prefix") + 1])
            prefix.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")
