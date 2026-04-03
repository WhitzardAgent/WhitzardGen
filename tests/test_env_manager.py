import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from whitzard.env import EnvManager
from whitzard.env import manager as env_manager_module
from whitzard.env.manager import EnvManagerError
from whitzard.registry import load_registry


class FakeManualEnvManager(EnvManager):
    def __init__(
        self,
        metadata_path: Path,
        *,
        registry=None,
        available: bool = True,
        env_prefixes: list[str] | None = None,
        validation_result: tuple[bool, str | None] = (True, None),
        validation_ttl_sec: int | None = None,
    ) -> None:
        super().__init__(
            registry=registry,
            metadata_path=metadata_path,
            validation_ttl_sec=validation_ttl_sec,
        )
        self.available = available
        self.env_prefixes = env_prefixes or []
        self.validation_result = validation_result
        self.validate_calls = 0

    def conda_available(self) -> bool:
        return self.available

    def validate_environment(self, env_id: str, spec) -> tuple[bool, str | None]:
        del env_id, spec
        self.validate_calls += 1
        return self.validation_result

    def _list_conda_env_prefixes(self) -> list[str]:
        return list(self.env_prefixes)

    def _conda_base_prefix(self) -> str | None:
        return "/opt/conda"


class EnvManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.runtime_root = self.tmpdir / "runtime"
        self.conda_envs_root = self.runtime_root / "conda_envs"
        self.conda_pkgs_root = self.runtime_root / "conda_pkgs"
        self.conda_cache_root = self.runtime_root / "cache"
        self.patchers = [
            patch.object(env_manager_module, "RUNTIME_ROOT", self.runtime_root),
            patch.object(env_manager_module, "CONDA_ENVS_ROOT", self.conda_envs_root),
            patch.object(env_manager_module, "CONDA_PKGS_ROOT", self.conda_pkgs_root),
            patch.object(env_manager_module, "CONDA_CACHE_ROOT", self.conda_cache_root),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()

    def test_resolve_spec_for_model_exposes_default_conda_env_name(self) -> None:
        manager = EnvManager(metadata_path=self.tmpdir / "env_metadata.json")
        spec = manager.resolve_spec_for_model("Z-Image")

        self.assertEqual(spec.spec_name, "zimage")
        self.assertEqual(spec.conda_env_name, "zimage")
        self.assertIn("diffusers", spec.validation_imports)

    def test_local_override_can_replace_conda_env_name(self) -> None:
        local_models_path = self.tmpdir / "local_models.yaml"
        local_models_path.write_text(
            """
Z-Image:
  conda_env_name: zimage_cluster
  local_path: /models/Z-Image
""".strip()
            + "\n",
            encoding="utf-8",
        )
        registry = load_registry(local_models_path=local_models_path)
        manager = EnvManager(registry=registry, metadata_path=self.tmpdir / "env_metadata.json")

        spec = manager.resolve_spec_for_model("Z-Image")

        self.assertEqual(spec.conda_env_name, "zimage_cluster")
        self.assertEqual(registry.get_model("Z-Image").conda_env_name, "zimage_cluster")

    def test_compute_env_id_uses_conda_env_name(self) -> None:
        manager = EnvManager(metadata_path=self.tmpdir / "env_metadata.json")
        spec = manager.resolve_spec_for_model("Z-Image")

        self.assertEqual(manager.compute_env_id(spec), "zimage")

    def test_wrap_command_uses_conda_run_with_env_name(self) -> None:
        manager = EnvManager(metadata_path=self.tmpdir / "env_metadata.json")

        command = manager.wrap_command("zimage", ["python", "-V"])
        self.assertEqual(command[:4], ["conda", "run", "-n", "zimage"])
        self.assertEqual(command[-2:], ["python", "-V"])

        foreground_command = manager.wrap_command("zimage", ["python", "-V"], foreground=True)
        self.assertEqual(
            foreground_command[:5],
            ["conda", "run", "--no-capture-output", "-n", "zimage"],
        )

    def test_environment_exists_matches_conda_env_name(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=["/opt/conda/envs/zimage", "/opt/conda/envs/cogvideo"],
        )

        exists, path = manager.environment_exists("zimage")

        self.assertTrue(exists)
        self.assertEqual(path, "/opt/conda/envs/zimage")

    def test_inspect_model_environment_reports_missing_env(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=[],
        )

        record = manager.inspect_model_environment("Z-Image")

        self.assertEqual(record.conda_env_name, "zimage")
        self.assertEqual(record.state, "missing")
        self.assertFalse(record.exists)
        self.assertIn("Required conda env zimage was not found", record.error or "")

    def test_ensure_ready_fails_clearly_when_env_is_missing(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=[],
        )
        progress_messages: list[str] = []

        with self.assertRaises(EnvManagerError) as context:
            manager.ensure_ready("Z-Image", foreground=True, progress=progress_messages.append)

        self.assertIn("Required conda env: zimage", str(context.exception))
        self.assertIn("Please create this environment manually before running.", str(context.exception))
        self.assertIn("Environment for Z-Image is not available.", "\n".join(progress_messages))

    def test_ensure_ready_fails_clearly_when_env_is_invalid(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=["/opt/conda/envs/zimage"],
            validation_result=(False, "missing diffusers"),
        )

        with self.assertRaises(EnvManagerError) as context:
            manager.ensure_ready("Z-Image", foreground=False)

        self.assertIn("exists but failed validation", str(context.exception))
        self.assertIn("missing diffusers", str(context.exception))

    def test_inspect_model_environment_reuses_recent_ready_validation(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=["/opt/conda/envs/zimage"],
            validation_result=(True, None),
            validation_ttl_sec=3600,
        )
        manager.metadata_path.write_text(
            json.dumps(
                {
                    "zimage": {
                        "env_id": "zimage",
                        "env_spec": "zimage",
                        "conda_env_name": "zimage",
                        "models": ["Z-Image"],
                        "state": "ready",
                        "path": "/opt/conda/envs/zimage",
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

    def test_doctor_forces_revalidation_even_when_ready_validation_is_cached(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=["/opt/conda/envs/zimage"],
            validation_result=(True, None),
            validation_ttl_sec=3600,
        )
        manager.metadata_path.write_text(
            json.dumps(
                {
                    "zimage": {
                        "env_id": "zimage",
                        "env_spec": "zimage",
                        "conda_env_name": "zimage",
                        "models": ["Z-Image"],
                        "state": "ready",
                        "path": "/opt/conda/envs/zimage",
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
        self.assertEqual(records[0].conda_env_name, "zimage")
        self.assertEqual(manager.validate_calls, 1)

    def test_inspect_model_environment_revalidates_stale_ready_validation(self) -> None:
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            env_prefixes=["/opt/conda/envs/zimage"],
            validation_result=(True, None),
            validation_ttl_sec=60,
        )
        manager.metadata_path.write_text(
            json.dumps(
                {
                    "zimage": {
                        "env_id": "zimage",
                        "env_spec": "zimage",
                        "conda_env_name": "zimage",
                        "models": ["Z-Image"],
                        "state": "ready",
                        "path": "/opt/conda/envs/zimage",
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

    def test_inspect_model_environment_validates_remote_provider_config(self) -> None:
        local_models_path = self.tmpdir / "local_models.yaml"
        local_models_path.write_text(
            """
OpenAI-Compatible-Chat:
  provider:
    base_url: https://example.test/v1
    api_key_env: OPENAI_API_KEY
    model_name: example-chat-model
""".strip()
            + "\n",
            encoding="utf-8",
        )
        registry = load_registry(local_models_path=local_models_path)
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            registry=registry,
            env_prefixes=["/opt/conda/envs/api_client"],
            validation_result=(True, None),
        )

        class _FakeResponse(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch(
                "whitzard.env.manager.urllib.request.urlopen",
                return_value=_FakeResponse(b'{"data": []}'),
            ):
                record = manager.inspect_model_environment("OpenAI-Compatible-Chat")

        self.assertEqual(record.conda_env_name, "api_client")
        self.assertEqual(record.state, "ready")
        self.assertTrue(record.provider_checks["api_key_env"]["ok"])
        self.assertTrue(record.provider_checks["healthcheck"]["ok"])
        self.assertEqual(
            record.provider_checks["healthcheck"]["value"],
            "https://example.test/v1/models",
        )

    def test_inspect_model_environment_reports_missing_remote_api_key(self) -> None:
        local_models_path = self.tmpdir / "local_models.yaml"
        local_models_path.write_text(
            """
OpenAI-Compatible-Chat:
  provider:
    base_url: https://example.test/v1
    api_key_env: OPENAI_API_KEY
    model_name: example-chat-model
""".strip()
            + "\n",
            encoding="utf-8",
        )
        registry = load_registry(local_models_path=local_models_path)
        manager = FakeManualEnvManager(
            metadata_path=self.tmpdir / "env_metadata.json",
            registry=registry,
            env_prefixes=["/opt/conda/envs/api_client"],
            validation_result=(True, None),
        )

        with patch.dict("os.environ", {}, clear=False):
            record = manager.inspect_model_environment("OpenAI-Compatible-Chat")

        self.assertEqual(record.state, "invalid")
        self.assertFalse(record.provider_checks["api_key_env"]["ok"])
        self.assertIn("Remote provider validation failed", record.error or "")
