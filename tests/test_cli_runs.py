import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from aigc.recovery import RecoveryPlan
from aigc.run_flow import run_single_model


class FakeEnvRecord:
    env_id = "env_test"
    state = "ready"


class FakeEnvManager:
    def ensure_environment(self, model_name: str):
        return FakeEnvRecord()

    def inspect_model_environment(self, model_name: str):
        return FakeEnvRecord()


@dataclass
class FakeDoctorRecord:
    model_name: str = "Z-Image"
    state: str = "ready"
    conda_env_name: str = "zimage"
    exists: bool = True
    path: str | None = "/opt/conda/envs/zimage"
    last_validation: dict = field(
        default_factory=lambda: {
            "passed": True,
            "error": None,
            "checked_at": "2026-03-18T09:00:00+00:00",
        }
    )
    error: str | None = None
    path_checks: dict = field(
        default_factory=lambda: {
            "local_path": {"value": "/models/Z-Image", "exists": True, "kind": "directory"}
        }
    )
    local_paths: dict = field(default_factory=lambda: {"local_path": "/models/Z-Image"})

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "state": self.state,
            "conda_env_name": self.conda_env_name,
            "exists": self.exists,
            "path": self.path,
            "last_validation": self.last_validation,
            "error": self.error,
            "path_checks": self.path_checks,
            "local_paths": self.local_paths,
        }


class RunsCliTests(unittest.TestCase):
    def test_handle_run_uses_profile_and_cli_overrides(self) -> None:
        from aigc.cli.main import handle_run

        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "profile_prompts.txt"
        prompts_path.write_text("a neon city\n", encoding="utf-8")
        profile_path = tmpdir / "image_mock.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "name: image_mock",
                    "models: [Z-Image, FLUX.1-dev]",
                    "prompts: profile_prompts.txt",
                    "execution_mode: real",
                    "runtime:",
                    "  available_gpus: [2, 3]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        summary = type("Summary", (), {"to_dict": lambda self: {"run_id": "run_profile"}})()
        args = type(
            "Args",
            (),
            {
                "profile": str(profile_path),
                "models": "Z-Image",
                "prompts": None,
                "run_name": None,
                "out": None,
                "mock": True,
                "execution_mode": None,
                "output": "json",
            },
        )()

        def fake_run_models(**kwargs):
            self.assertEqual(kwargs["model_names"], ["Z-Image"])
            self.assertEqual(kwargs["prompt_file"], prompts_path.resolve())
            self.assertEqual(kwargs["execution_mode"], "mock")
            self.assertEqual(kwargs["run_name"], "image_mock")
            self.assertEqual(kwargs["profile_name"], "image_mock")
            self.assertEqual(kwargs["profile_path"], str(profile_path))
            self.assertEqual(kwargs["profile_runtime"], {"available_gpus": [2, 3]})
            self.assertEqual(os.environ.get("AIGC_AVAILABLE_GPUS"), "2,3")
            return summary

        with patch("aigc.cli.main.run_models", side_effect=fake_run_models):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_run(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["run_id"], "run_profile")
        self.assertNotEqual(os.environ.get("AIGC_AVAILABLE_GPUS"), "2,3")

    def test_handle_run_rejects_mixed_modality_profile_early(self) -> None:
        from aigc.cli.main import handle_run
        from aigc.run_flow import RunFlowError

        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "profile_prompts.txt"
        prompts_path.write_text("a neon city\n", encoding="utf-8")
        profile_path = tmpdir / "mixed.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "models:",
                    "  - Z-Image",
                    "  - CogVideoX-5B",
                    "prompts: profile_prompts.txt",
                    "execution_mode: mock",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        args = type(
            "Args",
            (),
            {
                "profile": str(profile_path),
                "models": None,
                "prompts": None,
                "run_name": None,
                "out": None,
                "mock": False,
                "execution_mode": None,
                "output": "text",
            },
        )()

        with self.assertRaises(RunFlowError):
            handle_run(args)

    def test_doctor_text_output_shows_conda_env_and_path_existence(self) -> None:
        from aigc.cli.main import handle_doctor

        class DoctorManager:
            def conda_available(self) -> bool:
                return True

            def doctor(self, model_name=None):
                self.last_model_name = model_name
                return [FakeDoctorRecord()]

        args = type("Args", (), {"model": "Z-Image", "output": "text"})()
        manager = DoctorManager()

        with patch("aigc.cli.main.EnvManager", return_value=manager):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_doctor(args), 0)

        output = stream.getvalue()
        self.assertIn("Model: Z-Image", output)
        self.assertIn("Conda env: zimage", output)
        self.assertIn("Env exists: yes", output)
        self.assertIn("local_path_exists: yes", output)

    def test_run_store_commands_use_configured_runs_root(self) -> None:
        from aigc.cli.main import handle_runs_inspect, handle_runs_list

        tmpdir = Path(tempfile.mkdtemp())
        runtime_config = tmpdir / "local_runtime.yaml"
        configured_root = tmpdir / "configured_runs"
        runtime_config.write_text(
            f"paths:\n  runs_root: {configured_root}\n",
            encoding="utf-8",
        )
        run_root = configured_root / "run_configured"
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": "run_configured",
            "status": "completed",
            "execution_mode": "mock",
            "models": ["Z-Image"],
            "prompt_source": "prompts/example.txt",
            "prompt_count": 1,
            "task_count": 1,
            "output_dir": str(run_root),
            "records_exported": 1,
            "export_path": str(run_root / "exports" / "dataset.jsonl"),
        }
        (run_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        list_args = type("Args", (), {"output": "json"})()
        inspect_args = type("Args", (), {"run_id": "run_configured", "output": "json"})()

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(runtime_config)}, clear=False):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_runs_list(list_args), 0)
                listed = json.loads(stream.getvalue())
            self.assertEqual(listed[0]["run_id"], "run_configured")

            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_runs_inspect(inspect_args), 0)
                inspected = json.loads(stream.getvalue())
            self.assertEqual(inspected["output_dir"], str(run_root))

    def test_run_store_commands_have_manifest_and_dataset_visibility(self) -> None:
        from aigc.cli.main import handle_export_dataset, handle_runs_failures, handle_runs_inspect, handle_runs_list

        tmpdir = Path(tempfile.mkdtemp())
        runtime_config = tmpdir / "local_runtime.yaml"
        configured_root = tmpdir / "configured_runs"
        runtime_config.write_text(
            f"paths:\n  runs_root: {configured_root}\n",
            encoding="utf-8",
        )
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text("a calm lake at sunrise\n", encoding="utf-8")

        def fake_worker_runner(_env_record, task_file: Path, result_file: Path):
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            workdir = Path(task_payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            artifact_path = workdir / f"{task_payload['prompts'][0]['prompt_id']}.png"
            artifact_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + b"\x00\x00\x00\x01"
                + b"\x00\x00\x00\x01"
                + b"\x08\x02\x00\x00\x00"
            )
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "execution_mode": task_payload["execution_mode"],
                "plan": {"mode": "in_process"},
                "execution_result": {"exit_code": 0, "logs": "ok", "outputs": {}},
                "model_result": {
                    "status": "success",
                    "batch_items": [
                        {
                            "prompt_id": task_payload["prompts"][0]["prompt_id"],
                            "status": "success",
                            "metadata": {"execution_mode": task_payload["execution_mode"]},
                            "artifacts": [
                                {
                                    "type": "image",
                                    "path": str(artifact_path),
                                    "metadata": {"width": 1, "height": 1, "format": "png"},
                                }
                            ],
                        }
                    ],
                    "logs": "ok",
                    "metadata": {},
                },
            }
            result_file.write_text(json.dumps(result_payload), encoding="utf-8")
            return 0, "ok"

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(runtime_config)}, clear=False):
            summary = run_single_model(
                model_name="Z-Image",
                prompt_file=prompts_path,
                run_name="cli-runs-test",
                execution_mode="mock",
                env_manager=FakeEnvManager(),
                worker_runner=fake_worker_runner,
            )

            list_args = type("Args", (), {"output": "json"})()
            inspect_args = type("Args", (), {"run_id": summary.run_id, "output": "json"})()
            failures_args = type("Args", (), {"run_id": summary.run_id, "output": "json"})()
            export_args = type("Args", (), {"run_id": summary.run_id, "out": None, "output": "json"})()

            with redirect_stdout(StringIO()):
                self.assertEqual(handle_runs_list(list_args), 0)
                self.assertEqual(handle_runs_inspect(inspect_args), 0)
                self.assertEqual(handle_runs_failures(failures_args), 0)
                self.assertEqual(handle_export_dataset(export_args), 0)

    def test_runs_retry_handler_executes_recovery_run(self) -> None:
        from aigc.cli.main import handle_runs_retry

        plan = RecoveryPlan(
            recovery_mode="retry",
            source_run_id="run_failed",
            execution_mode="mock",
            prompt_source="prompts/example.txt",
            items_by_model={"Z-Image": []},
            selected_count=2,
            completed_count=1,
            failed_count=2,
            missing_count=0,
            source_manifest={"run_id": "run_failed"},
        )
        summary = type(
            "Summary",
            (),
            {
                "to_dict": lambda self: {
                    "run_id": "retry_run_001",
                    "model_names": ["Z-Image"],
                    "output_dir": "/tmp/retry_run_001",
                    "tasks_scheduled": 2,
                    "records_exported": 2,
                    "export_path": "/tmp/retry_run_001/exports/dataset.jsonl",
                    "execution_mode": "mock",
                }
            },
        )()
        args = type("Args", (), {"run_id": "run_failed", "model": None, "output": "json"})()

        with patch("aigc.cli.main.build_retry_plan", return_value=plan), patch(
            "aigc.cli.main.run_recovery_plan",
            return_value=summary,
        ):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_runs_retry(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["plan"]["recovery_mode"], "retry")
        self.assertEqual(payload["summary"]["run_id"], "retry_run_001")

    def test_runs_resume_handler_reports_no_missing_work(self) -> None:
        from aigc.cli.main import handle_runs_resume

        plan = RecoveryPlan(
            recovery_mode="resume",
            source_run_id="run_complete",
            execution_mode="mock",
            prompt_source="prompts/example.txt",
            items_by_model={},
            selected_count=0,
            completed_count=4,
            failed_count=0,
            missing_count=0,
            source_manifest={"run_id": "run_complete"},
        )
        args = type("Args", (), {"run_id": "run_complete", "model": None, "output": "text"})()

        with patch("aigc.cli.main.build_resume_plan", return_value=plan):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_runs_resume(args), 0)
            output = stream.getvalue()

        self.assertIn("Inspecting run run_complete...", output)
        self.assertIn("Nothing to resume.", output)
