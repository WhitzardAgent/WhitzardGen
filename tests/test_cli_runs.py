import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from whitzard.recovery import RecoveryPlan
from whitzard.run_flow import run_single_model


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
    def test_handle_prompts_generate_returns_bundle_summary(self) -> None:
        from whitzard.cli.main import handle_prompts_generate

        summary = type(
            "Summary",
            (),
            {
                "bundle_id": "prompt_bundle_001",
                "prompt_count": 12,
                "execution_mode": "mock",
                "llm_model": None,
                "prompt_template": "photorealistic_base",
                "prompt_style_family": "detailed_sentence",
                "target_model_name": None,
                "few_shot_example_count": 2,
                "bundle_dir": "/tmp/prompt_bundle_001",
                "prompts_path": "/tmp/prompt_bundle_001/prompts.jsonl",
                "manifest_path": "/tmp/prompt_bundle_001/prompt_manifest.json",
                "to_dict": lambda self: {
                    "bundle_id": "prompt_bundle_001",
                    "prompt_count": 12,
                    "execution_mode": "mock",
                    "llm_model": None,
                    "prompt_template": "photorealistic_base",
                    "prompt_style_family": "detailed_sentence",
                    "target_model_name": None,
                    "few_shot_example_count": 2,
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "tree": "prompts/theme_tree_example.yaml",
                "out": None,
                "count_config": None,
                "llm_model": None,
                "mock": True,
                "execution_mode": None,
                "seed": 42,
                "profile": None,
                "template": "photorealistic_base",
                "style_family": "detailed_sentence",
                "target_model": None,
                "intended_modality": None,
                "output": "json",
            },
        )()

        with patch("whitzard.cli.main.generate_prompt_bundle", return_value=summary):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_prompts_generate(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["bundle_id"], "prompt_bundle_001")
        self.assertEqual(payload["prompt_count"], 12)

    def test_handle_prompts_inspect_prints_bundle_summary(self) -> None:
        from whitzard.cli.main import handle_prompts_inspect

        args = type(
            "Args",
            (),
            {
                "path": "/tmp/prompt_bundle_001",
                "output": "text",
            },
        )()
        inspect_payload = {
            "bundle_dir": "/tmp/prompt_bundle_001",
            "manifest": {
                "bundle_id": "prompt_bundle_001",
                "tree_name": "realistic_video_prompts",
                "generation_profile": "photorealistic",
                "llm_model": "Qwen3-32B",
                "prompt_template": "photorealistic_base",
                "prompt_style_family": "detailed_sentence",
                "target_model_name": "Z-Image",
                "few_shot_example_count": 2,
            },
            "prompt_count": 8,
            "counts_by_category": {"Animals": 8},
        }

        with patch("whitzard.cli.main.inspect_prompt_bundle", return_value=inspect_payload):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_prompts_inspect(args), 0)

        output = stream.getvalue()
        self.assertIn("Prompt Count: 8", output)
        self.assertIn("Tree Name: realistic_video_prompts", output)
        self.assertIn("Template: photorealistic_base", output)
        self.assertIn("Style Family: detailed_sentence", output)

    def test_handle_models_canary_runs_one_model_with_requested_mode(self) -> None:
        from whitzard.cli.main import handle_models_canary

        summary = type(
            "Summary",
            (),
            {
                "status": "completed",
                "run_id": "canary-helios",
                "prompt_file": "prompts/canary_video.jsonl",
                "output_dir": "/tmp/canary-helios",
                "to_dict": lambda self: {
                    "status": "completed",
                    "run_id": "canary-helios",
                    "prompt_file": "prompts/canary_video.jsonl",
                    "output_dir": "/tmp/canary-helios",
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "model_name": "Helios",
                "prompt_file": None,
                "out": None,
                "mock": True,
                "execution_mode": None,
                "output": "json",
            },
        )()

        def fake_run_model_canary(**kwargs):
            self.assertEqual(kwargs["model_name"], "Helios")
            self.assertEqual(kwargs["execution_mode"], "mock")
            self.assertIsNone(kwargs["prompt_file"])
            return summary

        with patch("whitzard.cli.main.run_model_canary", side_effect=fake_run_model_canary):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_models_canary(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["run_id"], "canary-helios")
        self.assertEqual(payload["status"], "completed")

    def test_handle_annotate_uses_requested_profile_and_model(self) -> None:
        from whitzard.cli.main import handle_annotate

        summary = type(
            "Summary",
            (),
            {
                "bundle_dir": "/tmp/annotation_bundle",
                "source_run_id": "run_001",
                "annotator_model": "Qwen3-32B",
                "annotation_profile": "default_review",
                "annotation_template": "source_record_review_v1",
                "annotated_count": 12,
                "skipped_count": 2,
                "failed_count": 1,
                "annotations_path": "/tmp/annotation_bundle/annotations.jsonl",
                "manifest_path": "/tmp/annotation_bundle/annotation_manifest.json",
                "to_dict": lambda self: {
                    "bundle_dir": "/tmp/annotation_bundle",
                    "source_run_id": "run_001",
                    "annotator_model": "Qwen3-32B",
                    "annotation_profile": "default_review",
                    "annotation_template": "source_record_review_v1",
                    "annotated_count": 12,
                    "skipped_count": 2,
                    "failed_count": 1,
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "run_id": "run_001",
                "profile": "default_review",
                "model": "Qwen3-32B",
                "template": "source_record_review_v1",
                "out": "/tmp/annotation_bundle",
                "mock": True,
                "execution_mode": None,
                "output": "json",
            },
        )()

        def fake_annotate_run(run_id, **kwargs):
            self.assertEqual(run_id, "run_001")
            self.assertEqual(kwargs["annotation_profile"], "default_review")
            self.assertEqual(kwargs["annotator_model"], "Qwen3-32B")
            self.assertEqual(kwargs["template_name"], "source_record_review_v1")
            self.assertEqual(kwargs["out_dir"], "/tmp/annotation_bundle")
            self.assertEqual(kwargs["execution_mode"], "mock")
            return summary

        with patch("whitzard.cli.main.annotate_run", side_effect=fake_annotate_run):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_annotate(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["source_run_id"], "run_001")
        self.assertEqual(payload["annotator_model"], "Qwen3-32B")

    def test_handle_models_matrix_can_write_docs(self) -> None:
        from whitzard.cli.main import handle_models_matrix

        tmpdir = Path(tempfile.mkdtemp())
        args = type(
            "Args",
            (),
            {
                "output": "json",
                "write_docs": True,
                "docs_dir": str(tmpdir),
            },
        )()

        with redirect_stdout(StringIO()) as stream:
            self.assertEqual(handle_models_matrix(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertIn("rows", payload)
        self.assertTrue((tmpdir / "model_capability_matrix.md").exists())
        self.assertTrue((tmpdir / "model_capability_matrix.json").exists())
        self.assertTrue(any(row["name"] == "Helios" for row in payload["rows"]))
        self.assertTrue(
            any(
                row["name"] == "Z-Image" and row["supports_persistent_worker"]
                for row in payload["rows"]
            )
        )

    def test_handle_run_uses_profile_and_cli_overrides(self) -> None:
        from whitzard.cli.main import handle_run

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
                    "global_negative_prompt: low quality, blurry",
                    "generation_defaults:",
                    "  width: 1024",
                    "  num_inference_steps: 40",
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
                "continue_on_error": True,
                "max_failures": 5,
                "max_failure_rate": 0.25,
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
            self.assertEqual(
                kwargs["profile_generation_defaults"],
                {"width": 1024, "num_inference_steps": 40},
            )
            self.assertEqual(kwargs["profile_runtime"], {"available_gpus": [2, 3]})
            self.assertEqual(kwargs["profile_global_negative_prompt"], "low quality, blurry")
            self.assertEqual(kwargs["profile_conditionings"], [])
            self.assertEqual(kwargs["profile_prompt_rewrites"], [])
            self.assertTrue(kwargs["continue_on_error"])
            self.assertEqual(kwargs["max_failures"], 5)
            self.assertEqual(kwargs["max_failure_rate"], 0.25)
            self.assertEqual(os.environ.get("AIGC_AVAILABLE_GPUS"), "2,3")
            return summary

        with patch("whitzard.cli.main.run_models", side_effect=fake_run_models):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_run(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["run_id"], "run_profile")
        self.assertNotEqual(os.environ.get("AIGC_AVAILABLE_GPUS"), "2,3")

    def test_handle_run_rejects_mixed_modality_profile_early(self) -> None:
        from whitzard.cli.main import handle_run
        from whitzard.run_flow import RunFlowError

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
                "continue_on_error": False,
                "max_failures": None,
                "max_failure_rate": None,
                "output": "text",
            },
        )()

        with self.assertRaises(RunFlowError):
            handle_run(args)

    def test_doctor_text_output_shows_conda_env_and_path_existence(self) -> None:
        from whitzard.cli.main import handle_doctor

        class DoctorManager:
            def conda_available(self) -> bool:
                return True

            def doctor(self, model_name=None):
                self.last_model_name = model_name
                return [FakeDoctorRecord()]

        args = type("Args", (), {"model": "Z-Image", "output": "text"})()
        manager = DoctorManager()

        with patch("whitzard.cli.main.EnvManager", return_value=manager):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_doctor(args), 0)

        output = stream.getvalue()
        self.assertIn("Model: Z-Image", output)
        self.assertIn("Conda env: zimage", output)
        self.assertIn("Env exists: yes", output)
        self.assertIn("local_path_exists: yes", output)

    def test_run_store_commands_use_configured_runs_root(self) -> None:
        from whitzard.cli.main import handle_runs_inspect, handle_runs_list

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
        from whitzard.cli.main import handle_export_dataset, handle_runs_failures, handle_runs_inspect, handle_runs_list

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
            export_args = type(
                "Args",
                (),
                {
                    "run_ids": [summary.run_id],
                    "out": None,
                    "mode": "link",
                    "model": [],
                    "output": "json",
                },
            )()

            with redirect_stdout(StringIO()):
                self.assertEqual(handle_runs_list(list_args), 0)
                self.assertEqual(handle_runs_inspect(inspect_args), 0)
                self.assertEqual(handle_runs_failures(failures_args), 0)
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_export_dataset(export_args), 0)
            export_payload = json.loads(stream.getvalue())
            self.assertEqual(export_payload["source_run_ids"], [summary.run_id])
            self.assertEqual(export_payload["export_mode"], "link")
            self.assertTrue(Path(export_payload["bundle_path"]).exists())
            self.assertTrue(Path(export_payload["dataset_path"]).exists())
            self.assertTrue(Path(export_payload["manifest_path"]).exists())
            self.assertTrue(Path(export_payload["readme_path"]).exists())

    def test_handle_export_dataset_supports_multiple_runs_and_model_filter(self) -> None:
        from whitzard.cli.main import handle_export_dataset

        tmpdir = Path(tempfile.mkdtemp())
        runtime_config = tmpdir / "local_runtime.yaml"
        configured_root = tmpdir / "configured_runs"
        runtime_config.write_text(
            f"paths:\n  runs_root: {configured_root}\n",
            encoding="utf-8",
        )
        run_ids = ["run_001", "run_002"]
        model_map = {
            "run_001": ("Z-Image", "image", ".png", b"\x89PNG\r\n\x1a\n", "train"),
            "run_002": ("FLUX.1-dev", "image", ".png", b"\x89PNG\r\n\x1a\n", "val"),
        }
        for run_id in run_ids:
            run_root = configured_root / run_id
            export_root = run_root / "exports"
            artifact_root = run_root / "artifacts"
            export_root.mkdir(parents=True, exist_ok=True)
            artifact_root.mkdir(parents=True, exist_ok=True)
            model_name, artifact_type, suffix, content, split = model_map[run_id]
            artifact_path = artifact_root / f"{run_id}{suffix}"
            artifact_path.write_bytes(content)
            dataset_path = export_root / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "record_id": "rec_00000001",
                        "run_id": run_id,
                        "task_id": f"task_{run_id}",
                        "prompt_id": f"prompt_{run_id}",
                        "prompt": "sample",
                        "language": "en",
                        "model_name": model_name,
                        "model_version": "1.0",
                        "adapter_name": "Adapter",
                        "modality": "image",
                        "task_type": "t2i",
                        "artifact_type": artifact_type,
                        "artifact_path": str(artifact_path),
                        "artifact_metadata": {"format": "png"},
                        "generation_params": {},
                        "prompt_metadata": {"split": split},
                        "execution_metadata": {"status": "success", "execution_mode": "mock"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": "completed",
                        "execution_mode": "mock",
                        "models": [model_name],
                        "output_dir": str(run_root),
                        "export_path": str(dataset_path),
                    }
                ),
                encoding="utf-8",
            )

        args = type(
            "Args",
            (),
            {
                "run_ids": run_ids,
                "out": None,
                "mode": "link",
                "model": ["Z-Image"],
                "output": "json",
            },
        )()

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(runtime_config)}, clear=False):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_export_dataset(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["source_run_ids"], run_ids)
        self.assertEqual(payload["selected_models"], ["Z-Image"])
        self.assertEqual(payload["record_count"], 1)
        self.assertEqual(payload["filtered_out_count"], 1)
        dataset_records = [
            json.loads(line)
            for line in Path(payload["dataset_path"]).read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual([record["model_name"] for record in dataset_records], ["Z-Image"])

    def test_runs_retry_handler_executes_recovery_run(self) -> None:
        from whitzard.cli.main import handle_runs_retry

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

        with patch("whitzard.cli.main.build_retry_plan", return_value=plan), patch(
            "whitzard.cli.main.run_recovery_plan",
            return_value=summary,
        ):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_runs_retry(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["plan"]["recovery_mode"], "retry")
        self.assertEqual(payload["summary"]["run_id"], "retry_run_001")

    def test_runs_resume_handler_reports_no_missing_work(self) -> None:
        from whitzard.cli.main import handle_runs_resume

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

        with patch("whitzard.cli.main.build_resume_plan", return_value=plan):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_runs_resume(args), 0)
            output = stream.getvalue()

        self.assertIn("Inspecting run run_complete...", output)
        self.assertIn("Nothing to resume.", output)
