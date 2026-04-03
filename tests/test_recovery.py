import json
import tempfile
import unittest
from pathlib import Path

from whitzard.recovery import build_resume_plan, build_retry_plan
from whitzard.run_flow import run_recovery_plan
from whitzard.runtime.payloads import TaskPayload, TaskPrompt


class FakeEnvRecord:
    env_id = "env_test"
    state = "ready"


class FakeEnvManager:
    def ensure_ready(self, model_name: str, foreground: bool = True, progress=None):
        return FakeEnvRecord()

    def inspect_model_environment(self, model_name: str):
        return FakeEnvRecord()


class RecoveryTests(unittest.TestCase):
    def test_retry_plan_selects_failed_outputs_only(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        run_id = "run_failed"
        self._write_source_run(
            root=tmpdir,
            run_id=run_id,
            payloads=[
                TaskPayload(
                    task_id="task_000001",
                    model_name="Z-Image",
                    execution_mode="mock",
                    prompts=[
                        TaskPrompt(prompt_id="p001", prompt="a cat", language="en"),
                        TaskPrompt(prompt_id="p002", prompt="a dog", language="en"),
                    ],
                    params={"width": 1024},
                    workdir="/tmp/run_failed/task_000001",
                )
            ],
            samples=[
                {"model_name": "Z-Image", "prompt_id": "p001", "status": "success"},
                {"model_name": "Z-Image", "prompt_id": "p002", "status": "failed"},
            ],
            failures=[{"task_id": "task_000001", "model_name": "Z-Image", "error": "boom"}],
        )

        plan = build_retry_plan(run_id, runs_root=tmpdir)

        self.assertEqual(plan.selected_count, 1)
        self.assertEqual(plan.failed_count, 1)
        self.assertEqual(plan.completed_count, 1)
        self.assertEqual(plan.missing_count, 0)
        self.assertEqual(plan.model_names, ["Z-Image"])
        self.assertEqual(plan.items_by_model["Z-Image"][0].prompt.prompt_id, "p002")

    def test_resume_plan_selects_missing_outputs_only(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        run_id = "run_missing"
        self._write_source_run(
            root=tmpdir,
            run_id=run_id,
            payloads=[
                TaskPayload(
                    task_id="task_000001",
                    model_name="Wan2.2-T2V-A14B-Diffusers",
                    execution_mode="mock",
                    prompts=[
                        TaskPrompt(prompt_id="p101", prompt="storm over sea", language="en"),
                        TaskPrompt(prompt_id="p102", prompt="山谷中的云海", language="zh"),
                    ],
                    params={"width": 1280, "height": 720},
                    workdir="/tmp/run_missing/task_000001",
                )
            ],
            samples=[
                {
                    "model_name": "Wan2.2-T2V-A14B-Diffusers",
                    "prompt_id": "p101",
                    "status": "success",
                }
            ],
            failures=[],
        )

        plan = build_resume_plan(run_id, runs_root=tmpdir)

        self.assertEqual(plan.selected_count, 1)
        self.assertEqual(plan.completed_count, 1)
        self.assertEqual(plan.failed_count, 0)
        self.assertEqual(plan.missing_count, 1)
        self.assertEqual(
            plan.items_by_model["Wan2.2-T2V-A14B-Diffusers"][0].prompt.prompt_id,
            "p102",
        )

    def test_recovery_run_creates_lineage_manifest_and_skips_successful_outputs(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        source_run_id = "run_resume_source"
        self._write_source_run(
            root=tmpdir,
            run_id=source_run_id,
            payloads=[
                TaskPayload(
                    task_id="task_000001",
                    model_name="Z-Image",
                    execution_mode="mock",
                    prompts=[
                        TaskPrompt(prompt_id="p001", prompt="a cat", language="en"),
                        TaskPrompt(prompt_id="p002", prompt="a dog", language="en"),
                    ],
                    params={"width": 1024, "height": 1024},
                    workdir="/tmp/run_resume_source/task_000001",
                )
            ],
            samples=[
                {"model_name": "Z-Image", "prompt_id": "p001", "status": "success"},
            ],
            failures=[],
        )
        plan = build_resume_plan(source_run_id, runs_root=tmpdir)

        def fake_worker_runner(_env_record, task_file: Path, result_file: Path):
            payload = json.loads(task_file.read_text(encoding="utf-8"))
            workdir = Path(payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            batch_items = []
            for prompt in payload["prompts"]:
                artifact_path = workdir / f"{prompt['prompt_id']}.png"
                artifact_path.write_bytes(
                    b"\x89PNG\r\n\x1a\n"
                    + b"\x00\x00\x00\rIHDR"
                    + b"\x00\x00\x00\x01"
                    + b"\x00\x00\x00\x01"
                    + b"\x08\x02\x00\x00\x00"
                )
                batch_items.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "status": "success",
                        "metadata": {"execution_mode": payload["execution_mode"]},
                        "artifacts": [
                            {
                                "type": "image",
                                "path": str(artifact_path),
                                "metadata": {"width": 1, "height": 1, "format": "png"},
                            }
                        ],
                    }
                )
            result_payload = {
                "task_id": payload["task_id"],
                "model_name": payload["model_name"],
                "execution_mode": payload["execution_mode"],
                "plan": {"mode": "in_process"},
                "execution_result": {"exit_code": 0, "logs": "ok", "outputs": {}},
                "model_result": {
                    "status": "success",
                    "batch_items": batch_items,
                    "logs": "ok",
                    "metadata": {},
                },
            }
            result_file.write_text(json.dumps(result_payload), encoding="utf-8")
            return 0, "ok"

        summary = run_recovery_plan(
            recovery_plan=plan,
            out_dir=tmpdir / "recovery_run",
            env_manager=FakeEnvManager(),
            worker_runner=fake_worker_runner,
        )

        manifest = json.loads(
            (Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["parent_run_id"], source_run_id)
        self.assertEqual(manifest["source_run_id"], source_run_id)
        self.assertEqual(manifest["recovery_mode"], "resume")
        self.assertEqual(manifest["recovered_item_count"], 1)

        exported_records = [
            json.loads(line)
            for line in (Path(summary.output_dir) / "exports" / "dataset.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(exported_records), 1)
        self.assertEqual(exported_records[0]["prompt_id"], "p002")

    def _write_source_run(
        self,
        *,
        root: Path,
        run_id: str,
        payloads: list[TaskPayload],
        samples: list[dict],
        failures: list[dict],
    ) -> None:
        run_root = root / run_id
        (run_root / "tasks" / "z-image").mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "created_at": "2026-03-18T12:00:00+00:00",
            "models": sorted({payload.model_name for payload in payloads}),
            "prompt_source": "prompts/example.txt",
            "prompt_count": sum(len(payload.prompts) for payload in payloads),
            "execution_mode": payloads[0].execution_mode if payloads else "mock",
            "task_count": len(payloads),
            "output_dir": str(run_root),
        }
        (run_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        task_root = run_root / "tasks"
        for payload in payloads:
            model_slug = payload.model_name.lower().replace(".", "_").replace("-", "-")
            model_dir = task_root / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / f"{payload.task_id}.json").write_text(
                json.dumps(payload.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        (run_root / "samples.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in samples) + ("\n" if samples else ""),
            encoding="utf-8",
        )
        (run_root / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
