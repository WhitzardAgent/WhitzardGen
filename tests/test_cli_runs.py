import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from aigc.run_flow import run_single_model


class FakeEnvRecord:
    env_id = "env_test"
    state = "ready"


class FakeEnvManager:
    def ensure_environment(self, model_name: str):
        return FakeEnvRecord()

    def inspect_model_environment(self, model_name: str):
        return FakeEnvRecord()


class RunsCliTests(unittest.TestCase):
    def test_run_store_commands_have_manifest_and_dataset_visibility(self) -> None:
        from aigc.cli.main import handle_export_dataset, handle_runs_failures, handle_runs_inspect, handle_runs_list

        tmpdir = Path(tempfile.mkdtemp())
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
