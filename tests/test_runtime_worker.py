import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeWorkerTests(unittest.TestCase):
    def test_worker_executes_in_process_adapter(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = tmpdir / "test_models.json"
        task_path = tmpdir / "task.json"
        result_path = tmpdir / "result.json"
        workdir = tmpdir / "workdir"

        registry_path.write_text(
            textwrap.dedent(
                """
                {
                  "models": {
                    "Echo-Test": {
                      "version": "0.0",
                      "adapter": "EchoTestAdapter",
                      "modality": "text",
                      "task_type": "t2t",
                      "capabilities": {
                        "supports_batch_prompts": true,
                        "max_batch_size": 8,
                        "preferred_batch_size": 4,
                        "output_types": ["text"]
                      },
                      "runtime": {
                        "execution_mode": "in_process",
                        "gpu_required": false,
                        "env_spec": "flux_image"
                      },
                      "weights": {}
                    }
                  }
                }
                """
            ).strip(),
            encoding="utf-8",
        )
        task_path.write_text(
            json.dumps(
                {
                    "task_id": "task_001",
                    "model_name": "Echo-Test",
                    "execution_mode": "real",
                    "prompts": [
                        {"prompt_id": "p001", "prompt": "hello world", "language": "en"},
                        {"prompt_id": "p002", "prompt": "你好，世界", "language": "zh"},
                    ],
                    "params": {},
                    "workdir": str(workdir),
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "whitzard.runtime.worker",
                "--task-file",
                str(task_path),
                "--result-file",
                str(result_path),
                "--registry-file",
                str(registry_path),
            ],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["task_id"], "task_001")
        self.assertEqual(payload["model_result"]["status"], "success")
        self.assertTrue((workdir / "p001.txt").exists())
        self.assertEqual((workdir / "p001.txt").read_text(encoding="utf-8"), "hello world")

    def test_worker_failure_writes_traceback_to_result_and_stderr(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = tmpdir / "test_models.json"
        task_path = tmpdir / "task.json"
        result_path = tmpdir / "result.json"

        registry_path.write_text('{"models": {}}', encoding="utf-8")
        task_path.write_text(
            json.dumps(
                {
                    "task_id": "task_fail",
                    "model_name": "Missing-Model",
                    "execution_mode": "real",
                    "prompts": [{"prompt_id": "p001", "prompt": "hello", "language": "en"}],
                    "params": {},
                    "workdir": str(tmpdir / "workdir"),
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "whitzard.runtime.worker",
                "--task-file",
                str(task_path),
                "--result-file",
                str(result_path),
                "--registry-file",
                str(registry_path),
            ],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Traceback", result.stderr)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model_result"]["status"], "failed")
        self.assertIn("Traceback", payload["execution_result"]["logs"])
