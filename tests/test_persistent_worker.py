import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from aigc.run_flow import _PersistentWorkerSession

ROOT = Path(__file__).resolve().parents[1]


class PersistentWorkerTests(unittest.TestCase):
    def test_persistent_worker_session_ignores_blank_and_non_json_stdout_lines(self) -> None:
        logged_lines: list[str] = []
        session = _PersistentWorkerSession(
            model=SimpleNamespace(name="Noise-Test"),
            env_record=SimpleNamespace(),
            execution_mode="real",
            replica_id=0,
            gpu_assignment=[0],
            log_callback=logged_lines.append,
        )
        session.process = SimpleNamespace(
            stdout=SimpleNamespace(
                readline=self._readline_from(
                    [
                        "\n",
                        "Loading pipeline components... 20%\n",
                        '{"event":"ready","replica_id":0}\n',
                    ]
                )
            )
        )

        event = session._read_event()

        self.assertEqual(event["event"], "ready")
        self.assertEqual(
            logged_lines,
            ["[worker][Noise-Test][replica=0] stdout: Loading pipeline components... 20%"],
        )

    def test_persistent_worker_loads_once_and_runs_multiple_tasks(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = tmpdir / "test_models.json"
        load_counter_path = tmpdir / "load_counter.txt"
        workdir_one = tmpdir / "workdir_one"
        workdir_two = tmpdir / "workdir_two"
        task_path_one = tmpdir / "task_001.json"
        task_path_two = tmpdir / "task_002.json"
        result_path_one = tmpdir / "task_001.result.json"
        result_path_two = tmpdir / "task_002.result.json"

        registry_path.write_text(
            textwrap.dedent(
                f"""
                {{
                  "models": {{
                    "Echo-Test": {{
                      "version": "0.0",
                      "adapter": "EchoTestAdapter",
                      "modality": "text",
                      "task_type": "t2t",
                      "capabilities": {{
                        "supports_batch_prompts": true,
                        "max_batch_size": 8,
                        "preferred_batch_size": 4,
                        "output_types": ["text"]
                      }},
                      "runtime": {{
                        "execution_mode": "in_process",
                        "gpu_required": false,
                        "env_spec": "flux_image"
                      }},
                      "weights": {{
                        "load_counter_file": "{load_counter_path}"
                      }}
                    }}
                  }}
                }}
                """
            ).strip(),
            encoding="utf-8",
        )

        for task_path, result_path, workdir, task_id, prompt_id, prompt_text in (
            (task_path_one, result_path_one, workdir_one, "task_001", "p001", "hello world"),
            (task_path_two, result_path_two, workdir_two, "task_002", "p002", "hello again"),
        ):
            task_path.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "model_name": "Echo-Test",
                        "execution_mode": "real",
                        "worker_strategy": "persistent_worker",
                        "prompts": [
                            {
                                "prompt_id": prompt_id,
                                "prompt": prompt_text,
                                "language": "en",
                            }
                        ],
                        "params": {},
                        "workdir": str(workdir),
                    }
                ),
                encoding="utf-8",
            )

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aigc.runtime.persistent_worker",
                "--model-name",
                "Echo-Test",
                "--execution-mode",
                "real",
                "--registry-file",
                str(registry_path),
            ],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            ready_event = json.loads(process.stdout.readline())
            self.assertEqual(ready_event["event"], "ready")

            process.stdin.write(
                json.dumps(
                    {
                        "command": "run_task",
                        "task_file": str(task_path_one),
                        "result_file": str(result_path_one),
                    }
                )
                + "\n"
            )
            process.stdin.flush()
            event_one = json.loads(process.stdout.readline())
            self.assertEqual(event_one["event"], "task_complete")
            self.assertEqual(event_one["status"], "success")

            process.stdin.write(
                json.dumps(
                    {
                        "command": "run_task",
                        "task_file": str(task_path_two),
                        "result_file": str(result_path_two),
                    }
                )
                + "\n"
            )
            process.stdin.flush()
            event_two = json.loads(process.stdout.readline())
            self.assertEqual(event_two["event"], "task_complete")
            self.assertEqual(event_two["status"], "success")

            process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            process.stdin.flush()
            shutdown_event = json.loads(process.stdout.readline())
            self.assertEqual(shutdown_event["event"], "shutdown")
        finally:
            if process.stdin is not None:
                process.stdin.close()
            stderr_output = process.stderr.read() if process.stderr is not None else ""
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            process.wait(timeout=10)

        self.assertEqual(process.returncode, 0, msg=stderr_output)
        self.assertEqual(load_counter_path.read_text(encoding="utf-8"), "1")

        result_one = json.loads(result_path_one.read_text(encoding="utf-8"))
        result_two = json.loads(result_path_two.read_text(encoding="utf-8"))
        self.assertEqual(result_one["model_result"]["status"], "success")
        self.assertEqual(result_two["model_result"]["status"], "success")
        self.assertTrue((workdir_one / "p001.txt").exists())
        self.assertTrue((workdir_two / "p002.txt").exists())

        self.assertRegex(
            stderr_output,
            r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[worker\]\[Echo-Test\]\[replica=0\] GPUs=\[\] starting persistent worker",
        )
        self.assertIn("[worker][Echo-Test][replica=0] GPUs=[] loading model...", stderr_output)
        self.assertIn("[worker][Echo-Test][replica=0] GPUs=[] model loaded successfully", stderr_output)
        self.assertIn("[worker][Echo-Test][replica=0] GPUs=[] ready", stderr_output)
        self.assertIn("[worker][Echo-Test][replica=0] GPUs=[] running task task_001 batch_size=1", stderr_output)
        self.assertIn("[worker][Echo-Test][replica=0] GPUs=[] running task task_002 batch_size=1", stderr_output)
        self.assertIn("[worker][Echo-Test][replica=0] GPUs=[] shutting down", stderr_output)

    def _readline_from(self, lines: list[str]):
        iterator = iter(lines)

        def readline() -> str:
            return next(iterator, "")

        return readline
