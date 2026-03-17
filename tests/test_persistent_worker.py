import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aigc.run_flow import REPO_ROOT, RunFlowError, _PersistentWorkerSession
from aigc.registry import load_registry


class _FakeEnvRecord:
    env_id = "env_test"
    state = "ready"


class PersistentWorkerTests(unittest.TestCase):
    def test_persistent_worker_loads_once_and_runs_multiple_tasks(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = self._write_registry(tmpdir, {})
        registry = load_registry(registry_path)
        model = registry.get_model("Echo-Test")
        load_counter_path = Path(str(model.weights["load_counter_file"]))

        task_one = self._write_task(
            tmpdir=tmpdir,
            task_id="task_001",
            prompt_id="p001",
            prompt_text="hello world",
        )
        task_two = self._write_task(
            tmpdir=tmpdir,
            task_id="task_002",
            prompt_id="p002",
            prompt_text="hello again",
        )
        logged_lines: list[str] = []
        replica_log_path = tmpdir / "replica_0.log"

        with self._patched_worker_command():
            with _PersistentWorkerSession(
                model=model,
                env_record=_FakeEnvRecord(),
                execution_mode="real",
                replica_id=0,
                gpu_assignment=[],
                replica_log_path=replica_log_path,
                log_callback=logged_lines.append,
            ) as session:
                rc_one, logs_one = session.run_task(task_one)
                rc_two, logs_two = session.run_task(task_two)

        self.assertEqual((rc_one, logs_one), (0, ""))
        self.assertEqual((rc_two, logs_two), (0, ""))
        self.assertEqual(load_counter_path.read_text(encoding="utf-8"), "1")
        self.assertTrue((Path(task_one.payload.workdir) / "p001.txt").exists())
        self.assertTrue((Path(task_two.payload.workdir) / "p002.txt").exists())
        self.assertTrue(replica_log_path.exists())
        replica_log = replica_log_path.read_text(encoding="utf-8")
        self.assertIn("starting persistent worker", replica_log)
        self.assertIn("running task task_001 batch_size=1", replica_log)
        self.assertIn("running task task_002 batch_size=1", replica_log)
        self.assertIn("shutting down", replica_log)
        self.assertTrue(any("ready" in line for line in logged_lines))

    def test_startup_crash_surfaces_worker_failure_not_broken_pipe(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = tmpdir / "broken_registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "models": {
                        "Broken-Start": {
                            "version": "0.0",
                            "adapter": "EchoTestAdapter",
                            "modality": "text",
                            "task_type": "t2t",
                            "capabilities": {"output_types": ["text"]},
                            "runtime": {"execution_mode": "in_process", "env_spec": "flux_image"},
                            "weights": {"crash_on_load": True},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        model = SimpleNamespace(name="Broken-Start", registry_source=str(registry_path))

        with self.assertRaises(RunFlowError) as context:
            with self._patched_worker_command():
                with _PersistentWorkerSession(
                    model=model,
                    env_record=_FakeEnvRecord(),
                    execution_mode="real",
                    replica_id=0,
                    gpu_assignment=[0],
                ):
                    pass

        message = str(context.exception)
        self.assertIn("failed during startup", message)
        self.assertIn("EchoTestAdapter crash_on_load", message)
        self.assertNotIn("Broken pipe", message)

    def test_worker_exit_before_task_started_reports_accept_failure(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = self._write_registry(tmpdir, {})
        registry = load_registry(registry_path)
        model = registry.get_model("Echo-Test")
        missing_task_path = tmpdir / "missing_task.json"
        result_path = tmpdir / "missing_task.result.json"
        prepared_task = SimpleNamespace(
            task_file=missing_task_path,
            result_file=result_path,
            payload=SimpleNamespace(task_id="task_missing"),
        )

        session = _PersistentWorkerSession(
            model=model,
            env_record=_FakeEnvRecord(),
            execution_mode="real",
            replica_id=0,
            gpu_assignment=[],
        )
        with self._patched_worker_command():
            with session:
                with self.assertRaises(RunFlowError) as context:
                    session.run_task(prepared_task)  # type: ignore[arg-type]
        self.assertIn("before accepting task task_missing", str(context.exception))
        session.close()

    def test_worker_exit_during_task_reports_last_task_context(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        registry_path = self._write_registry(tmpdir, {"hard_exit_in_execute": True})
        registry = load_registry(registry_path)
        model = registry.get_model("Echo-Test")
        prepared_task = self._write_task(
            tmpdir=tmpdir,
            task_id="task_001",
            prompt_id="p001",
            prompt_text="boom",
        )

        session = _PersistentWorkerSession(
            model=model,
            env_record=_FakeEnvRecord(),
            execution_mode="real",
            replica_id=0,
            gpu_assignment=[],
        )
        with self._patched_worker_command():
            with session:
                with self.assertRaises(RunFlowError) as context:
                    session.run_task(prepared_task)

        message = str(context.exception)
        self.assertIn("while executing task task_001", message)
        self.assertIn("Worker exit code", message)
        self.assertNotIn("Broken pipe", message)

    def _write_registry(self, tmpdir: Path, weight_overrides: dict[str, object]) -> Path:
        registry_path = tmpdir / "test_models.json"
        load_counter_path = tmpdir / "load_counter.txt"
        weights = {"load_counter_file": str(load_counter_path), **weight_overrides}
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
                      "weights": {json.dumps(weights)}
                    }}
                  }}
                }}
                """
            ).strip(),
            encoding="utf-8",
        )
        return registry_path

    def _write_task(
        self,
        *,
        tmpdir: Path,
        task_id: str,
        prompt_id: str,
        prompt_text: str,
    ):
        workdir = tmpdir / task_id
        task_path = tmpdir / f"{task_id}.json"
        result_path = tmpdir / f"{task_id}.result.json"
        payload = {
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
        task_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            task_file=task_path,
            result_file=result_path,
            payload=SimpleNamespace(task_id=task_id, workdir=str(workdir)),
        )

    def _patched_worker_command(self):
        def fake_build_worker_command_and_env(
            *,
            env_record,
            model_name,
            execution_mode,
            module_name,
            extra_args,
            env_overrides=None,
        ):
            env = os.environ.copy()
            pythonpath = str(REPO_ROOT / "src")
            env["PYTHONPATH"] = pythonpath
            if env_overrides:
                env.update(env_overrides)
            return [sys.executable, "-m", module_name, *extra_args], env

        return patch("aigc.run_flow._build_worker_command_and_env", side_effect=fake_build_worker_command_and_env)
