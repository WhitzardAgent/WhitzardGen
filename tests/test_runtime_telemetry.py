from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aigc.runtime_telemetry import RunTelemetry


class RuntimeTelemetryTests(unittest.TestCase):
    def test_telemetry_computes_throughput_and_eta(self) -> None:
        emitted: list[str] = []
        clock = [0.0]

        def now() -> float:
            return clock[0]

        tmpdir = Path(tempfile.mkdtemp())
        telemetry = RunTelemetry(
            run_id="run_telemetry",
            execution_mode="real",
            emit_callback=emitted.append,
            status_path=tmpdir / "runtime_status.json",
            emit_prompt_interval=2,
            emit_sec_interval=999.0,
            time_source=now,
        )
        prepared_task = SimpleNamespace(payload=SimpleNamespace(prompts=[object(), object()]))
        telemetry.set_plan(prepared_tasks_by_model={"Z-Image": [prepared_task, prepared_task]})

        telemetry.record_task_start(task_id="task_001", model_name="Z-Image", replica_id=None)
        clock[0] = 30.0
        telemetry.record_task_outcome(
            task_id="task_001",
            model_name="Z-Image",
            replica_id=None,
            successful_prompts=2,
            failed_prompts=0,
            task_failed=False,
        )

        self.assertGreaterEqual(len(emitted), 2)
        self.assertIn("[THROUGHPUT] overall prompts=2/4 rate=4.0/min", emitted[0])
        self.assertIn("eta=00:00:30", emitted[0])
        self.assertIn("[THROUGHPUT] model=Z-Image prompts=2/4 rate=4.0/min", emitted[1])

        snapshot = json.loads((tmpdir / "runtime_status.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["processed_prompts"], 2)
        self.assertEqual(snapshot["eta_sec"], 30)
        self.assertEqual(snapshot["models"]["Z-Image"]["processed_prompts"], 2)

    def test_telemetry_tracks_replica_progress_and_model_load_events(self) -> None:
        emitted: list[str] = []
        clock = [0.0]

        def now() -> float:
            return clock[0]

        telemetry = RunTelemetry(
            run_id="run_replicas",
            execution_mode="real",
            emit_callback=emitted.append,
            emit_prompt_interval=1,
            emit_sec_interval=999.0,
            time_source=now,
        )
        prepared_task = SimpleNamespace(payload=SimpleNamespace(prompts=[object()]))
        telemetry.set_plan(prepared_tasks_by_model={"Wan2.2-T2V-A14B-Diffusers": [prepared_task, prepared_task]})
        telemetry.register_replica_assignments(
            model_name="Wan2.2-T2V-A14B-Diffusers",
            replica_plans=[
                SimpleNamespace(replica_id=0, gpu_assignment=[0], tasks=[prepared_task]),
                SimpleNamespace(replica_id=1, gpu_assignment=[1], tasks=[prepared_task]),
            ],
        )
        telemetry.record_runtime_event(
            "[worker][Wan2.2-T2V-A14B-Diffusers][replica=0] GPUs=[0] model loaded successfully in 5.20s"
        )
        telemetry.record_runtime_event(
            "[worker][Wan2.2-T2V-A14B-Diffusers][replica=1] GPUs=[1] ready"
        )
        clock[0] = 60.0
        telemetry.record_task_outcome(
            task_id="task_001",
            model_name="Wan2.2-T2V-A14B-Diffusers",
            replica_id=0,
            successful_prompts=1,
            failed_prompts=0,
            task_failed=False,
        )

        self.assertTrue(
            any(
                "replicas_active=2/2" in line
                for line in emitted
                if "model=Wan2.2-T2V-A14B-Diffusers" in line
            )
        )
        self.assertTrue(any("[REPLICA] model=Wan2.2-T2V-A14B-Diffusers r0 [0]" in line for line in emitted))

        snapshot = telemetry.snapshot_dict()
        model_metrics = snapshot["models"]["Wan2.2-T2V-A14B-Diffusers"]
        self.assertEqual(model_metrics["avg_model_load_sec"], 5.2)
        self.assertEqual(model_metrics["active_replicas"], 2)
        self.assertEqual(snapshot["replicas"]["Wan2.2-T2V-A14B-Diffusers"]["r0"]["processed_prompts"], 1)

    def test_telemetry_tracks_unavailable_secondary_replicas(self) -> None:
        telemetry = RunTelemetry(
            run_id="run_replicas",
            execution_mode="real",
            emit_callback=lambda _line: None,
            emit_prompt_interval=1,
            emit_sec_interval=999.0,
        )
        prepared_task = SimpleNamespace(payload=SimpleNamespace(prompts=[object()]))
        telemetry.set_plan(prepared_tasks_by_model={"CogVideoX-5B": [prepared_task]})
        telemetry.register_replica_assignments(
            model_name="CogVideoX-5B",
            replica_plans=[
                SimpleNamespace(replica_id=0, gpu_assignment=[0], tasks=[prepared_task]),
                SimpleNamespace(replica_id=1, gpu_assignment=[1], tasks=[]),
            ],
        )
        telemetry.record_runtime_event(
            "[worker][CogVideoX-5B][replica=0] GPUs=[0] ready"
        )
        telemetry.record_replica_startup_failure(
            model_name="CogVideoX-5B",
            replica_id=1,
            gpu_assignment=[1],
            unavailable=False,
        )
        telemetry.record_replica_startup_failure(
            model_name="CogVideoX-5B",
            replica_id=1,
            gpu_assignment=[1],
            unavailable=True,
        )

        snapshot = telemetry.snapshot_dict()
        self.assertEqual(snapshot["models"]["CogVideoX-5B"]["replica_startup_failures"], 2)
        self.assertEqual(snapshot["models"]["CogVideoX-5B"]["active_replicas"], 1)
        self.assertTrue(snapshot["replicas"]["CogVideoX-5B"]["r1"]["unavailable"])

    def test_telemetry_tracks_replica_task_progress_state(self) -> None:
        telemetry = RunTelemetry(
            run_id="run_progress",
            execution_mode="real",
            emit_callback=lambda _line: None,
            emit_prompt_interval=1,
            emit_sec_interval=999.0,
        )
        prepared_task = SimpleNamespace(payload=SimpleNamespace(prompts=[object(), object()]))
        telemetry.set_plan(prepared_tasks_by_model={"Z-Image": [prepared_task]})
        telemetry.register_replica_assignments(
            model_name="Z-Image",
            replica_plans=[SimpleNamespace(replica_id=0, gpu_assignment=[0], tasks=[prepared_task])],
        )

        telemetry.record_runtime_event(
            "[progress] model=Z-Image replica=0 task=task_001 batch=2 phase=generating step=12/40 true_progress=yes"
        )
        snapshot = telemetry.snapshot_dict()
        replica = snapshot["replicas"]["Z-Image"]["r0"]

        self.assertEqual(replica["current_task_id"], "task_001")
        self.assertEqual(replica["batch_size"], 2)
        self.assertEqual(replica["current_phase"], "generating")
        self.assertEqual(replica["current_step"], 12)
        self.assertEqual(replica["total_steps"], 40)
        self.assertTrue(replica["supports_true_progress"])
