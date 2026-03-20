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

        self.assertTrue(any("replicas=2" in line for line in emitted if "model=Wan2.2-T2V-A14B-Diffusers" in line))
        self.assertTrue(any("[REPLICA] model=Wan2.2-T2V-A14B-Diffusers r0 [0]" in line for line in emitted))

        snapshot = telemetry.snapshot_dict()
        model_metrics = snapshot["models"]["Wan2.2-T2V-A14B-Diffusers"]
        self.assertEqual(model_metrics["avg_model_load_sec"], 5.2)
        self.assertEqual(snapshot["replicas"]["Wan2.2-T2V-A14B-Diffusers"]["r0"]["processed_prompts"], 1)

