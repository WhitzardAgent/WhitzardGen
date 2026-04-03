from __future__ import annotations

import io
import unittest

from whitzard.ui.runtime_ui import RuntimeTerminalUI
from whitzard.utils.progress import (
    RunHeaderData,
    RunSummaryData,
    TextRunProgress,
    build_run_progress,
    summarize_task_statuses,
)


class ProgressTests(unittest.TestCase):
    def test_text_run_progress_renders_structured_header(self) -> None:
        buffer = io.StringIO()
        progress = TextRunProgress(stream=buffer)

        progress.run_header(
            RunHeaderData(
                run_id="run_001",
                execution_mode="real",
                model_names=["Z-Image", "FLUX.1-dev"],
                prompt_source="prompts/test_image_100.txt",
                prompt_count=100,
                output_dir="/very/long/path/to/runs/run_001",
                running_log_path="/very/long/path/to/runs/run_001/running.log",
                profile_label="image_real",
            )
        )

        output = buffer.getvalue()
        self.assertIn("[RUN] run_001 | mode=real", output)
        self.assertIn("[RUN] models=Z-Image, FLUX.1-dev", output)
        self.assertIn("[RUN] prompts=100 | source=", output)
        self.assertIn("[RUN] profile=image_real", output)

    def test_text_run_progress_stage_and_task_messages(self) -> None:
        buffer = io.StringIO()
        progress = TextRunProgress(stream=buffer)

        progress.stage_start(1, 3, "Loading prompts")
        progress.stage_end(1, 3, "Loading prompts")
        progress.task_start(
            current=1,
            total=2,
            model_name="Z-Image",
            prompts=4,
            execution_mode="mock",
        )
        progress.task_end(
            current=1,
            total=2,
            model_name="Z-Image",
            status="success",
            artifacts=4,
        )

        output = buffer.getvalue()
        self.assertRegex(output, r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[STAGE 1/3\] Loading prompts\.\.\.")
        self.assertIn("[STAGE 1/3] Loading prompts...", output)
        self.assertIn("[STAGE 1/3] Loading prompts - done", output)
        self.assertIn("[TASK] 1/2 model=Z-Image prompts=4 mode=mock", output)
        self.assertIn(
            "[TASK] 1/2 model=Z-Image status=success artifacts=4",
            output,
        )

    def test_text_run_progress_renders_worker_and_replica_events(self) -> None:
        buffer = io.StringIO()
        progress = TextRunProgress(stream=buffer)

        progress.env_message("[run][Wan2.2-T2V-A14B-Diffusers] replica=0 assigned 50 tasks GPUs=[0]")
        progress.env_message("[run][Wan2.2-T2V-A14B-Diffusers] replica=1 assigned 50 tasks GPUs=[1]")
        progress.env_message(
            "[worker][Wan2.2-T2V-A14B-Diffusers][replica=0] GPUs=[0] loading model..."
        )
        progress.env_message(
            "[worker][Wan2.2-T2V-A14B-Diffusers][replica=0] GPUs=[0] ready"
        )

        output = buffer.getvalue()
        self.assertIn("[SCHED] model=Wan2.2-T2V-A14B-Diffusers replica=0 assigned 50 tasks GPUs=[0]", output)
        self.assertIn("[WORKER] model=Wan2.2-T2V-A14B-Diffusers replica=r0 gpus=[0] loading", output)
        self.assertIn("[WORKER] model=Wan2.2-T2V-A14B-Diffusers replica=r0 ready", output)
        self.assertIn("[REPLICA] model=Wan2.2-T2V-A14B-Diffusers", output)

    def test_text_run_progress_summary(self) -> None:
        buffer = io.StringIO()
        progress = TextRunProgress(stream=buffer)
        summary = RunSummaryData(
            status="completed",
            run_id="run_001",
            execution_mode="mock",
            model_names=["Z-Image", "FLUX.1-dev"],
            prompt_count=8,
            task_count=4,
            success_tasks=4,
            failed_tasks=0,
            output_dir="/tmp/runs/run_001",
            dataset_path="/tmp/runs/run_001/exports/dataset.jsonl",
            manifest_path="/tmp/runs/run_001/run_manifest.json",
            wall_time_sec=12.5,
            processed_prompt_outputs=8,
            failed_prompt_outputs=0,
            throughput_per_min=38.4,
        )

        progress.print_summary(summary)
        output = buffer.getvalue()

        self.assertRegex(output, r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[SUMMARY\] completed run_id=run_001")
        self.assertIn("[SUMMARY] completed run_id=run_001", output)
        self.assertIn("[SUMMARY] mode=mock models=Z-Image, FLUX.1-dev", output)
        self.assertIn("[SUMMARY] prompts=8 tasks=4 success=4 failed=0", output)
        self.assertIn("[SUMMARY] prompt_outputs=8 failed_outputs=0 rate=38.4/min", output)
        self.assertIn("[SUMMARY] wall_time_sec=12.50", output)
        self.assertIn("[SUMMARY] out=", output)
        self.assertIn("[SUMMARY] dataset=", output)
        self.assertIn("[SUMMARY] manifest=", output)

    def test_text_run_progress_preserves_throughput_event_lines(self) -> None:
        buffer = io.StringIO()
        progress = TextRunProgress(stream=buffer)

        progress.env_message("[THROUGHPUT] overall prompts=120/800 rate=95.2/min failed=1 eta=00:07:09")
        progress.env_message("[REPLICA] model=Wan2.2-T2V-A14B-Diffusers r0 [0] ready 12/50 rate=2.4/min")

        output = buffer.getvalue()
        self.assertIn("[THROUGHPUT] overall prompts=120/800 rate=95.2/min failed=1 eta=00:07:09", output)
        self.assertIn("[REPLICA] model=Wan2.2-T2V-A14B-Diffusers r0 [0] ready 12/50 rate=2.4/min", output)

    def test_runtime_ui_emits_warning_tag_for_worker_warning_lines(self) -> None:
        ui = RuntimeTerminalUI()

        lines = ui.render_event(
            "[worker][LongCat-Video][replica=2] GPUs=[2] FutureWarning: cache API will change soon"
        )

        self.assertEqual(
            lines,
            ["[WARN] model=LongCat-Video replica=r2 FutureWarning: cache API will change soon"],
        )

    def test_runtime_ui_builds_semantic_rich_renderables_when_color_enabled(self) -> None:
        try:
            from rich.text import Text
        except Exception as exc:  # pragma: no cover - depends on local env
            self.skipTest(f"rich not installed: {exc}")

        ui = RuntimeTerminalUI(enable_color=True)
        rendered = ui.render_console_line(
            "2026-03-20 20:15:00 [THROUGHPUT] overall prompts=120/800 rate=95.2/min failed=1 eta=00:07:09"
        )

        self.assertIsInstance(rendered, Text)
        self.assertEqual(
            rendered.plain,
            "2026-03-20 20:15:00 [THROUGHPUT] overall prompts=120/800 rate=95.2/min failed=1 eta=00:07:09",
        )
        self.assertGreater(len(rendered.spans), 0)

    def test_runtime_ui_builds_semantic_summary_renderable_when_color_enabled(self) -> None:
        try:
            from rich.text import Text
        except Exception as exc:  # pragma: no cover - depends on local env
            self.skipTest(f"rich not installed: {exc}")

        ui = RuntimeTerminalUI(enable_color=True)
        rendered = ui.render_console_line(
            "2026-03-20 20:15:00 [SUMMARY] completed_with_failures run_id=run_001"
        )

        self.assertIsInstance(rendered, Text)
        self.assertIn("completed_with_failures", rendered.plain)
        self.assertGreaterEqual(len(rendered.spans), 2)

    def test_runtime_ui_live_dashboard_tracks_progress_lines(self) -> None:
        try:
            from rich.console import Group
        except Exception as exc:  # pragma: no cover - depends on local env
            self.skipTest(f"rich not installed: {exc}")

        ui = RuntimeTerminalUI(enable_color=True)
        ui.render_header(
            RunHeaderData(
                run_id="run_001",
                execution_mode="real",
                model_names=["Wan2.2-T2V-A14B-Diffusers"],
                prompt_source="prompts/test.txt",
                output_dir="/tmp/runs/run_001",
                running_log_path="/tmp/runs/run_001/running.log",
                prompt_count=20,
            )
        )
        ui.render_event("[worker][Wan2.2-T2V-A14B-Diffusers][replica=0] GPUs=[0] ready")
        ui.render_event(
            "[progress] model=Wan2.2-T2V-A14B-Diffusers replica=0 task=task_001 batch=2 phase=generating step=12/40 true_progress=yes"
        )

        dashboard = ui.render_live_dashboard(
            ["2026-03-20 20:15:00 [THROUGHPUT] model=Wan2.2-T2V-A14B-Diffusers prompts=2/20 rate=6.0/min"]
        )
        self.assertIsInstance(dashboard, Group)

    def test_build_run_progress_uses_null_for_json(self) -> None:
        progress = build_run_progress(output_mode="json")
        # NullRunProgress has no observable behavior; we just ensure stage
        # calls do not crash.
        progress.stage_start(1, 1, "Loading")
        progress.stage_end(1, 1, "Loading")

    def test_summarize_task_statuses(self) -> None:
        success, failed = summarize_task_statuses(
            ["success", "failed", "success", "partial_success"]
        )
        self.assertEqual(success, 2)
        self.assertEqual(failed, 2)
