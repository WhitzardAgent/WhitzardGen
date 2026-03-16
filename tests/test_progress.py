from __future__ import annotations

import io
import unittest

from aigc.utils.progress import (
    RunSummaryData,
    TextRunProgress,
    build_run_progress,
    summarize_task_statuses,
)


class ProgressTests(unittest.TestCase):
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
        self.assertIn("[1/3] Loading prompts...", output)
        self.assertIn("[1/3] Loading prompts - done", output)
        self.assertIn("Running task 1/2 | model=Z-Image | prompts=4 | mode=mock", output)
        self.assertIn(
            "Task 1/2 finished | model=Z-Image | status=success | artifacts=4",
            output,
        )

    def test_text_run_progress_summary(self) -> None:
        buffer = io.StringIO()
        progress = TextRunProgress(stream=buffer)
        summary = RunSummaryData(
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
        )

        progress.print_summary(summary)
        output = buffer.getvalue()

        self.assertIn("Run complete", output)
        self.assertIn("run_id: run_001", output)
        self.assertIn("mode: mock", output)
        self.assertIn("models: Z-Image, FLUX.1-dev", output)
        self.assertIn("prompts: 8", output)
        self.assertIn("tasks: 4", output)
        self.assertIn("success: 4", output)
        self.assertIn("failed: 0", output)
        self.assertIn("output_dir:", output)
        self.assertIn("dataset:", output)
        self.assertIn("manifest:", output)

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

