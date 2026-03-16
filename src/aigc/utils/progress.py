from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import IO, Iterable


def _safe_isatty(stream: IO[str] | None) -> bool:
    try:
        return bool(stream and stream.isatty())
    except Exception:
        return False


@dataclass(slots=True)
class RunSummaryData:
    run_id: str
    execution_mode: str
    model_names: list[str]
    prompt_count: int
    task_count: int
    success_tasks: int
    failed_tasks: int
    output_dir: str
    dataset_path: str
    manifest_path: str


class RunProgress:
    """Abstract progress reporter for long-running runs."""

    def stage_start(self, index: int, total: int, name: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stage_end(self, index: int, total: int, name: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def env_message(self, message: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def task_start(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        prompts: int,
        execution_mode: str,
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def task_end(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        status: str,
        artifacts: int | None = None,
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def print_summary(self, summary: RunSummaryData) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class NullRunProgress(RunProgress):
    """No-op implementation used for JSON or fully quiet output."""

    def stage_start(self, index: int, total: int, name: str) -> None:
        return

    def stage_end(self, index: int, total: int, name: str) -> None:
        return

    def env_message(self, message: str) -> None:
        return

    def task_start(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        prompts: int,
        execution_mode: str,
    ) -> None:
        return

    def task_end(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        status: str,
        artifacts: int | None = None,
    ) -> None:
        return

    def print_summary(self, summary: RunSummaryData) -> None:
        return


class TextRunProgress(RunProgress):
    """Plain-text progress reporter that works in all terminals."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream: IO[str] = stream or sys.stderr

    def _write(self, line: str) -> None:
        try:
            print(line, file=self._stream, flush=True)
        except Exception:
            # Best-effort only; never crash the run on progress failure.
            return

    def stage_start(self, index: int, total: int, name: str) -> None:
        self._write(f"[{index}/{total}] {name}...")

    def stage_end(self, index: int, total: int, name: str) -> None:
        # Keep end lines concise but explicit for non-interactive logs.
        self._write(f"[{index}/{total}] {name} - done")

    def env_message(self, message: str) -> None:
        self._write(message)

    def task_start(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        prompts: int,
        execution_mode: str,
    ) -> None:
        self._write(
            f"Running task {current}/{total} | model={model_name} | prompts={prompts} | mode={execution_mode}"
        )

    def task_end(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        status: str,
        artifacts: int | None = None,
    ) -> None:
        suffix = f" | artifacts={artifacts}" if artifacts is not None else ""
        self._write(
            f"Task {current}/{total} finished | model={model_name} | status={status}{suffix}"
        )

    def print_summary(self, summary: RunSummaryData) -> None:
        models_display = ", ".join(summary.model_names)
        self._write("Run complete")
        self._write(f"run_id: {summary.run_id}")
        self._write(f"mode: {summary.execution_mode}")
        self._write(f"models: {models_display}")
        self._write(f"prompts: {summary.prompt_count}")
        self._write(f"tasks: {summary.task_count}")
        self._write(f"success: {summary.success_tasks}")
        self._write(f"failed: {summary.failed_tasks}")
        self._write(f"output_dir: {summary.output_dir}")
        self._write(f"dataset: {summary.dataset_path}")
        self._write(f"manifest: {summary.manifest_path}")


def build_run_progress(
    *,
    output_mode: str = "text",
    stream: IO[str] | None = None,
    force_plain: bool | None = None,
) -> RunProgress:
    """Factory that builds an appropriate RunProgress implementation.

    - JSON output always uses the NullRunProgress.
    - For text output, prefer a plain-text implementation that works in both
      interactive and non-interactive terminals. If richer TUI is added later,
      this factory is the only place that needs to change.
    """
    if output_mode == "json":
        return NullRunProgress()

    if force_plain is True:
        return TextRunProgress(stream=stream)

    # For now, we always use text progress. This hook allows future rich-based
    # implementations without touching call sites.
    return TextRunProgress(stream=stream)


def summarize_task_statuses(statuses: Iterable[str]) -> tuple[int, int]:
    """Count successful vs failed tasks from a status iterator."""
    success = 0
    failed = 0
    for status in statuses:
        if status == "success":
            success += 1
        else:
            failed += 1
    return success, failed

