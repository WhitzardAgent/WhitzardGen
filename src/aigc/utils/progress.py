from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import IO, Iterable

from aigc.utils.runtime_logging import RunLogger, format_log_line


def _safe_isatty(stream: IO[str] | None) -> bool:
    try:
        return bool(stream and stream.isatty())
    except Exception:
        return False


@dataclass(slots=True)
class RunSummaryData:
    status: str
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
    failures_path: str | None = None
    running_log_path: str | None = None


def format_stage_start_line(index: int, total: int, name: str) -> str:
    return f"[{index}/{total}] {name}..."


def format_stage_end_line(index: int, total: int, name: str) -> str:
    return f"[{index}/{total}] {name} - done"


def format_task_start_line(
    *,
    current: int,
    total: int,
    model_name: str,
    prompts: int,
    execution_mode: str,
) -> str:
    return f"Running task {current}/{total} | model={model_name} | prompts={prompts} | mode={execution_mode}"


def format_task_end_line(
    *,
    current: int,
    total: int,
    model_name: str,
    status: str,
    artifacts: int | None = None,
) -> str:
    suffix = f" | artifacts={artifacts}" if artifacts is not None else ""
    return f"Task {current}/{total} finished | model={model_name} | status={status}{suffix}"


def format_summary_lines(summary: RunSummaryData) -> list[str]:
    models_display = ", ".join(summary.model_names)
    title = "Run complete" if summary.status == "completed" else "Run failed"
    lines = [
        title,
        f"status: {summary.status}",
        f"run_id: {summary.run_id}",
        f"mode: {summary.execution_mode}",
        f"models: {models_display}",
        f"prompts: {summary.prompt_count}",
        f"tasks: {summary.task_count}",
        f"success: {summary.success_tasks}",
        f"failed: {summary.failed_tasks}",
        f"output_dir: {summary.output_dir}",
        f"dataset: {summary.dataset_path}",
        f"manifest: {summary.manifest_path}",
    ]
    if summary.failures_path:
        lines.append(f"failures: {summary.failures_path}")
    if summary.running_log_path:
        lines.append(f"running_log: {summary.running_log_path}")
    return lines


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
            print(format_log_line(line), file=self._stream, flush=True)
        except Exception:
            # Best-effort only; never crash the run on progress failure.
            return

    def stage_start(self, index: int, total: int, name: str) -> None:
        self._write(format_stage_start_line(index, total, name))

    def stage_end(self, index: int, total: int, name: str) -> None:
        # Keep end lines concise but explicit for non-interactive logs.
        self._write(format_stage_end_line(index, total, name))

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
            format_task_start_line(
                current=current,
                total=total,
                model_name=model_name,
                prompts=prompts,
                execution_mode=execution_mode,
            )
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
        self._write(
            format_task_end_line(
                current=current,
                total=total,
                model_name=model_name,
                status=status,
                artifacts=artifacts,
            )
        )

    def print_summary(self, summary: RunSummaryData) -> None:
        for line in format_summary_lines(summary):
            self._write(line)


class LoggedRunProgress(RunProgress):
    """Mirror progress events into a persistent run log while preserving console UX."""

    def __init__(self, *, base: RunProgress, logger: RunLogger) -> None:
        self._base = base
        self._logger = logger

    def stage_start(self, index: int, total: int, name: str) -> None:
        self._logger.log(format_stage_start_line(index, total, name))
        self._base.stage_start(index, total, name)

    def stage_end(self, index: int, total: int, name: str) -> None:
        self._logger.log(format_stage_end_line(index, total, name))
        self._base.stage_end(index, total, name)

    def env_message(self, message: str) -> None:
        self._logger.log(message)
        self._base.env_message(message)

    def task_start(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        prompts: int,
        execution_mode: str,
    ) -> None:
        self._logger.log(
            format_task_start_line(
                current=current,
                total=total,
                model_name=model_name,
                prompts=prompts,
                execution_mode=execution_mode,
            )
        )
        self._base.task_start(
            current=current,
            total=total,
            model_name=model_name,
            prompts=prompts,
            execution_mode=execution_mode,
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
        self._logger.log(
            format_task_end_line(
                current=current,
                total=total,
                model_name=model_name,
                status=status,
                artifacts=artifacts,
            )
        )
        self._base.task_end(
            current=current,
            total=total,
            model_name=model_name,
            status=status,
            artifacts=artifacts,
        )

    def print_summary(self, summary: RunSummaryData) -> None:
        for line in format_summary_lines(summary):
            self._logger.log(line)
        self._base.print_summary(summary)


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
