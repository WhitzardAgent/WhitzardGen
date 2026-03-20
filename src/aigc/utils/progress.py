from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import IO, Iterable

from aigc.ui.runtime_ui import RuntimeTerminalUI
from aigc.utils.runtime_logging import RunLogger, format_log_line

try:  # pragma: no cover - optional dependency
    from rich.console import Console
    from rich.live import Live
except Exception:  # pragma: no cover - graceful fallback without rich
    Console = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]


def _safe_isatty(stream: IO[str] | None) -> bool:
    try:
        return bool(stream and stream.isatty())
    except Exception:
        return False


@dataclass(slots=True)
class RunHeaderData:
    run_id: str
    execution_mode: str
    model_names: list[str]
    prompt_source: str
    output_dir: str
    running_log_path: str
    prompt_count: int | None = None
    profile_label: str | None = None


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
    wall_time_sec: float | None = None
    processed_prompt_outputs: int | None = None
    failed_prompt_outputs: int | None = None
    throughput_per_min: float | None = None
    stop_reason: str | None = None


def format_stage_start_line(index: int, total: int, name: str) -> str:
    return f"[STAGE {index}/{total}] {name}..."


def format_stage_end_line(index: int, total: int, name: str) -> str:
    return f"[STAGE {index}/{total}] {name} - done"


def format_header_lines(header: RunHeaderData) -> list[str]:
    lines = [
        f"[run] Starting run run_id={header.run_id} mode={header.execution_mode} models={','.join(header.model_names)}",
        f"[run] Prompt source: {header.prompt_source}",
        f"[run] Output dir: {header.output_dir}",
        f"[run] Running log: {header.running_log_path}",
    ]
    if header.prompt_count is not None:
        lines.append(f"[run] Prompt count: {header.prompt_count}")
    if header.profile_label:
        lines.append(f"[run] Profile: {header.profile_label}")
    return lines


def format_task_start_line(
    *,
    current: int,
    total: int,
    model_name: str,
    prompts: int,
    execution_mode: str,
) -> str:
    return f"[TASK] {current}/{total} model={model_name} prompts={prompts} mode={execution_mode}"


def format_task_end_line(
    *,
    current: int,
    total: int,
    model_name: str,
    status: str,
    artifacts: int | None = None,
) -> str:
    suffix = f" | artifacts={artifacts}" if artifacts is not None else ""
    return f"[TASK] {current}/{total} model={model_name} status={status}{suffix}"


def format_summary_lines(summary: RunSummaryData) -> list[str]:
    models_display = ", ".join(summary.model_names)
    if summary.status == "completed":
        title = "Run complete"
    elif summary.status == "completed_with_failures":
        title = "Run complete with failures"
    else:
        title = "Run failed"
    lines = [
        f"[summary] {title}",
        f"[summary] status: {summary.status}",
        f"[summary] run_id: {summary.run_id}",
        f"[summary] mode: {summary.execution_mode}",
        f"[summary] models: {models_display}",
        f"[summary] prompts: {summary.prompt_count}",
        f"[summary] tasks: {summary.task_count}",
        f"[summary] success: {summary.success_tasks}",
        f"[summary] failed: {summary.failed_tasks}",
    ]
    if summary.processed_prompt_outputs is not None:
        throughput_suffix = ""
        if summary.throughput_per_min is not None:
            throughput_suffix = f" rate={summary.throughput_per_min:.1f}/min"
        lines.append(
            f"[summary] prompt_outputs: {summary.processed_prompt_outputs} failed_outputs: {summary.failed_prompt_outputs or 0}{throughput_suffix}"
        )
    if summary.wall_time_sec is not None:
        lines.append(f"[summary] wall_time_sec: {summary.wall_time_sec:.2f}")
    lines.extend(
        [
        f"[summary] output_dir: {summary.output_dir}",
        f"[summary] dataset: {summary.dataset_path}",
        f"[summary] manifest: {summary.manifest_path}",
        ]
    )
    if summary.stop_reason:
        lines.append(f"[summary] reason: {summary.stop_reason}")
    if summary.failures_path:
        lines.append(f"[summary] failures: {summary.failures_path}")
    if summary.running_log_path:
        lines.append(f"[summary] running_log: {summary.running_log_path}")
    return lines


class RunProgress:
    """Abstract progress reporter for long-running runs."""

    def run_header(self, header: RunHeaderData) -> None:  # pragma: no cover - interface
        raise NotImplementedError

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

    def close(self) -> None:  # pragma: no cover - interface
        return


class NullRunProgress(RunProgress):
    """No-op implementation used for JSON or fully quiet output."""

    def run_header(self, header: RunHeaderData) -> None:
        return

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

    def close(self) -> None:
        return


class TextRunProgress(RunProgress):
    """Plain-text progress reporter that works in all terminals."""

    def __init__(self, stream: IO[str] | None = None, *, enable_color: bool | None = None) -> None:
        self._stream: IO[str] = stream or sys.stderr
        self._enable_color = _safe_isatty(self._stream) if enable_color is None else enable_color
        self._ui = RuntimeTerminalUI(enable_color=self._enable_color)
        self._console = None
        self._live = None
        self._recent_events: list[str] = []
        if self._enable_color and Console is not None:
            self._console = Console(
                file=self._stream,
                force_terminal=True,
                color_system="truecolor",
                soft_wrap=True,
                highlight=False,
            )
        if self._console is not None and Live is not None:
            self._live = Live(
                self._ui.render_live_dashboard(self._recent_events),
                console=self._console,
                auto_refresh=False,
                transient=False,
                vertical_overflow="visible",
            )
            self._live.start()

    def _write(self, line: str) -> None:
        try:
            rendered = format_log_line(line)
            if self._live is not None:
                if not line.startswith("[REPLICA]"):
                    self._recent_events.append(rendered)
                    self._recent_events = self._recent_events[-8:]
                self._refresh_live()
            elif self._console is not None:
                self._console.print(self._ui.render_console_line(rendered))
            else:
                print(rendered, file=self._stream, flush=True)
        except Exception:
            # Best-effort only; never crash the run on progress failure.
            return

    def _refresh_live(self) -> None:
        if self._live is None:
            return
        self._live.update(self._ui.render_live_dashboard(self._recent_events), refresh=True)

    def run_header(self, header: RunHeaderData) -> None:
        for line in self._ui.render_header(header):
            self._write(line)

    def stage_start(self, index: int, total: int, name: str) -> None:
        self._write(self._ui.render_stage_start(index, total, name))

    def stage_end(self, index: int, total: int, name: str) -> None:
        self._write(self._ui.render_stage_end(index, total, name))

    def env_message(self, message: str) -> None:
        if self._live is None and message.strip().startswith("[REPLICA]"):
            self._write(message)
            return
        lines = self._ui.render_event(message)
        for line in lines:
            self._write(line)
        if not lines:
            self._refresh_live()

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
            self._ui.render_task_start(
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
            self._ui.render_task_end(
                current=current,
                total=total,
                model_name=model_name,
                status=status,
                artifacts=artifacts,
            )
        )

    def print_summary(self, summary: RunSummaryData) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        for line in self._ui.render_summary(summary):
            self._write(line)

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


class LoggedRunProgress(RunProgress):
    """Mirror progress events into a persistent run log while preserving console UX."""

    def __init__(self, *, base: RunProgress, logger: RunLogger) -> None:
        self._base = base
        self._logger = logger

    def run_header(self, header: RunHeaderData) -> None:
        for line in format_header_lines(header):
            self._logger.log(line)
        self._base.run_header(header)

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

    def close(self) -> None:
        self._base.close()


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
