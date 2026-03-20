from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReplicaStatus:
    replica_id: int
    gpu_assignment: str = "[]"
    state: str = "starting"
    completed_tasks: int = 0
    assigned_tasks: int = 0


class RuntimeTerminalUI:
    """Render a cleaner terminal-friendly operational view for long runs."""

    _TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+")
    _WORKER_RE = re.compile(
        r"^\[worker\]\[(?P<model>[^\]]+)\]\[replica=(?P<replica>\d+)\] "
        r"GPUs=(?P<gpus>\[[^\]]*\]) (?P<message>.+)$"
    )
    _RUN_MODEL_RE = re.compile(r"^\[run\]\[(?P<model>[^\]]+)\] (?P<message>.+)$")

    def __init__(self) -> None:
        self._replicas: dict[str, dict[int, ReplicaStatus]] = {}

    def render_header(self, header) -> list[str]:
        lines = [
            f"[RUN] {header.run_id} | mode={header.execution_mode}",
            f"[RUN] models={', '.join(header.model_names)}",
            f"[RUN] prompts={header.prompt_count if header.prompt_count is not None else '-'} | source={self._shorten_path(header.prompt_source)}",
            f"[RUN] out={self._shorten_path(header.output_dir)}",
            f"[RUN] log={self._shorten_path(header.running_log_path)}",
        ]
        if getattr(header, "profile_label", None):
            lines.append(f"[RUN] profile={header.profile_label}")
        return lines

    def render_stage_start(self, index: int, total: int, name: str) -> str:
        return f"[STAGE {index}/{total}] {name}..."

    def render_stage_end(self, index: int, total: int, name: str) -> str:
        return f"[STAGE {index}/{total}] {name} - done"

    def render_task_start(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        prompts: int,
        execution_mode: str,
    ) -> str:
        return f"[TASK] {current}/{total} model={model_name} prompts={prompts} mode={execution_mode}"

    def render_task_end(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        status: str,
        artifacts: int | None = None,
    ) -> str:
        suffix = f" artifacts={artifacts}" if artifacts is not None else ""
        return f"[TASK] {current}/{total} model={model_name} status={status}{suffix}"

    def render_summary(self, summary) -> list[str]:
        if summary.status == "completed":
            title = "[SUMMARY] completed"
        elif summary.status == "completed_with_failures":
            title = "[SUMMARY] completed_with_failures"
        else:
            title = "[SUMMARY] failed"
        lines = [
            f"{title} run_id={summary.run_id}",
            f"[SUMMARY] mode={summary.execution_mode} models={', '.join(summary.model_names)}",
            f"[SUMMARY] prompts={summary.prompt_count} tasks={summary.task_count} success={summary.success_tasks} failed={summary.failed_tasks}",
        ]
        if getattr(summary, "processed_prompt_outputs", None) is not None:
            throughput_suffix = ""
            if getattr(summary, "throughput_per_min", None) is not None:
                throughput_suffix = f" rate={summary.throughput_per_min:.1f}/min"
            lines.append(
                f"[SUMMARY] prompt_outputs={summary.processed_prompt_outputs} failed_outputs={getattr(summary, 'failed_prompt_outputs', 0) or 0}{throughput_suffix}"
            )
        if getattr(summary, "wall_time_sec", None) is not None:
            lines.append(f"[SUMMARY] wall_time_sec={summary.wall_time_sec:.2f}")
        lines.extend(
            [
                f"[SUMMARY] out={self._shorten_path(summary.output_dir)}",
                f"[SUMMARY] dataset={self._shorten_path(summary.dataset_path)}",
                f"[SUMMARY] manifest={self._shorten_path(summary.manifest_path)}",
            ]
        )
        if summary.failures_path:
            lines.append(f"[SUMMARY] failures={self._shorten_path(summary.failures_path)}")
        if summary.running_log_path:
            lines.append(f"[SUMMARY] log={self._shorten_path(summary.running_log_path)}")
        return lines

    def render_event(self, message: str) -> list[str]:
        normalized = self._TIMESTAMP_RE.sub("", message.strip(), count=1)
        if not normalized:
            return []
        if self._is_noise(normalized):
            return []
        if normalized.startswith("[run] ERROR:"):
            return [f"[ERROR] {normalized.removeprefix('[run] ERROR:').strip()}"]
        if normalized.startswith("Ensuring environment for model: "):
            model_name = normalized.removeprefix("Ensuring environment for model: ").strip()
            return [f"[ENV] model={model_name} ensuring environment"]
        worker_match = self._WORKER_RE.match(normalized)
        if worker_match:
            return self._render_worker_event(
                model_name=worker_match.group("model"),
                replica_id=int(worker_match.group("replica")),
                gpu_assignment=worker_match.group("gpus"),
                message=worker_match.group("message"),
            )
        run_match = self._RUN_MODEL_RE.match(normalized)
        if run_match:
            return self._render_run_model_event(
                model_name=run_match.group("model"),
                message=run_match.group("message"),
            )
        if normalized.startswith("[run] "):
            return [f"[RUN] {normalized.removeprefix('[run] ').strip()}"]
        return [normalized]

    def _render_worker_event(
        self,
        *,
        model_name: str,
        replica_id: int,
        gpu_assignment: str,
        message: str,
    ) -> list[str]:
        state = self._replicas.setdefault(model_name, {}).setdefault(
            replica_id,
            ReplicaStatus(replica_id=replica_id),
        )
        state.gpu_assignment = gpu_assignment
        lines: list[str] = []

        if message == "starting persistent worker":
            state.state = "starting"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} gpus={gpu_assignment} starting")
        elif message == "loading model...":
            state.state = "loading"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} gpus={gpu_assignment} loading")
        elif message.startswith("model loaded successfully in "):
            state.state = "loaded"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} {message}")
        elif message == "ready":
            state.state = "ready"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} ready")
        elif message.startswith("running task "):
            state.state = "running"
            lines.append(f"[TASK] model={model_name} replica=r{replica_id} {message}")
        elif message.startswith("finished task "):
            state.state = "ready"
            state.completed_tasks += 1
            lines.append(f"[TASK] model={model_name} replica=r{replica_id} {message}")
        elif message == "shutting down":
            state.state = "stopped"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} shutting down")
        else:
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} {message}")

        snapshot = self._render_replica_snapshot(model_name)
        if snapshot:
            lines.append(snapshot)
        return lines

    def _render_run_model_event(self, *, model_name: str, message: str) -> list[str]:
        lines: list[str] = []
        if message.startswith("available_gpus="):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("gpus_per_replica="):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("starting ") and " replicas" in message:
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("replica=") and "assigned" in message:
            replica_id = int(message.split("replica=", 1)[1].split()[0])
            assigned_tasks = int(message.split("assigned ", 1)[1].split()[0])
            gpu_assignment = message.split("GPUs=", 1)[1].strip()
            state = self._replicas.setdefault(model_name, {}).setdefault(
                replica_id,
                ReplicaStatus(replica_id=replica_id),
            )
            state.assigned_tasks = assigned_tasks
            state.gpu_assignment = gpu_assignment
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("warming replica "):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif "ready (" in message and message.startswith("replica="):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("all replicas ready"):
            lines.append(f"[SCHED] model={model_name} {message}")
        else:
            lines.append(f"[RUN] model={model_name} {message}")

        snapshot = self._render_replica_snapshot(model_name)
        if snapshot:
            lines.append(snapshot)
        return lines

    def _render_replica_snapshot(self, model_name: str) -> str | None:
        states = self._replicas.get(model_name, {})
        if not states:
            return None
        if len(states) < 2 and not any(state.assigned_tasks for state in states.values()):
            return None
        parts = []
        for replica_id in sorted(states):
            state = states[replica_id]
            total = state.assigned_tasks if state.assigned_tasks else "?"
            parts.append(
                f"r{replica_id} {state.gpu_assignment} {state.state} {state.completed_tasks}/{total}"
            )
        return f"[REPLICA] model={model_name} " + " | ".join(parts)

    def _is_noise(self, message: str) -> bool:
        noisy_tokens = (
            "Loading pipeline components...",
            "Loading checkpoint shards:",
            "Loading weights:",
            "Downloading shards:",
            "%|",
        )
        return any(token in message for token in noisy_tokens)

    def _shorten_path(self, value: str, *, max_len: int = 88) -> str:
        if len(value) <= max_len:
            return value
        path = Path(value)
        parts = path.parts
        if len(parts) >= 3:
            shortened = Path(parts[0]) / "..." / Path(*parts[-3:])
            shortened_str = str(shortened)
            if len(shortened_str) <= max_len:
                return shortened_str
        keep = max_len // 2 - 2
        return f"{value[:keep]}...{value[-keep:]}"
