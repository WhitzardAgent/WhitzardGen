from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+")
_WORKER_RE = re.compile(
    r"^\[worker\]\[(?P<model>[^\]]+)\]\[replica=(?P<replica>\d+)\] "
    r"GPUs=(?P<gpus>\[[^\]]*\]) (?P<message>.+)$"
)
_PROGRESS_RE = re.compile(
    r"^\[progress\]\s+model=(?P<model>\S+)\s+replica=(?P<replica>\d+)\s+"
    r"task=(?P<task>\S+)(?:\s+batch_id=(?P<batch_id>\S+))?(?:\s+batch=(?P<batch_size>\d+))?\s+"
    r"phase=(?P<phase>\S+)(?:\s+step=(?P<current>\d+)/(?P<total>\d+))?(?:\s+true_progress=(?P<true_progress>\S+))?"
)


@dataclass(slots=True)
class ReplicaTelemetry:
    replica_id: int
    gpu_assignment: list[int] = field(default_factory=list)
    assigned_tasks: int = 0
    assigned_prompts: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    successful_prompts: int = 0
    failed_prompts: int = 0
    state: str = "pending"
    load_duration_sec: float | None = None
    first_active_at: float | None = None
    startup_failures: int = 0
    unavailable: bool = False
    current_task_id: str | None = None
    current_batch_id: str | None = None
    batch_size: int | None = None
    current_phase: str | None = None
    current_step: int | None = None
    total_steps: int | None = None
    last_progress_at: float | None = None
    current_task_started_at: float | None = None
    supports_true_progress: bool = False

    @property
    def processed_prompts(self) -> int:
        return self.successful_prompts + self.failed_prompts


@dataclass(slots=True)
class ModelTelemetry:
    model_name: str
    total_prompts: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    successful_prompts: int = 0
    failed_prompts: int = 0
    first_active_at: float | None = None
    load_durations_sec: list[float] = field(default_factory=list)
    replicas: dict[int, ReplicaTelemetry] = field(default_factory=dict)

    @property
    def processed_prompts(self) -> int:
        return self.successful_prompts + self.failed_prompts


class RunTelemetry:
    def __init__(
        self,
        *,
        run_id: str,
        execution_mode: str,
        emit_callback: Callable[[str], None],
        status_path: str | Path | None = None,
        emit_prompt_interval: int = 10,
        emit_sec_interval: float = 15.0,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self.run_id = run_id
        self.execution_mode = execution_mode
        self._emit_callback = emit_callback
        self._status_path = Path(status_path) if status_path is not None else None
        self._emit_prompt_interval = max(emit_prompt_interval, 1)
        self._emit_sec_interval = max(emit_sec_interval, 1.0)
        self._time = time_source or time.monotonic
        self._started_at = self._time()
        self._last_emit_at = self._started_at
        self._next_emit_processed = self._emit_prompt_interval
        self._total_prompts = 0
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._successful_prompts = 0
        self._failed_prompts = 0
        self._models: dict[str, ModelTelemetry] = {}
        self._conditioning_summary: dict[str, object] | None = None
        self._prompt_rewrite_summary: dict[str, object] | None = None
        self._task_started_at: dict[str, float] = {}
        self._write_status_snapshot(status="starting")

    def set_plan(self, *, prepared_tasks_by_model: dict[str, list[object]], append: bool = False) -> None:
        if not append:
            self._total_prompts = 0
            self._total_tasks = 0
            for model_metrics in self._models.values():
                model_metrics.total_tasks = 0
                model_metrics.total_prompts = 0
        for model_name, prepared_tasks in prepared_tasks_by_model.items():
            model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
            task_count = len(prepared_tasks)
            prompt_count = sum(
                len(getattr(prepared_task.payload, "prompts", [])) for prepared_task in prepared_tasks
            )
            if append:
                model_metrics.total_tasks += task_count
                model_metrics.total_prompts += prompt_count
            else:
                model_metrics.total_tasks = task_count
                model_metrics.total_prompts = prompt_count
            self._total_tasks += task_count
            self._total_prompts += prompt_count
        self._write_status_snapshot(status="planned")

    def set_conditioning_summary(self, summary: dict[str, object] | None) -> None:
        self._conditioning_summary = dict(summary) if summary else None
        self._write_status_snapshot(status="running")

    def set_prompt_rewrite_summary(self, summary: dict[str, object] | None) -> None:
        self._prompt_rewrite_summary = dict(summary) if summary else None
        self._write_status_snapshot(status="running")

    def register_replica_assignments(
        self,
        *,
        model_name: str,
        replica_plans: list[object],
    ) -> None:
        model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
        for replica_plan in replica_plans:
            replica = model_metrics.replicas.setdefault(
                int(replica_plan.replica_id),
                ReplicaTelemetry(replica_id=int(replica_plan.replica_id)),
            )
            replica.gpu_assignment = list(replica_plan.gpu_assignment)
            replica.assigned_tasks = len(replica_plan.tasks)
            replica.assigned_prompts = sum(
                len(prepared_task.payload.prompts) for prepared_task in replica_plan.tasks
            )
        self._write_status_snapshot(status="planned")

    def record_runtime_event(self, message: str) -> None:
        normalized = _TIMESTAMP_RE.sub("", message.strip(), count=1)
        if not normalized:
            return
        match = _WORKER_RE.match(normalized)
        if match is None:
            progress_match = _PROGRESS_RE.match(normalized)
            if progress_match is not None:
                self.record_progress_event(
                    model_name=progress_match.group("model"),
                    replica_id=int(progress_match.group("replica")),
                    task_id=progress_match.group("task"),
                    batch_id=progress_match.group("batch_id"),
                    batch_size=int(progress_match.group("batch_size")) if progress_match.group("batch_size") else None,
                    phase=progress_match.group("phase"),
                    current_step=int(progress_match.group("current")) if progress_match.group("current") else None,
                    total_steps=int(progress_match.group("total")) if progress_match.group("total") else None,
                    supports_true_progress=(progress_match.group("true_progress") == "yes"),
                )
            return

        model_name = match.group("model")
        replica_id = int(match.group("replica"))
        gpu_assignment = _parse_gpu_assignment(match.group("gpus"))
        worker_message = match.group("message")
        model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
        replica = model_metrics.replicas.setdefault(
            replica_id,
            ReplicaTelemetry(replica_id=replica_id),
        )
        replica.gpu_assignment = gpu_assignment
        now = self._time()
        if model_metrics.first_active_at is None:
            model_metrics.first_active_at = now
        if replica.first_active_at is None:
            replica.first_active_at = now

        if worker_message == "starting persistent worker":
            replica.state = "starting"
        elif worker_message == "loading model...":
            replica.state = "loading"
        elif worker_message.startswith("model loaded successfully in "):
            replica.state = "loaded"
            load_duration = _extract_load_duration(worker_message)
            if load_duration is not None:
                replica.load_duration_sec = load_duration
                model_metrics.load_durations_sec.append(load_duration)
        elif worker_message == "ready":
            replica.state = "ready"
        elif worker_message.startswith("running task "):
            replica.state = "running"
        elif worker_message.startswith("finished task "):
            replica.state = "ready"
        elif worker_message == "shutting down":
            replica.state = "stopped"
        self._write_status_snapshot(status="running")

    def record_progress_event(
        self,
        *,
        model_name: str,
        replica_id: int,
        task_id: str,
        batch_id: str | None,
        batch_size: int | None,
        phase: str,
        current_step: int | None,
        total_steps: int | None,
        supports_true_progress: bool,
    ) -> None:
        now = self._time()
        model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
        replica = model_metrics.replicas.setdefault(replica_id, ReplicaTelemetry(replica_id=replica_id))
        if model_metrics.first_active_at is None:
            model_metrics.first_active_at = now
        if replica.first_active_at is None:
            replica.first_active_at = now
        replica.current_task_id = task_id
        replica.current_batch_id = batch_id
        replica.batch_size = batch_size
        replica.current_phase = phase
        replica.current_step = current_step
        replica.total_steps = total_steps
        replica.last_progress_at = now
        replica.supports_true_progress = supports_true_progress
        if replica.current_task_started_at is None:
            replica.current_task_started_at = now
        if phase in {"preparing_batch", "generating", "exporting"}:
            replica.state = "running"
        elif phase == "completed":
            replica.state = "ready"
        elif phase == "failed":
            replica.state = "failed"
        self._write_status_snapshot(status="running")

    def record_task_start(
        self,
        *,
        task_id: str,
        model_name: str,
        replica_id: int | None,
    ) -> None:
        now = self._time()
        self._task_started_at[task_id] = now
        model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
        if model_metrics.first_active_at is None:
            model_metrics.first_active_at = now
        if replica_id is not None:
            replica = model_metrics.replicas.setdefault(replica_id, ReplicaTelemetry(replica_id=replica_id))
            if replica.first_active_at is None:
                replica.first_active_at = now
            replica.state = "running"
            replica.current_task_id = task_id
            replica.current_task_started_at = now
        self._write_status_snapshot(status="running")

    def record_replica_startup_failure(
        self,
        *,
        model_name: str,
        replica_id: int,
        gpu_assignment: list[int],
        unavailable: bool,
    ) -> None:
        model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
        replica = model_metrics.replicas.setdefault(replica_id, ReplicaTelemetry(replica_id=replica_id))
        replica.gpu_assignment = list(gpu_assignment)
        replica.startup_failures += 1
        replica.unavailable = unavailable
        replica.state = "unavailable" if unavailable else "failed"
        if model_metrics.first_active_at is None:
            model_metrics.first_active_at = self._time()
        self._write_status_snapshot(status="running")

    def record_task_outcome(
        self,
        *,
        task_id: str,
        model_name: str,
        replica_id: int | None,
        successful_prompts: int,
        failed_prompts: int,
        task_failed: bool,
    ) -> None:
        now = self._time()
        self._task_started_at.pop(task_id, None)
        processed_prompts = successful_prompts + failed_prompts
        self._completed_tasks += 1
        if task_failed:
            self._failed_tasks += 1
        model_metrics = self._models.setdefault(model_name, ModelTelemetry(model_name=model_name))
        model_metrics.completed_tasks += 1
        if task_failed:
            model_metrics.failed_tasks += 1
        model_metrics.successful_prompts += successful_prompts
        model_metrics.failed_prompts += failed_prompts
        if model_metrics.first_active_at is None:
            model_metrics.first_active_at = now

        self._successful_prompts += successful_prompts
        self._failed_prompts += failed_prompts

        if replica_id is not None:
            replica = model_metrics.replicas.setdefault(replica_id, ReplicaTelemetry(replica_id=replica_id))
            replica.completed_tasks += 1
            if task_failed:
                replica.failed_tasks += 1
            replica.successful_prompts += successful_prompts
            replica.failed_prompts += failed_prompts
            replica.state = "ready" if not task_failed else "failed"
            replica.current_phase = "completed" if not task_failed else "failed"
            replica.current_step = replica.total_steps
            replica.current_task_id = None
            replica.current_batch_id = None
            replica.current_task_started_at = None
            if replica.first_active_at is None:
                replica.first_active_at = now

        self._maybe_emit(force=task_failed or processed_prompts >= self._emit_prompt_interval)

    def finalize(self, *, status: str) -> dict[str, object]:
        self._emit(force=True)
        snapshot = self.snapshot_dict(status=status)
        self._write_status_snapshot(status=status)
        return snapshot

    def snapshot_dict(self, *, status: str = "running") -> dict[str, object]:
        elapsed_sec = max(self._time() - self._started_at, 0.0)
        processed_prompts = self.processed_prompts
        overall_rate = _rate_per_min(processed_prompts, elapsed_sec)
        eta_sec = _estimate_eta_sec(
            processed=processed_prompts,
            total=self._total_prompts,
            rate_per_min=overall_rate,
        )
        models_payload: dict[str, object] = {}
        replicas_payload: dict[str, object] = {}
        for model_name, model_metrics in sorted(self._models.items()):
            model_elapsed = _elapsed_since(model_metrics.first_active_at, self._started_at, self._time)
            model_rate = _rate_per_min(model_metrics.processed_prompts, model_elapsed)
            started_replicas = sum(
                1
                for replica in model_metrics.replicas.values()
                if replica.state not in {"pending"}
            )
            active_replicas = sum(
                1
                for replica in model_metrics.replicas.values()
                if replica.state not in {"pending", "failed", "unavailable"}
            )
            models_payload[model_name] = {
                "total_prompts": model_metrics.total_prompts,
                "processed_prompts": model_metrics.processed_prompts,
                "successful_prompts": model_metrics.successful_prompts,
                "failed_prompts": model_metrics.failed_prompts,
                "completed_tasks": model_metrics.completed_tasks,
                "failed_tasks": model_metrics.failed_tasks,
                "total_tasks": model_metrics.total_tasks,
                "rate_per_min": round(model_rate, 3),
                "eta_sec": _estimate_eta_sec(
                    processed=model_metrics.processed_prompts,
                    total=model_metrics.total_prompts,
                    rate_per_min=model_rate,
                ),
                "requested_replicas": len(model_metrics.replicas),
                "started_replicas": started_replicas,
                "active_replicas": active_replicas,
                "replica_startup_failures": sum(replica.startup_failures for replica in model_metrics.replicas.values()),
                "avg_model_load_sec": round(sum(model_metrics.load_durations_sec) / len(model_metrics.load_durations_sec), 3)
                if model_metrics.load_durations_sec
                else None,
            }
            replicas_payload[model_name] = {
                f"r{replica_id}": {
                    "replica_id": replica.replica_id,
                    "gpu_assignment": list(replica.gpu_assignment),
                    "state": replica.state,
                    "assigned_tasks": replica.assigned_tasks,
                    "assigned_prompts": replica.assigned_prompts,
                    "completed_tasks": replica.completed_tasks,
                    "failed_tasks": replica.failed_tasks,
                    "processed_prompts": replica.processed_prompts,
                    "successful_prompts": replica.successful_prompts,
                    "failed_prompts": replica.failed_prompts,
                    "rate_per_min": round(
                        _rate_per_min(
                            replica.processed_prompts,
                            _elapsed_since(replica.first_active_at, self._started_at, self._time),
                        ),
                        3,
                    ),
                    "load_duration_sec": replica.load_duration_sec,
                    "startup_failures": replica.startup_failures,
                    "unavailable": replica.unavailable,
                    "current_task_id": replica.current_task_id,
                    "current_batch_id": replica.current_batch_id,
                    "batch_size": replica.batch_size,
                    "current_phase": replica.current_phase,
                    "current_step": replica.current_step,
                    "total_steps": replica.total_steps,
                    "last_progress_at": round(replica.last_progress_at, 3) if replica.last_progress_at is not None else None,
                    "current_task_started_at": round(replica.current_task_started_at, 3) if replica.current_task_started_at is not None else None,
                    "supports_true_progress": replica.supports_true_progress,
                }
                for replica_id, replica in sorted(model_metrics.replicas.items())
            }
        return {
            "run_id": self.run_id,
            "execution_mode": self.execution_mode,
            "status": status,
            "elapsed_sec": round(elapsed_sec, 3),
            "total_prompts": self._total_prompts,
            "processed_prompts": processed_prompts,
            "successful_prompts": self._successful_prompts,
            "failed_prompts": self._failed_prompts,
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "rate_per_min": round(overall_rate, 3),
            "eta_sec": eta_sec,
            "models": models_payload,
            "replicas": replicas_payload,
            "conditioning": dict(self._conditioning_summary) if self._conditioning_summary else None,
            "prompt_rewrite": dict(self._prompt_rewrite_summary) if self._prompt_rewrite_summary else None,
        }

    @property
    def processed_prompts(self) -> int:
        return self._successful_prompts + self._failed_prompts

    def _maybe_emit(self, *, force: bool = False) -> None:
        now = self._time()
        if self.processed_prompts <= 0 and not force:
            self._write_status_snapshot(status="running")
            return
        if not force:
            if self.processed_prompts < self._next_emit_processed and now - self._last_emit_at < self._emit_sec_interval:
                self._write_status_snapshot(status="running")
                return
        self._emit(force=force)

    def _emit(self, *, force: bool) -> None:
        snapshot = self.snapshot_dict(status="running")
        processed_prompts = int(snapshot["processed_prompts"])
        if processed_prompts <= 0 and not force:
            self._write_status_snapshot(status="running")
            return
        for line in self._format_lines(snapshot):
            self._emit_callback(line)
        self._last_emit_at = self._time()
        while self._next_emit_processed <= processed_prompts:
            self._next_emit_processed += self._emit_prompt_interval
        self._write_status_snapshot(status="running")

    def _format_lines(self, snapshot: dict[str, object]) -> list[str]:
        lines = [
            _format_throughput_line(
                scope="overall",
                prompts_processed=int(snapshot["processed_prompts"]),
                prompts_total=int(snapshot["total_prompts"]),
                failed_prompts=int(snapshot["failed_prompts"]),
                rate_per_min=float(snapshot["rate_per_min"]),
                eta_sec=snapshot["eta_sec"],
            )
        ]
        models_payload = snapshot["models"]
        assert isinstance(models_payload, dict)
        replicas_payload = snapshot["replicas"]
        assert isinstance(replicas_payload, dict)
        for model_name, payload in models_payload.items():
            assert isinstance(payload, dict)
            replicas = replicas_payload.get(model_name) if isinstance(replicas_payload, dict) else None
            replica_count = len(replicas) if isinstance(replicas, dict) else 0
            lines.append(
                _format_throughput_line(
                    scope=f"model={model_name}",
                    prompts_processed=int(payload["processed_prompts"]),
                    prompts_total=int(payload["total_prompts"]),
                    failed_prompts=int(payload["failed_prompts"]),
                    rate_per_min=float(payload["rate_per_min"]),
                    eta_sec=payload["eta_sec"],
                    replica_count=replica_count if replica_count > 1 else None,
                    active_replica_count=int(payload.get("active_replicas", 0)),
                )
            )
            if isinstance(replicas, dict) and len(replicas) > 1:
                for replica_name, replica_payload in sorted(replicas.items()):
                    assert isinstance(replica_payload, dict)
                    lines.append(
                        _format_replica_line(
                            model_name=model_name,
                            replica_name=replica_name,
                            replica_payload=replica_payload,
                        )
                    )
        return lines

    def _write_status_snapshot(self, *, status: str) -> None:
        if self._status_path is None:
            return
        self._status_path.parent.mkdir(parents=True, exist_ok=True)
        self._status_path.write_text(
            json.dumps(self.snapshot_dict(status=status), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _parse_gpu_assignment(raw: str) -> list[int]:
    content = raw.strip()[1:-1].strip()
    if not content:
        return []
    return [int(part.strip()) for part in content.split(",") if part.strip()]


def _extract_load_duration(message: str) -> float | None:
    match = re.search(r"in\s+([0-9]+(?:\.[0-9]+)?)s$", message)
    if match is None:
        return None
    return float(match.group(1))


def _elapsed_since(
    first_active_at: float | None,
    default_started_at: float,
    time_source: Callable[[], float],
) -> float:
    start = first_active_at if first_active_at is not None else default_started_at
    return max(time_source() - start, 0.0)


def _rate_per_min(processed: int, elapsed_sec: float) -> float:
    if processed <= 0 or elapsed_sec <= 0:
        return 0.0
    return processed / (elapsed_sec / 60.0)


def _estimate_eta_sec(*, processed: int, total: int, rate_per_min: float) -> int | None:
    remaining = max(total - processed, 0)
    if processed <= 0 or remaining <= 0 or rate_per_min <= 0:
        return None
    return int(round((remaining / rate_per_min) * 60.0))


def _format_eta(eta_sec: object) -> str | None:
    if eta_sec in (None, ""):
        return None
    total_sec = int(eta_sec)
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_throughput_line(
    *,
    scope: str,
    prompts_processed: int,
    prompts_total: int,
    failed_prompts: int,
    rate_per_min: float,
    eta_sec: object,
    replica_count: int | None = None,
    active_replica_count: int | None = None,
) -> str:
    parts = [
        "[THROUGHPUT]",
        scope,
        f"prompts={prompts_processed}/{prompts_total}",
        f"rate={rate_per_min:.1f}/min",
        f"failed={failed_prompts}",
    ]
    if replica_count is not None:
        if active_replica_count is not None and active_replica_count > 0:
            parts.append(f"replicas_active={active_replica_count}/{replica_count}")
        else:
            parts.append(f"replicas={replica_count}")
    eta = _format_eta(eta_sec)
    if eta is not None:
        parts.append(f"eta={eta}")
    return " ".join(parts)


def _format_replica_line(
    *,
    model_name: str,
    replica_name: str,
    replica_payload: dict[str, object],
) -> str:
    gpus = replica_payload.get("gpu_assignment") or []
    return (
        f"[REPLICA] model={model_name} {replica_name} {gpus} "
        f"{replica_payload.get('state', 'unknown')} "
        f"{replica_payload.get('processed_prompts', 0)}/{replica_payload.get('assigned_prompts', 0)} "
        f"rate={float(replica_payload.get('rate_per_min', 0.0)):.1f}/min "
        f"startup_failures={int(replica_payload.get('startup_failures', 0))}"
    )
