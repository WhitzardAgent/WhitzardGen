from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class TaskProgressReporter:
    model_name: str
    replica_id: int | None
    task_id: str
    batch_id: str | None
    batch_size: int
    emit_event: Callable[[dict[str, Any]], None] | None = None
    emit_log: Callable[[str], None] | None = None
    supports_true_progress: bool = False
    _last_logged_phase: str | None = None
    _last_logged_bucket: int = -1

    def phase(self, phase: str, *, message: str | None = None) -> None:
        payload = self._base_payload(phase=phase, message=message)
        if self.emit_event is not None:
            self.emit_event(payload)
        if self.emit_log is not None and phase != self._last_logged_phase:
            self.emit_log(self._format_log_line(payload))
            self._last_logged_phase = phase

    def step(
        self,
        current_step: int,
        total_steps: int,
        *,
        phase: str = "generating",
        message: str | None = None,
        supports_true_progress: bool = True,
    ) -> None:
        payload = self._base_payload(
            phase=phase,
            current_step=max(int(current_step), 0),
            total_steps=max(int(total_steps), 0),
            message=message,
            supports_true_progress=supports_true_progress,
        )
        if self.emit_event is not None:
            self.emit_event(payload)

        if self.emit_log is None:
            return
        total = max(int(total_steps), 1)
        current = max(int(current_step), 0)
        bucket = min(10, int((current / total) * 10))
        if bucket != self._last_logged_bucket or phase != self._last_logged_phase or current >= total:
            self.emit_log(self._format_log_line(payload))
            self._last_logged_bucket = bucket
            self._last_logged_phase = phase

    def _base_payload(
        self,
        *,
        phase: str,
        current_step: int | None = None,
        total_steps: int | None = None,
        message: str | None = None,
        supports_true_progress: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": "task_progress",
            "model_name": self.model_name,
            "replica_id": self.replica_id,
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "phase": phase,
            "supports_true_progress": self.supports_true_progress if supports_true_progress is None else supports_true_progress,
        }
        if current_step is not None:
            payload["current_step"] = current_step
        if total_steps is not None:
            payload["total_steps"] = total_steps
        if message:
            payload["message"] = message
        return payload

    def _format_log_line(self, payload: dict[str, Any]) -> str:
        parts = [
            "[progress]",
            f"model={payload['model_name']}",
            f"replica={payload['replica_id'] if payload['replica_id'] is not None else '-'}",
            f"task={payload['task_id']}",
            f"batch={payload.get('batch_size', 0)}",
            f"phase={payload.get('phase', 'unknown')}",
        ]
        if payload.get("current_step") is not None and payload.get("total_steps") is not None:
            parts.append(f"step={payload['current_step']}/{payload['total_steps']}")
        if payload.get("supports_true_progress"):
            parts.append("true_progress=yes")
        if payload.get("message"):
            parts.append(f"message={payload['message']}")
        return " ".join(parts)


def emit_progress_event(progress_callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)
