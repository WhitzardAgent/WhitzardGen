from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aigc.registry.models import ModelInfo
from aigc.runtime.payloads import TaskPayload


class RunLedgerWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def append_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self._lock:
            for record in records:
                self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()


def build_sample_ledger_records(
    *,
    run_id: str,
    model: ModelInfo,
    task_payload: TaskPayload,
    task_result: dict[str, Any] | None,
    error_message: str | None = None,
    failure_category: str | None = None,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    prompt_lookup = {prompt.prompt_id: prompt for prompt in task_payload.prompts}
    resolved_timestamp = timestamp or datetime.now(UTC).isoformat()
    batch_items = list((task_result or {}).get("model_result", {}).get("batch_items", []))
    if not batch_items:
        return [
            _failure_record(
                timestamp=resolved_timestamp,
                run_id=run_id,
                model=model,
                task_payload=task_payload,
                prompt_id=prompt.prompt_id,
                prompt=prompt.prompt,
                negative_prompt=prompt.negative_prompt,
                language=prompt.language,
                prompt_metadata=dict(prompt.metadata),
                batch_index=index,
                error_message=error_message
                or _extract_error_message(task_result)
                or "Task failed before prompt-level outputs were available.",
                failure_category=failure_category,
            )
            for index, prompt in enumerate(task_payload.prompts)
        ]

    records: list[dict[str, Any]] = []
    for batch_index, batch_item in enumerate(batch_items):
        prompt_id = str(batch_item.get("prompt_id", ""))
        prompt = prompt_lookup.get(prompt_id)
        if prompt is None:
            continue
        batch_metadata = dict(batch_item.get("metadata", {}))
        artifacts = list(batch_item.get("artifacts", []))
        first_artifact = artifacts[0] if artifacts else {}
        status = str(batch_item.get("status", "failed"))
        records.append(
            {
                "timestamp": resolved_timestamp,
                "run_id": run_id,
                "task_id": task_payload.task_id,
                "replica_id": batch_metadata.get(
                    "replica_id",
                    task_payload.runtime_config.get("replica_id"),
                ),
                "model_name": model.name,
                "prompt_id": prompt.prompt_id,
                "prompt": prompt.prompt,
                "negative_prompt": prompt.negative_prompt,
                "language": prompt.language,
                "prompt_metadata": dict(prompt.metadata),
                "status": status,
                "artifact_type": first_artifact.get("type"),
                "artifact_path": first_artifact.get("path"),
                "artifact_count": len(artifacts),
                "error_message": (
                    None
                    if status == "success"
                    else error_message
                    or str(batch_item.get("logs") or batch_item.get("error") or "").strip()
                    or _extract_error_message(task_result)
                ),
                "failure_category": (
                    None
                    if status == "success"
                    else failure_category
                ),
                "batch_id": batch_metadata.get("batch_id", task_payload.batch_id),
                "batch_index": batch_metadata.get("batch_index", batch_index),
                "execution_mode": batch_metadata.get(
                    "execution_mode",
                    (task_result or {}).get("execution_mode", task_payload.execution_mode),
                ),
                "gpu_assignment": batch_metadata.get(
                    "gpu_assignment",
                    task_payload.runtime_config.get("gpu_assignment"),
                ),
                "conditioning_source": dict(prompt.metadata.get("conditioning_source", {}))
                if isinstance(prompt.metadata.get("conditioning_source"), dict)
                else prompt.metadata.get("conditioning_source"),
                "conditioning_bindings": dict(prompt.metadata.get("conditioning_bindings", {}))
                if isinstance(prompt.metadata.get("conditioning_bindings"), dict)
                else prompt.metadata.get("conditioning_bindings"),
                "conditioning_artifact_ids": list(prompt.metadata.get("conditioning_artifact_ids", []))
                if isinstance(prompt.metadata.get("conditioning_artifact_ids"), list)
                else [],
            }
        )
    if records:
        return records
    return [
        _failure_record(
            timestamp=resolved_timestamp,
            run_id=run_id,
            model=model,
            task_payload=task_payload,
            prompt_id=prompt.prompt_id,
            prompt=prompt.prompt,
            negative_prompt=prompt.negative_prompt,
            language=prompt.language,
            prompt_metadata=dict(prompt.metadata),
            batch_index=index,
            error_message=error_message
            or _extract_error_message(task_result)
            or "Task failed before prompt-level outputs were available.",
            failure_category=failure_category,
        )
        for index, prompt in enumerate(task_payload.prompts)
    ]


def _failure_record(
    *,
    timestamp: str,
    run_id: str,
    model: ModelInfo,
    task_payload: TaskPayload,
    prompt_id: str,
    prompt: str,
    negative_prompt: str | None,
    language: str,
    prompt_metadata: dict[str, Any],
    batch_index: int,
    error_message: str,
    failure_category: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "task_id": task_payload.task_id,
        "replica_id": task_payload.runtime_config.get("replica_id"),
        "model_name": model.name,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "language": language,
        "prompt_metadata": dict(prompt_metadata),
        "status": "failed",
        "artifact_type": None,
        "artifact_path": None,
        "artifact_count": 0,
        "error_message": error_message,
        "failure_category": failure_category,
        "batch_id": task_payload.batch_id,
        "batch_index": batch_index,
        "execution_mode": task_payload.execution_mode,
        "gpu_assignment": task_payload.runtime_config.get("gpu_assignment"),
        "conditioning_source": dict(prompt_metadata.get("conditioning_source", {}))
        if isinstance(prompt_metadata.get("conditioning_source"), dict)
        else prompt_metadata.get("conditioning_source"),
        "conditioning_bindings": dict(prompt_metadata.get("conditioning_bindings", {}))
        if isinstance(prompt_metadata.get("conditioning_bindings"), dict)
        else prompt_metadata.get("conditioning_bindings"),
        "conditioning_artifact_ids": list(prompt_metadata.get("conditioning_artifact_ids", []))
        if isinstance(prompt_metadata.get("conditioning_artifact_ids"), list)
        else [],
    }


def _extract_error_message(task_result: dict[str, Any] | None) -> str | None:
    if not task_result:
        return None
    model_logs = str(task_result.get("model_result", {}).get("logs") or "").strip()
    exec_logs = str(task_result.get("execution_result", {}).get("logs") or "").strip()
    return model_logs or exec_logs or None
