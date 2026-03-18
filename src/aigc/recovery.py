from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aigc.prompts.models import PromptRecord
from aigc.run_store import (
    load_failures_summary,
    load_run_manifest,
    load_samples_ledger,
    load_task_payloads,
)


@dataclass(slots=True)
class RecoveryItem:
    model_name: str
    prompt: PromptRecord
    params: dict[str, Any]
    execution_mode: str
    source_task_id: str


@dataclass(slots=True)
class RecoveryState:
    source_run_id: str
    manifest: dict[str, Any]
    samples: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    planned_items: dict[tuple[str, str], RecoveryItem]
    successful_keys: set[tuple[str, str]]
    failed_keys: set[tuple[str, str]]
    missing_keys: set[tuple[str, str]]


@dataclass(slots=True)
class RecoveryPlan:
    recovery_mode: str
    source_run_id: str
    execution_mode: str
    prompt_source: str
    items_by_model: dict[str, list[RecoveryItem]]
    selected_count: int
    completed_count: int
    failed_count: int
    missing_count: int
    source_manifest: dict[str, Any]
    source_failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        return [model_name for model_name, items in self.items_by_model.items() if items]


class RecoveryError(RuntimeError):
    """Raised when recovery planning cannot be completed."""


def build_recovery_state(
    run_id: str,
    *,
    runs_root: str | Path | None = None,
) -> RecoveryState:
    manifest = load_run_manifest(run_id, runs_root=runs_root)
    failures = load_failures_summary(run_id, runs_root=runs_root)
    samples = load_samples_ledger(run_id, runs_root=runs_root)
    task_payloads = load_task_payloads(run_id, runs_root=runs_root)

    planned_items: dict[tuple[str, str], RecoveryItem] = {}
    task_lookup: dict[str, Any] = {}
    for payload in task_payloads:
        task_lookup[payload.task_id] = payload
        for prompt in payload.prompts:
            key = (payload.model_name, prompt.prompt_id)
            planned_items[key] = RecoveryItem(
                model_name=payload.model_name,
                prompt=PromptRecord(
                    prompt_id=prompt.prompt_id,
                    prompt=prompt.prompt,
                    language=prompt.language,
                    negative_prompt=prompt.negative_prompt,
                    metadata=dict(prompt.metadata),
                ),
                params=dict(payload.params),
                execution_mode=payload.execution_mode,
                source_task_id=payload.task_id,
            )

    successful_keys: set[tuple[str, str]] = set()
    failed_keys: set[tuple[str, str]] = set()
    for record in samples:
        model_name = str(record.get("model_name", ""))
        prompt_id = str(record.get("prompt_id", ""))
        if not model_name or not prompt_id:
            continue
        key = (model_name, prompt_id)
        status = str(record.get("status", ""))
        if status == "success":
            successful_keys.add(key)
            failed_keys.discard(key)
        elif status == "failed" and key not in successful_keys:
            failed_keys.add(key)

    for failure in failures:
        task_id = str(failure.get("task_id", ""))
        if not task_id or task_id not in task_lookup:
            continue
        payload = task_lookup[task_id]
        for prompt in payload.prompts:
            key = (payload.model_name, prompt.prompt_id)
            if key not in successful_keys:
                failed_keys.add(key)

    missing_keys = set(planned_items) - successful_keys - failed_keys
    return RecoveryState(
        source_run_id=run_id,
        manifest=manifest,
        samples=samples,
        failures=failures,
        planned_items=planned_items,
        successful_keys=successful_keys,
        failed_keys=failed_keys,
        missing_keys=missing_keys,
    )


def build_retry_plan(
    run_id: str,
    *,
    model_name: str | None = None,
    runs_root: str | Path | None = None,
) -> RecoveryPlan:
    state = build_recovery_state(run_id, runs_root=runs_root)
    selected_keys = _filter_keys(state.failed_keys, model_name=model_name)
    return _build_recovery_plan_from_state(
        state,
        recovery_mode="retry",
        selected_keys=selected_keys,
    )


def build_resume_plan(
    run_id: str,
    *,
    model_name: str | None = None,
    runs_root: str | Path | None = None,
) -> RecoveryPlan:
    state = build_recovery_state(run_id, runs_root=runs_root)
    selected_keys = _filter_keys(state.missing_keys, model_name=model_name)
    return _build_recovery_plan_from_state(
        state,
        recovery_mode="resume",
        selected_keys=selected_keys,
    )


def _build_recovery_plan_from_state(
    state: RecoveryState,
    *,
    recovery_mode: str,
    selected_keys: list[tuple[str, str]],
) -> RecoveryPlan:
    items_by_model: dict[str, list[RecoveryItem]] = {}
    for key in selected_keys:
        item = state.planned_items.get(key)
        if item is None:
            continue
        items_by_model.setdefault(item.model_name, []).append(item)

    return RecoveryPlan(
        recovery_mode=recovery_mode,
        source_run_id=state.source_run_id,
        execution_mode=str(state.manifest.get("execution_mode", "real")),
        prompt_source=str(state.manifest.get("prompt_source", "-")),
        items_by_model=items_by_model,
        selected_count=sum(len(items) for items in items_by_model.values()),
        completed_count=len(state.successful_keys),
        failed_count=len(state.failed_keys),
        missing_count=len(state.missing_keys),
        source_manifest=state.manifest,
        source_failures=state.failures,
    )


def recovery_plan_to_dict(plan: RecoveryPlan) -> dict[str, Any]:
    return {
        "recovery_mode": plan.recovery_mode,
        "source_run_id": plan.source_run_id,
        "execution_mode": plan.execution_mode,
        "prompt_source": plan.prompt_source,
        "selected_count": plan.selected_count,
        "completed_count": plan.completed_count,
        "failed_count": plan.failed_count,
        "missing_count": plan.missing_count,
        "models": plan.model_names,
        "items_by_model": {
            model_name: [
                {
                    "model_name": item.model_name,
                    "prompt_id": item.prompt.prompt_id,
                    "prompt": item.prompt.prompt,
                    "language": item.prompt.language,
                    "negative_prompt": item.prompt.negative_prompt,
                    "params": item.params,
                    "execution_mode": item.execution_mode,
                    "source_task_id": item.source_task_id,
                }
                for item in items
            ]
            for model_name, items in plan.items_by_model.items()
        },
    }


def _filter_keys(
    keys: set[tuple[str, str]],
    *,
    model_name: str | None,
) -> list[tuple[str, str]]:
    ordered = sorted(keys)
    if model_name is None:
        return ordered
    return [key for key in ordered if key[0] == model_name]
