from __future__ import annotations

import contextlib
import json
import inspect
import os
import re
import secrets
import shlex
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from aigc.env import EnvManager, EnvManagerError, EnvironmentRecord
from aigc.exporters import build_dataset_records, export_jsonl
from aigc.prompts import PromptRecord, load_prompts
from aigc.recovery import RecoveryItem, RecoveryPlan
from aigc.registry.models import ModelInfo
from aigc.registry import load_registry
from aigc.run_ledger import RunLedgerWriter, build_sample_ledger_records
from aigc.run_store import write_failures_summary, write_run_manifest
from aigc.runtime.persistent_ipc import (
    PersistentWorkerQueueManager,
    create_queue_method_names,
    register_parent_queues,
    unregister_parent_queues,
)
from aigc.runtime.payloads import TaskPayload, TaskPrompt
from aigc.runtime_telemetry import RunTelemetry
from aigc.settings import get_default_seed, get_runs_root
from aigc.utils.progress import (
    LoggedRunProgress,
    RunProgress,
    RunHeaderData,
    RunSummaryData,
    NullRunProgress,
    summarize_task_statuses,
)
from aigc.utils.runtime_logging import RunLogger

REPO_ROOT = Path(__file__).resolve().parents[2]


class RunFlowError(RuntimeError):
    """Raised when the minimal run flow fails."""


@dataclass(slots=True)
class RunSummary:
    status: str
    run_id: str
    model_names: list[str]
    prompt_file: str
    output_dir: str
    tasks_scheduled: int
    records_exported: int
    export_path: str
    execution_mode: str = "real"
    running_log_path: str | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def model_name(self) -> str | None:
        if len(self.model_names) == 1:
            return self.model_names[0]
        return None

    @property
    def mock_mode(self) -> bool:
        return self.execution_mode == "mock"


@dataclass(slots=True)
class PreparedTask:
    model: ModelInfo
    payload: TaskPayload
    task_file: Path
    result_file: Path
    batch_number: int
    total_tasks_for_model: int


@dataclass(slots=True)
class ReplicaPlan:
    replica_id: int
    gpu_assignment: list[int]
    tasks: list[PreparedTask]


@dataclass(slots=True)
class FailurePolicy:
    continue_on_error: bool = False
    max_failures: int | None = None
    max_failure_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "continue_on_error": self.continue_on_error,
            "max_failures": self.max_failures,
            "max_failure_rate": self.max_failure_rate,
        }


@dataclass(slots=True)
class FailureBudget:
    total_planned_outputs: int
    failed_prompt_outputs: int = 0
    failed_tasks: int = 0

    def record(self, *, failed_prompt_outputs: int, task_failed: bool) -> None:
        self.failed_prompt_outputs += max(failed_prompt_outputs, 0)
        if task_failed:
            self.failed_tasks += 1

    @property
    def failure_rate(self) -> float:
        if self.total_planned_outputs <= 0:
            return 0.0
        return self.failed_prompt_outputs / self.total_planned_outputs

    def check_thresholds(self, policy: FailurePolicy) -> str | None:
        if policy.max_failures is not None and self.failed_prompt_outputs > policy.max_failures:
            return (
                "Failure policy threshold exceeded: "
                f"failed_outputs={self.failed_prompt_outputs} "
                f"> max_failures={policy.max_failures}"
            )
        if (
            policy.max_failure_rate is not None
            and self.total_planned_outputs > 0
            and self.failure_rate > policy.max_failure_rate
        ):
            return (
                "Failure policy threshold exceeded: "
                f"failure_rate={self.failure_rate:.3f} "
                f"> max_failure_rate={policy.max_failure_rate:.3f} "
                f"({self.failed_prompt_outputs}/{self.total_planned_outputs})"
            )
        return None


@dataclass(slots=True)
class TaskExecutionOutcome:
    result_payload: dict
    task_failed: bool
    failed_prompt_count: int
    successful_prompt_count: int
    duration_sec: float
    failure_message: str | None = None
    failure_category: str | None = None


@dataclass(slots=True)
class _WorkerCrashContext:
    phase: str
    task_id: str | None
    exitcode: int | None
    error: str
    traceback_text: str = ""
    log_tail: list[str] = field(default_factory=list)


def _build_profile_manifest(
    *,
    profile_name: str | None,
    profile_path: str | None,
    profile_generation_defaults: dict[str, object] | None,
    profile_runtime: dict[str, object] | None,
) -> dict[str, object] | None:
    if not profile_name and not profile_path and not profile_generation_defaults and not profile_runtime:
        return None
    payload: dict[str, object] = {}
    if profile_name:
        payload["name"] = profile_name
    if profile_path:
        payload["path"] = profile_path
    if profile_generation_defaults:
        payload["generation_defaults"] = dict(profile_generation_defaults)
    if profile_runtime:
        payload["runtime"] = dict(profile_runtime)
    return payload


def _mirror_runtime_event(*, logger: RunLogger, terminal_progress: RunProgress, message: str) -> None:
    logger.log(message, already_timestamped=True)
    terminal_progress.env_message(message)


def _handle_runtime_event(
    *,
    logger: RunLogger,
    terminal_progress: RunProgress,
    telemetry: RunTelemetry | None,
    message: str,
) -> None:
    if telemetry is not None:
        telemetry.record_runtime_event(message)
    _mirror_runtime_event(
        logger=logger,
        terminal_progress=terminal_progress,
        message=message,
    )


def _build_per_model_summary(
    *,
    models: list[ModelInfo],
    all_records: list[dict],
    task_results: list[tuple[ModelInfo, TaskPayload, dict]],
    worker_strategies: dict[str, str],
    replica_plans_by_model: dict[str, list[ReplicaPlan]],
) -> dict[str, dict[str, object]]:
    per_model_summary: dict[str, dict[str, object]] = {}
    for model in models:
        model_records = [record for record in all_records if record["model_name"] == model.name]
        replica_plans = replica_plans_by_model.get(model.name, [])
        per_model_summary[model.name] = {
            "record_count": len(model_records),
            "task_count": len([result for result in task_results if result[0].name == model.name]),
            "modality": model.modality,
            "task_type": model.task_type,
            "worker_strategy": worker_strategies.get(model.name, "per_task_worker"),
            "replica_count": len(replica_plans) or 1,
            "conda_env_name": model.conda_env_name,
            "execution_mode": model.execution_mode,
            "backend_execution_mode": model.backend_execution_mode,
            "supports_batch_prompts": bool(model.capabilities.get("supports_batch_prompts")),
            "max_batch_size": int(model.capabilities.get("max_batch_size", 1)),
            "gpus_per_replica": model.gpus_per_replica,
            "supports_multi_replica": model.supports_multi_replica,
            "max_gpus": model.max_gpus,
            "local_paths": dict(model.local_paths),
            "replicas": [
                {
                    "replica_id": replica_plan.replica_id,
                    "gpu_assignment": replica_plan.gpu_assignment,
                    "task_count": len(replica_plan.tasks),
                }
                for replica_plan in replica_plans
            ],
        }
    return per_model_summary


def run_single_model(
    *,
    model_name: str,
    prompt_file: str | Path,
    out_dir: str | Path | None = None,
    run_name: str | None = None,
    execution_mode: str = "real",
    mock_mode: bool | None = None,
    env_manager: EnvManager | None = None,
    worker_runner: Callable[[EnvironmentRecord, Path, Path], tuple[int, str]] | None = None,
    batch_limit: int | None = None,
    progress: RunProgress | None = None,
    profile_generation_defaults: dict[str, object] | None = None,
    continue_on_error: bool | None = None,
    max_failures: int | None = None,
    max_failure_rate: float | None = None,
) -> RunSummary:
    return run_models(
        model_names=[model_name],
        prompt_file=prompt_file,
        out_dir=out_dir,
        run_name=run_name,
        execution_mode=execution_mode,
        mock_mode=mock_mode,
        env_manager=env_manager,
        worker_runner=worker_runner,
        batch_limit=batch_limit,
        progress=progress,
        profile_generation_defaults=profile_generation_defaults,
        continue_on_error=continue_on_error,
        max_failures=max_failures,
        max_failure_rate=max_failure_rate,
    )


def run_recovery_plan(
    *,
    recovery_plan: RecoveryPlan,
    out_dir: str | Path | None = None,
    run_name: str | None = None,
    env_manager: EnvManager | None = None,
    worker_runner: Callable[[EnvironmentRecord, Path, Path], tuple[int, str]] | None = None,
    batch_limit: int | None = None,
    progress: RunProgress | None = None,
    continue_on_error: bool | None = None,
    max_failures: int | None = None,
    max_failure_rate: float | None = None,
) -> RunSummary:
    registry = load_registry()
    if not recovery_plan.model_names:
        raise RunFlowError(
            f"No {recovery_plan.recovery_mode}able prompt outputs were found for run {recovery_plan.source_run_id}."
        )

    models = [registry.get_model(model_name) for model_name in recovery_plan.model_names]
    modality = models[0].modality
    invalid_models = [model.name for model in models if model.modality != modality]
    if invalid_models:
        raise RunFlowError(
            "Recovery runs require all selected models to share one modality. "
            f"Expected {modality}, got mismatched models: {', '.join(invalid_models)}."
        )

    resolved_execution_mode = recovery_plan.execution_mode
    run_id = _generate_run_id(run_name or f"{recovery_plan.recovery_mode}-{recovery_plan.source_run_id}")
    run_root = Path(out_dir) if out_dir else get_runs_root() / run_id
    tasks_dir = run_root / "tasks"
    workdir_root = run_root / "workdir"
    exports_dir = run_root / "exports"
    artifacts_root = run_root / "artifacts"
    workers_root = run_root / "workers"
    for directory in (run_root, tasks_dir, workdir_root, exports_dir, artifacts_root, workers_root):
        directory.mkdir(parents=True, exist_ok=True)
    base_progress = progress or NullRunProgress()
    run_logger = RunLogger(log_path=run_root / "running.log")
    progress = LoggedRunProgress(base=base_progress, logger=run_logger)
    ledger_writer = RunLedgerWriter(run_root / "samples.jsonl")
    runtime_status_path = run_root / "runtime_status.json"
    telemetry = RunTelemetry(
        run_id=run_id,
        execution_mode=resolved_execution_mode,
        emit_callback=progress.env_message,
        status_path=runtime_status_path,
    )
    prompt_source_label = f"{recovery_plan.recovery_mode}:{recovery_plan.source_run_id}:{recovery_plan.prompt_source}"
    failure_policy = _resolve_failure_policy(
        continue_on_error=continue_on_error,
        max_failures=max_failures,
        max_failure_rate=max_failure_rate,
        profile_runtime=None,
    )

    total_stages = 8
    stage_index = 1
    progress.run_header(
        RunHeaderData(
            run_id=run_id,
            execution_mode=resolved_execution_mode,
            model_names=[model.name for model in models],
            prompt_source=prompt_source_label,
            prompt_count=recovery_plan.selected_count,
            output_dir=str(run_root),
            running_log_path=str(run_root / "running.log"),
            profile_label=f"{recovery_plan.recovery_mode}:{recovery_plan.source_run_id}",
        )
    )

    progress.stage_start(stage_index, total_stages, "Loading recovery plan")
    manager = env_manager or EnvManager(registry=registry)
    progress.stage_end(stage_index, total_stages, "Loading recovery plan")
    stage_index += 1

    created_at = datetime.now(UTC).isoformat()
    initial_manifest = {
        "run_id": run_id,
        "status": "running",
        "created_at": created_at,
        "models": [model.name for model in models],
        "prompt_source": prompt_source_label,
        "prompt_count": recovery_plan.selected_count,
        "execution_mode": resolved_execution_mode,
        "task_count": 0,
        "output_dir": str(run_root),
        "export_paths": {
            "samples_jsonl": str(run_root / "samples.jsonl"),
            "runtime_status_json": str(runtime_status_path),
        },
        "samples_ledger_path": str(run_root / "samples.jsonl"),
        "runtime_status_path": str(runtime_status_path),
        "running_log_path": str(run_logger.log_path),
        "registry_path": str(registry.registry_path),
        "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
        "parent_run_id": recovery_plan.source_run_id,
        "source_run_id": recovery_plan.source_run_id,
        "recovery_mode": recovery_plan.recovery_mode,
        "recovered_item_count": recovery_plan.selected_count,
        "failure_policy": failure_policy.to_dict(),
    }
    write_run_manifest(run_root, initial_manifest)
    failures: list[dict[str, object]] = []
    task_results: list[tuple[ModelInfo, TaskPayload, dict]] = []
    export_path: Path | None = None
    failures_path = run_root / "failures.json"
    manifest_path = run_root / "run_manifest.json"
    threshold_stop_reason: str | None = None
    try:
        progress.stage_start(stage_index, total_stages, "Ensuring environments")
        env_records: dict[str, EnvironmentRecord] = {}
        for model in models:
            progress.env_message(f"Ensuring environment for model: {model.name}")
            env_records[model.name] = _resolve_environment_record(
                manager=manager,
                model_name=model.name,
                execution_mode=resolved_execution_mode,
                progress=progress.env_message,
            )
        progress.stage_end(stage_index, total_stages, "Ensuring environments")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Preparing recovery tasks")
        prepared_tasks_by_model, worker_strategies = _prepare_recovery_tasks(
            models=models,
            recovery_plan=recovery_plan,
            tasks_dir=tasks_dir,
            workdir_root=workdir_root,
            artifacts_root=artifacts_root,
            batch_limit=batch_limit,
            worker_runner=worker_runner,
            registry=registry,
        )
        failure_budget = FailureBudget(
            total_planned_outputs=sum(
                len(prepared_task.payload.prompts)
                for prepared_tasks in prepared_tasks_by_model.values()
                for prepared_task in prepared_tasks
            )
        )
        telemetry.set_plan(prepared_tasks_by_model=prepared_tasks_by_model)
        progress.stage_end(stage_index, total_stages, "Preparing recovery tasks")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Running tasks")
        replica_plans_by_model: dict[str, list[ReplicaPlan]] = {}
        for model in models:
            strategy = worker_strategies[model.name]
            prepared_tasks = prepared_tasks_by_model.get(model.name, [])
            if strategy == "persistent_worker" and prepared_tasks:
                replica_plans = _plan_replicas_for_model(
                    model=model,
                    prepared_tasks=prepared_tasks,
                    execution_mode=resolved_execution_mode,
                )
                replica_plans_by_model[model.name] = replica_plans
                telemetry.register_replica_assignments(
                    model_name=model.name,
                    replica_plans=replica_plans,
                )
                _log_replica_plan(progress=progress, model=model, replica_plans=replica_plans)
                if len(replica_plans) == 1:
                    replica_plan = replica_plans[0]
                    with _PersistentWorkerSession(
                        model=model,
                        env_record=env_records[model.name],
                        execution_mode=resolved_execution_mode,
                        replica_id=replica_plan.replica_id,
                        gpu_assignment=replica_plan.gpu_assignment,
                        replica_log_path=_replica_log_path(workers_root, model.name, replica_plan.replica_id),
                        log_callback=lambda line: _handle_runtime_event(
                            logger=run_logger,
                            terminal_progress=base_progress,
                            telemetry=telemetry,
                            message=line,
                        ),
                    ) as session:
                        for prepared_task in replica_plan.tasks:
                            outcome = _execute_prepared_task(
                                prepared_task=prepared_task,
                                runner=lambda task, session=session: session.run_task(task),
                                run_id=run_id,
                                run_root=run_root,
                                failures=failures,
                                progress=progress,
                                ledger_writer=ledger_writer,
                                telemetry=telemetry,
                            )
                            task_results.append((model, prepared_task.payload, outcome.result_payload))
                            _apply_failure_policy_after_task(
                                outcome=outcome,
                                prepared_task=prepared_task,
                                failure_policy=failure_policy,
                                failure_budget=failure_budget,
                                progress=progress,
                                model_name=model.name,
                            )
                else:
                    _run_persistent_worker_replicas(
                        model=model,
                        env_record=env_records[model.name],
                        replica_plans=replica_plans,
                        execution_mode=resolved_execution_mode,
                        run_id=run_id,
                        run_root=run_root,
                        workers_root=workers_root,
                        failures=failures,
                        progress=progress,
                        ledger_writer=ledger_writer,
                        failure_policy=failure_policy,
                        failure_budget=failure_budget,
                        task_results_out=task_results,
                        telemetry=telemetry,
                        log_callback=lambda line: _handle_runtime_event(
                            logger=run_logger,
                            terminal_progress=base_progress,
                            telemetry=telemetry,
                            message=line,
                        ),
                    )
            else:
                runner = worker_runner or _run_worker_task
                for prepared_task in prepared_tasks:
                    outcome = _execute_prepared_task(
                        prepared_task=prepared_task,
                        runner=lambda task, env_record=env_records[model.name], runner=runner: _invoke_worker_runner(
                            runner,
                            env_record,
                            task.task_file,
                            task.result_file,
                            log_callback=lambda line: _handle_runtime_event(
                                logger=run_logger,
                                terminal_progress=base_progress,
                                telemetry=telemetry,
                                message=line,
                            ),
                        ),
                        run_id=run_id,
                        run_root=run_root,
                        failures=failures,
                        progress=progress,
                        ledger_writer=ledger_writer,
                        telemetry=telemetry,
                    )
                    task_results.append((model, prepared_task.payload, outcome.result_payload))
                    _apply_failure_policy_after_task(
                        outcome=outcome,
                        prepared_task=prepared_task,
                        failure_policy=failure_policy,
                        failure_budget=failure_budget,
                        progress=progress,
                        model_name=model.name,
                    )
                if prepared_tasks:
                    replica_plans_by_model[model.name] = [
                        ReplicaPlan(replica_id=0, gpu_assignment=[], tasks=list(prepared_tasks))
                    ]
                    telemetry.register_replica_assignments(
                        model_name=model.name,
                        replica_plans=replica_plans_by_model[model.name],
                    )
        task_results.sort(key=lambda item: item[1].task_id)
        progress.stage_end(stage_index, total_stages, "Running tasks")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Exporting dataset")
        all_records = []
        record_index = 1
        for model, payload, result_payload in task_results:
            task_records = build_dataset_records(
                run_id=run_id,
                model=model,
                task_payload=payload,
                task_result=result_payload,
                record_start_index=record_index,
            )
            all_records.extend(task_records)
            record_index += len(task_records)
        export_path = export_jsonl(all_records, exports_dir / "dataset.jsonl")
        progress.stage_end(stage_index, total_stages, "Exporting dataset")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Writing run manifest")
        per_model_summary = _build_per_model_summary(
            models=models,
            all_records=all_records,
            task_results=task_results,
            worker_strategies=worker_strategies,
            replica_plans_by_model=replica_plans_by_model,
        )
        runtime_metrics = telemetry.finalize(
            status="completed_with_failures" if failure_budget.failed_prompt_outputs else "completed"
        )

        run_manifest = {
            "run_id": run_id,
            "status": "completed_with_failures" if failure_budget.failed_prompt_outputs else "completed",
            "created_at": created_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "models": [model.name for model in models],
            "prompt_source": prompt_source_label,
            "prompt_count": recovery_plan.selected_count,
            "execution_mode": resolved_execution_mode,
            "task_count": len(task_results),
            "records_exported": len(all_records),
            "output_dir": str(run_root),
            "export_paths": {
                "dataset_jsonl": str(export_path),
                "samples_jsonl": str(run_root / "samples.jsonl"),
                "runtime_status_json": str(runtime_status_path),
                "running_log": str(run_logger.log_path),
            },
            "export_path": str(export_path),
            "running_log_path": str(run_logger.log_path),
            "samples_ledger_path": str(run_root / "samples.jsonl"),
            "runtime_status_path": str(runtime_status_path),
            "failures_path": str(failures_path),
            "per_model_summary": per_model_summary,
            "runtime_metrics": runtime_metrics,
            "registry_path": str(registry.registry_path),
            "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
            "parent_run_id": recovery_plan.source_run_id,
            "source_run_id": recovery_plan.source_run_id,
            "recovery_mode": recovery_plan.recovery_mode,
            "recovered_item_count": recovery_plan.selected_count,
            "failure_policy": failure_policy.to_dict(),
            "failed_prompt_outputs": failure_budget.failed_prompt_outputs,
            "failure_rate": failure_budget.failure_rate,
        }
        manifest_path = write_run_manifest(run_root, run_manifest)
        failures_path = write_failures_summary(run_root, failures)
        progress.stage_end(stage_index, total_stages, "Writing run manifest")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Done")
        task_statuses = [
            result["model_result"]["status"]
            for _model, _payload, result in task_results
        ]
        success_tasks, failed_tasks = summarize_task_statuses(task_statuses)
        progress.print_summary(
            RunSummaryData(
                status="completed" if not failure_budget.failed_prompt_outputs else "completed_with_failures",
                run_id=run_id,
                execution_mode=resolved_execution_mode,
                model_names=[model.name for model in models],
                prompt_count=recovery_plan.selected_count,
                task_count=len(task_results),
                success_tasks=success_tasks,
                failed_tasks=failed_tasks,
                output_dir=str(run_root),
                dataset_path=str(export_path),
                manifest_path=str(manifest_path),
                failures_path=str(failures_path),
                running_log_path=str(run_logger.log_path),
                wall_time_sec=runtime_metrics.get("elapsed_sec"),
                processed_prompt_outputs=runtime_metrics.get("processed_prompts"),
                failed_prompt_outputs=runtime_metrics.get("failed_prompts"),
                throughput_per_min=runtime_metrics.get("rate_per_min"),
                stop_reason=None,
            )
        )
        progress.stage_end(stage_index, total_stages, "Done")
    except Exception as exc:
        if threshold_stop_reason is None and str(exc).startswith("Failure policy threshold exceeded"):
            threshold_stop_reason = str(exc)
        progress.env_message(f"[run] ERROR: {exc}")
        failure_manifest = {
            **initial_manifest,
            "status": "failed",
            "completed_at": datetime.now(UTC).isoformat(),
            "task_count": len(
                [path for path in tasks_dir.rglob("*.json") if ".result." not in path.name]
            ),
            "error": str(exc),
            "stop_reason": threshold_stop_reason,
            "running_log_path": str(run_logger.log_path),
            "samples_ledger_path": str(run_root / "samples.jsonl"),
            "runtime_status_path": str(runtime_status_path),
            "failures_path": str(failures_path),
            "failure_policy": failure_policy.to_dict(),
            "runtime_metrics": telemetry.finalize(status="failed"),
        }
        manifest_path = write_run_manifest(run_root, failure_manifest)
        failures_path = write_failures_summary(run_root, failures)
        task_statuses = [
            result["model_result"]["status"]
            for _model, _payload, result in task_results
        ]
        success_tasks, failed_tasks = summarize_task_statuses(task_statuses)
        progress.print_summary(
            RunSummaryData(
                status="failed",
                run_id=run_id,
                execution_mode=resolved_execution_mode,
                model_names=[model.name for model in models],
                prompt_count=recovery_plan.selected_count,
                task_count=len(task_results),
                success_tasks=success_tasks,
                failed_tasks=max(failed_tasks, 1),
                output_dir=str(run_root),
                dataset_path=str(export_path) if export_path is not None else "-",
                manifest_path=str(manifest_path),
                failures_path=str(failures_path),
                running_log_path=str(run_logger.log_path),
                wall_time_sec=failure_manifest["runtime_metrics"].get("elapsed_sec")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                processed_prompt_outputs=failure_manifest["runtime_metrics"].get("processed_prompts")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                failed_prompt_outputs=failure_manifest["runtime_metrics"].get("failed_prompts")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                throughput_per_min=failure_manifest["runtime_metrics"].get("rate_per_min")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                stop_reason=threshold_stop_reason or str(exc),
            )
        )
        raise
    finally:
        ledger_writer.close()
        run_logger.close()

    return RunSummary(
        status="completed_with_failures" if failure_budget.failed_prompt_outputs else "completed",
        run_id=run_id,
        model_names=[model.name for model in models],
        prompt_file=prompt_source_label,
        output_dir=str(run_root),
        tasks_scheduled=len(task_results),
        records_exported=len(all_records),
        export_path=str(export_path),
        execution_mode=resolved_execution_mode,
        running_log_path=str(run_logger.log_path),
        stop_reason=None,
    )


def run_models(
    *,
    model_names: list[str],
    prompt_file: str | Path,
    out_dir: str | Path | None = None,
    run_name: str | None = None,
    execution_mode: str = "real",
    mock_mode: bool | None = None,
    env_manager: EnvManager | None = None,
    worker_runner: Callable[[EnvironmentRecord, Path, Path], tuple[int, str]] | None = None,
    batch_limit: int | None = None,
    progress: RunProgress | None = None,
    profile_name: str | None = None,
    profile_path: str | None = None,
    profile_generation_defaults: dict[str, object] | None = None,
    profile_runtime: dict[str, object] | None = None,
    continue_on_error: bool | None = None,
    max_failures: int | None = None,
    max_failure_rate: float | None = None,
) -> RunSummary:
    resolved_execution_mode = _resolve_execution_mode(
        execution_mode=execution_mode,
        mock_mode=mock_mode,
    )
    registry = load_registry()
    if not model_names:
        raise RunFlowError("aigc run requires at least one model.")
    models = [registry.get_model(model_name) for model_name in model_names]
    modality = models[0].modality
    invalid_models = [model.name for model in models if model.modality != modality]
    if invalid_models:
        raise RunFlowError(
            "aigc run requires all selected models to share one modality. "
            f"Expected {modality}, got mismatched models: {', '.join(invalid_models)}."
        )

    run_id = _generate_run_id(run_name)
    run_root = Path(out_dir) if out_dir else get_runs_root() / run_id
    tasks_dir = run_root / "tasks"
    workdir_root = run_root / "workdir"
    exports_dir = run_root / "exports"
    artifacts_root = run_root / "artifacts"
    workers_root = run_root / "workers"
    for directory in (run_root, tasks_dir, workdir_root, exports_dir, artifacts_root, workers_root):
        directory.mkdir(parents=True, exist_ok=True)
    base_progress = progress or NullRunProgress()
    run_logger = RunLogger(log_path=run_root / "running.log")
    progress = LoggedRunProgress(base=base_progress, logger=run_logger)

    total_stages = 9
    stage_index = 1
    profile_label = profile_name or Path(profile_path).stem if profile_path else None
    progress.run_header(
        RunHeaderData(
            run_id=run_id,
            execution_mode=resolved_execution_mode,
            model_names=[model.name for model in models],
            prompt_source=str(prompt_file),
            output_dir=str(run_root),
            running_log_path=str(run_root / "running.log"),
            profile_label=profile_label,
        )
    )

    progress.stage_start(stage_index, total_stages, "Loading prompts")
    prompts = load_prompts(prompt_file, warn=progress.env_message)
    progress.stage_end(stage_index, total_stages, "Loading prompts")
    stage_index += 1

    progress.stage_start(stage_index, total_stages, "Validating prompts")
    if not prompts:
        raise RunFlowError("Prompt file did not produce any valid prompts.")
    progress.stage_end(stage_index, total_stages, "Validating prompts")
    stage_index += 1
    progress.env_message(f"[run] Prompt count: {len(prompts)}")

    progress.stage_start(stage_index, total_stages, "Resolving models")

    manager = env_manager or EnvManager(registry=registry)
    failure_policy = _resolve_failure_policy(
        continue_on_error=continue_on_error,
        max_failures=max_failures,
        max_failure_rate=max_failure_rate,
        profile_runtime=profile_runtime,
    )
    progress.stage_end(stage_index, total_stages, "Resolving models")
    stage_index += 1
    ledger_path = run_root / "samples.jsonl"
    ledger_writer = RunLedgerWriter(ledger_path)
    runtime_status_path = run_root / "runtime_status.json"
    telemetry = RunTelemetry(
        run_id=run_id,
        execution_mode=resolved_execution_mode,
        emit_callback=progress.env_message,
        status_path=runtime_status_path,
    )
    created_at = datetime.now(UTC).isoformat()
    initial_manifest = {
        "run_id": run_id,
        "status": "running",
        "created_at": created_at,
        "models": [model.name for model in models],
        "prompt_source": str(prompt_file),
        "prompt_count": len(prompts),
        "execution_mode": resolved_execution_mode,
        "task_count": 0,
        "output_dir": str(run_root),
        "export_paths": {
            "samples_jsonl": str(ledger_path),
            "runtime_status_json": str(runtime_status_path),
        },
        "samples_ledger_path": str(ledger_path),
        "runtime_status_path": str(runtime_status_path),
        "running_log_path": str(run_logger.log_path),
        "registry_path": str(registry.registry_path),
        "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
        "profile": _build_profile_manifest(
            profile_name=profile_name,
            profile_path=profile_path,
            profile_generation_defaults=profile_generation_defaults,
            profile_runtime=profile_runtime,
        ),
        "failure_policy": failure_policy.to_dict(),
    }
    write_run_manifest(run_root, initial_manifest)
    failures: list[dict[str, object]] = []
    task_results: list[tuple[ModelInfo, TaskPayload, dict]] = []
    export_path: Path | None = None
    failures_path = run_root / "failures.json"
    manifest_path = run_root / "run_manifest.json"
    threshold_stop_reason: str | None = None
    try:
        progress.stage_start(stage_index, total_stages, "Ensuring environments")
        env_records: dict[str, EnvironmentRecord] = {}
        for model in models:
            progress.env_message(f"Ensuring environment for model: {model.name}")
            env_records[model.name] = _resolve_environment_record(
                manager=manager,
                model_name=model.name,
                execution_mode=resolved_execution_mode,
                progress=progress.env_message,
            )
        progress.stage_end(stage_index, total_stages, "Ensuring environments")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Preparing tasks")
        prepared_tasks_by_model: dict[str, list[PreparedTask]] = {}
        worker_strategies: dict[str, str] = {}
        task_counter = 1
        for model in models:
            strategy = _resolve_worker_strategy(
                registry=registry,
                model=model,
                worker_runner=worker_runner,
            )
            worker_strategies[model.name] = strategy

            model_slug = _slugify(model.name)
            model_tasks_dir = tasks_dir / model_slug
            model_workdir_root = workdir_root / model_slug
            model_artifacts_root = artifacts_root / model_slug
            for directory in (model_tasks_dir, model_workdir_root, model_artifacts_root):
                directory.mkdir(parents=True, exist_ok=True)

            batched_prompts = list(
                _batch_prompts_for_model(
                    model=model,
                    prompts=prompts,
                    batch_limit=batch_limit,
                    generation_defaults=profile_generation_defaults,
                )
            )
            total_tasks_for_model = len(batched_prompts)
            model_prepared_tasks: list[PreparedTask] = []
            for batch_number, prompt_batch in enumerate(batched_prompts, start=1):
                task_id = f"task_{task_counter:06d}"
                batch_id = f"{model_slug}_batch_{batch_number:06d}"
                task_workdir = model_workdir_root / task_id
                payload = TaskPayload(
                    task_id=task_id,
                    model_name=model.name,
                    execution_mode=resolved_execution_mode,
                    prompts=[_prompt_to_task_prompt(prompt) for prompt in prompt_batch],
                    params=_default_generation_params(
                        model,
                        prompt_batch,
                        generation_defaults=profile_generation_defaults,
                    ),
                    workdir=str(task_workdir),
                    batch_id=batch_id,
                    runtime_config={"worker_strategy": strategy},
                    worker_strategy=strategy,
                )
                task_file = model_tasks_dir / f"{task_id}.json"
                result_file = model_tasks_dir / f"{task_id}.result.json"
                task_file.write_text(
                    json.dumps(payload.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                model_prepared_tasks.append(
                    PreparedTask(
                        model=model,
                        payload=payload,
                        task_file=task_file,
                        result_file=result_file,
                        batch_number=batch_number,
                        total_tasks_for_model=total_tasks_for_model,
                    )
                )
                task_counter += 1

            prepared_tasks_by_model[model.name] = model_prepared_tasks

        failure_budget = FailureBudget(
            total_planned_outputs=sum(
                len(prepared_task.payload.prompts)
                for prepared_tasks in prepared_tasks_by_model.values()
                for prepared_task in prepared_tasks
            )
        )
        telemetry.set_plan(prepared_tasks_by_model=prepared_tasks_by_model)
        progress.stage_end(stage_index, total_stages, "Preparing tasks")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Running tasks")
        replica_plans_by_model: dict[str, list[ReplicaPlan]] = {}
        for model in models:
            strategy = worker_strategies[model.name]
            prepared_tasks = prepared_tasks_by_model.get(model.name, [])
            if strategy == "persistent_worker" and prepared_tasks:
                replica_plans = _plan_replicas_for_model(
                    model=model,
                    prepared_tasks=prepared_tasks,
                    execution_mode=resolved_execution_mode,
                )
                replica_plans_by_model[model.name] = replica_plans
                telemetry.register_replica_assignments(
                    model_name=model.name,
                    replica_plans=replica_plans,
                )
                _log_replica_plan(progress=progress, model=model, replica_plans=replica_plans)
                if len(replica_plans) == 1:
                    replica_plan = replica_plans[0]
                    with _PersistentWorkerSession(
                        model=model,
                        env_record=env_records[model.name],
                        execution_mode=resolved_execution_mode,
                        replica_id=replica_plan.replica_id,
                        gpu_assignment=replica_plan.gpu_assignment,
                        replica_log_path=_replica_log_path(workers_root, model.name, replica_plan.replica_id),
                        log_callback=lambda line: _handle_runtime_event(
                            logger=run_logger,
                            terminal_progress=base_progress,
                            telemetry=telemetry,
                            message=line,
                        ),
                    ) as session:
                        for prepared_task in replica_plan.tasks:
                            outcome = _execute_prepared_task(
                                prepared_task=prepared_task,
                                runner=lambda task, session=session: session.run_task(task),
                                run_id=run_id,
                                run_root=run_root,
                                failures=failures,
                                progress=progress,
                                ledger_writer=ledger_writer,
                                telemetry=telemetry,
                            )
                            task_results.append((model, prepared_task.payload, outcome.result_payload))
                            _apply_failure_policy_after_task(
                                outcome=outcome,
                                prepared_task=prepared_task,
                                failure_policy=failure_policy,
                                failure_budget=failure_budget,
                                progress=progress,
                                model_name=model.name,
                            )
                else:
                    _run_persistent_worker_replicas(
                        model=model,
                        env_record=env_records[model.name],
                        replica_plans=replica_plans,
                        execution_mode=resolved_execution_mode,
                        run_id=run_id,
                        run_root=run_root,
                        workers_root=workers_root,
                        failures=failures,
                        progress=progress,
                        ledger_writer=ledger_writer,
                        telemetry=telemetry,
                        failure_policy=failure_policy,
                        failure_budget=failure_budget,
                        task_results_out=task_results,
                        log_callback=lambda line: _handle_runtime_event(
                            logger=run_logger,
                            terminal_progress=base_progress,
                            telemetry=telemetry,
                            message=line,
                        ),
                    )
            else:
                runner = worker_runner or _run_worker_task
                for prepared_task in prepared_tasks:
                    outcome = _execute_prepared_task(
                        prepared_task=prepared_task,
                        runner=lambda task, env_record=env_records[model.name], runner=runner: _invoke_worker_runner(
                            runner,
                            env_record,
                            task.task_file,
                            task.result_file,
                            log_callback=lambda line: _handle_runtime_event(
                                logger=run_logger,
                                terminal_progress=base_progress,
                                telemetry=telemetry,
                                message=line,
                            ),
                        ),
                        run_id=run_id,
                        run_root=run_root,
                        failures=failures,
                        progress=progress,
                        ledger_writer=ledger_writer,
                        telemetry=telemetry,
                    )
                    task_results.append((model, prepared_task.payload, outcome.result_payload))
                    _apply_failure_policy_after_task(
                        outcome=outcome,
                        prepared_task=prepared_task,
                        failure_policy=failure_policy,
                        failure_budget=failure_budget,
                        progress=progress,
                        model_name=model.name,
                    )
                if prepared_tasks:
                    replica_plans_by_model[model.name] = [
                        ReplicaPlan(replica_id=0, gpu_assignment=[], tasks=list(prepared_tasks))
                    ]
                    telemetry.register_replica_assignments(
                        model_name=model.name,
                        replica_plans=replica_plans_by_model[model.name],
                    )
        task_results.sort(key=lambda item: item[1].task_id)
        progress.stage_end(stage_index, total_stages, "Running tasks")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Exporting dataset")
        all_records = []
        record_index = 1
        for model, payload, result_payload in task_results:
            task_records = build_dataset_records(
                run_id=run_id,
                model=model,
                task_payload=payload,
                task_result=result_payload,
                record_start_index=record_index,
            )
            all_records.extend(task_records)
            record_index += len(task_records)

        export_path = export_jsonl(all_records, exports_dir / "dataset.jsonl")
        progress.stage_end(stage_index, total_stages, "Exporting dataset")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Writing run manifest")
        per_model_summary = _build_per_model_summary(
            models=models,
            all_records=all_records,
            task_results=task_results,
            worker_strategies=worker_strategies,
            replica_plans_by_model=replica_plans_by_model,
        )
        runtime_metrics = telemetry.finalize(
            status="completed_with_failures" if failure_budget.failed_prompt_outputs else "completed"
        )

        run_manifest = {
            "run_id": run_id,
            "status": "completed_with_failures" if failure_budget.failed_prompt_outputs else "completed",
            "created_at": created_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "models": [model.name for model in models],
            "prompt_source": str(prompt_file),
            "prompt_count": len(prompts),
            "execution_mode": resolved_execution_mode,
            "task_count": len(task_results),
            "records_exported": len(all_records),
            "output_dir": str(run_root),
            "export_paths": {
                "dataset_jsonl": str(export_path),
                "samples_jsonl": str(ledger_path),
                "runtime_status_json": str(runtime_status_path),
                "running_log": str(run_logger.log_path),
            },
            "export_path": str(export_path),
            "running_log_path": str(run_logger.log_path),
            "samples_ledger_path": str(ledger_path),
            "runtime_status_path": str(runtime_status_path),
            "failures_path": str(failures_path),
            "per_model_summary": per_model_summary,
            "runtime_metrics": runtime_metrics,
            "registry_path": str(registry.registry_path),
            "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
            "profile": _build_profile_manifest(
                profile_name=profile_name,
                profile_path=profile_path,
                profile_generation_defaults=profile_generation_defaults,
                profile_runtime=profile_runtime,
            ),
            "failure_policy": failure_policy.to_dict(),
            "failed_prompt_outputs": failure_budget.failed_prompt_outputs,
            "failure_rate": failure_budget.failure_rate,
        }
        (run_root / "run.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        manifest_path = write_run_manifest(run_root, run_manifest)
        failures_path = write_failures_summary(run_root, failures)
        progress.stage_end(stage_index, total_stages, "Writing run manifest")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Done")

        # Compute simple success / failure counts for the final summary.
        task_statuses = [
            result["model_result"]["status"]
            for _model, _payload, result in task_results
        ]
        success_tasks, failed_tasks = summarize_task_statuses(task_statuses)

        progress.print_summary(
            RunSummaryData(
                status="completed" if not failure_budget.failed_prompt_outputs else "completed_with_failures",
                run_id=run_id,
                execution_mode=resolved_execution_mode,
                model_names=[model.name for model in models],
                prompt_count=len(prompts),
                task_count=len(task_results),
                success_tasks=success_tasks,
                failed_tasks=failed_tasks,
                output_dir=str(run_root),
                dataset_path=str(export_path),
                manifest_path=str(manifest_path),
                failures_path=str(failures_path),
                running_log_path=str(run_logger.log_path),
                wall_time_sec=runtime_metrics.get("elapsed_sec"),
                processed_prompt_outputs=runtime_metrics.get("processed_prompts"),
                failed_prompt_outputs=runtime_metrics.get("failed_prompts"),
                throughput_per_min=runtime_metrics.get("rate_per_min"),
                stop_reason=None,
            )
        )

        progress.stage_end(stage_index, total_stages, "Done")
    except Exception as exc:
        if threshold_stop_reason is None and str(exc).startswith("Failure policy threshold exceeded"):
            threshold_stop_reason = str(exc)
        progress.env_message(f"[run] ERROR: {exc}")
        failure_manifest = {
            **initial_manifest,
            "status": "failed",
            "completed_at": datetime.now(UTC).isoformat(),
            "task_count": len(
                [
                    path
                    for path in tasks_dir.rglob("*.json")
                    if ".result." not in path.name
                ]
            ),
            "error": str(exc),
            "stop_reason": threshold_stop_reason,
            "running_log_path": str(run_logger.log_path),
            "samples_ledger_path": str(ledger_path),
            "runtime_status_path": str(runtime_status_path),
            "failures_path": str(failures_path),
            "failure_policy": failure_policy.to_dict(),
            "runtime_metrics": telemetry.finalize(status="failed"),
        }
        manifest_path = write_run_manifest(run_root, failure_manifest)
        failures_path = write_failures_summary(run_root, failures)
        task_statuses = [
            result["model_result"]["status"]
            for _model, _payload, result in task_results
        ]
        success_tasks, failed_tasks = summarize_task_statuses(task_statuses)
        progress.print_summary(
            RunSummaryData(
                status="failed",
                run_id=run_id,
                execution_mode=resolved_execution_mode,
                model_names=[model.name for model in models],
                prompt_count=len(prompts) if "prompts" in locals() else 0,
                task_count=len(task_results),
                success_tasks=success_tasks,
                failed_tasks=max(failed_tasks, 1),
                output_dir=str(run_root),
                dataset_path=str(export_path) if export_path is not None else "-",
                manifest_path=str(manifest_path),
                failures_path=str(failures_path),
                running_log_path=str(run_logger.log_path),
                wall_time_sec=failure_manifest["runtime_metrics"].get("elapsed_sec")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                processed_prompt_outputs=failure_manifest["runtime_metrics"].get("processed_prompts")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                failed_prompt_outputs=failure_manifest["runtime_metrics"].get("failed_prompts")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                throughput_per_min=failure_manifest["runtime_metrics"].get("rate_per_min")
                if isinstance(failure_manifest.get("runtime_metrics"), dict)
                else None,
                stop_reason=threshold_stop_reason or str(exc),
            )
        )
        raise
    finally:
        ledger_writer.close()
        run_logger.close()

    return RunSummary(
        status="completed_with_failures" if failure_budget.failed_prompt_outputs else "completed",
        run_id=run_id,
        model_names=[model.name for model in models],
        prompt_file=str(prompt_file),
        output_dir=str(run_root),
        tasks_scheduled=len(task_results),
        records_exported=len(all_records),
        export_path=str(export_path),
        execution_mode=resolved_execution_mode,
        running_log_path=str(run_logger.log_path),
        stop_reason=None,
    )


class _PersistentWorkerSession:
    def __init__(
        self,
        *,
        model: ModelInfo,
        env_record: EnvironmentRecord,
        execution_mode: str,
        replica_id: int = 0,
        gpu_assignment: list[int] | None = None,
        replica_log_path: Path | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.env_record = env_record
        self.execution_mode = execution_mode
        self.replica_id = replica_id
        self.gpu_assignment = list(gpu_assignment or [])
        self.replica_log_path = replica_log_path
        self.log_callback = log_callback
        self.process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._replica_log_handle = None
        self._state = "starting"
        self._current_task_id: str | None = None
        self._recent_log_lines: deque[str] = deque(maxlen=50)
        self._authkey_hex = secrets.token_hex(16)
        self._command_method, self._event_method, self._log_method = create_queue_method_names()
        self._manager_socket_path = Path("/tmp") / (
            f"aigc_{_slugify(self.model.name)}_{self.replica_id}_{self._authkey_hex[:10]}.sock"
        )
        self._manager = None
        self._manager_server = None
        self._manager_thread: threading.Thread | None = None
        self._command_queue = None
        self._event_queue = None
        self._log_queue = None

    def __enter__(self) -> "_PersistentWorkerSession":
        try:
            self._open_replica_log()
            self._start_queue_manager()
            extra_args = [
                "--model-name",
                self.model.name,
                "--execution-mode",
                self.execution_mode,
                "--replica-id",
                str(self.replica_id),
                "--gpu-assignment",
                ",".join(str(gpu_id) for gpu_id in self.gpu_assignment),
                "--manager-address",
                str(self._manager.address),
                "--manager-authkey",
                self._authkey_hex,
                "--command-method",
                self._command_method,
                "--event-method",
                self._event_method,
                "--log-method",
                self._log_method,
            ]
            if self.model.registry_source:
                extra_args.extend(["--registry-file", str(self.model.registry_source)])
            command, env = _build_worker_command_and_env(
                env_record=self.env_record,
                model_name=self.model.name,
                execution_mode=self.execution_mode,
                module_name="aigc.runtime.persistent_worker",
                extra_args=extra_args,
                env_overrides=_build_replica_env_overrides(self.gpu_assignment),
            )
            self.process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._start_stdout_pump()
            self._start_stderr_pump()
            event = self._wait_for_event(
                expected_event="ready",
                phase="startup",
                task_id=None,
            )
            self._state = "ready"
            if event.get("event") != "ready":
                raise RunFlowError(
                    f"Persistent worker for {self.model.name} replica={self.replica_id} did not become ready: {event}"
                )
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    def run_task(self, prepared_task: PreparedTask) -> tuple[int, str]:
        self._current_task_id = prepared_task.payload.task_id
        self._state = "running_task"
        self._write_command(
            {
                "command": "run_task",
                "task_file": str(prepared_task.task_file),
                "result_file": str(prepared_task.result_file),
            }
        )
        started_event = self._wait_for_event(
            expected_event="task_started",
            phase="accept_task",
            task_id=prepared_task.payload.task_id,
        )
        if started_event.get("task_id") != prepared_task.payload.task_id:
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} accepted an unexpected task event: {started_event}"
            )
        event = self._wait_for_event(
            expected_event="task_complete",
            phase="task_execution",
            task_id=prepared_task.payload.task_id,
        )
        if event.get("event") != "task_complete":
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} returned unexpected event: {event}"
            )
        self._state = "ready"
        self._current_task_id = None
        status = str(event.get("status", "failed"))
        error = str(event.get("error", "")).strip()
        return (0 if status != "failed" else 1), error

    def close(self) -> None:
        try:
            self._close_process()
        finally:
            self._shutdown_queue_manager()
            self._close_replica_log()

    def _close_process(self) -> None:
        if self.process is None:
            self._state = "stopped"
            return
        try:
            if self.process.poll() is None and self._state not in {"failed", "stopped"}:
                self._state = "shutting_down"
                self._write_command({"command": "shutdown"})
                self._wait_for_event(
                    expected_event="shutdown",
                    phase="shutdown",
                    task_id=self._current_task_id,
                    allow_process_exit=True,
                )
        finally:
            returncode = self.process.wait(timeout=30)
            self._drain_log_queue()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=5)
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=5)
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()
            self.process = None
            if returncode != 0 and self._state != "failed":
                raise RunFlowError(
                    f"Persistent worker for {self.model.name} replica={self.replica_id} exited with code {returncode}."
                )
            self._state = "stopped"

    def _write_command(self, payload: dict[str, object]) -> None:
        if self._command_queue is None:
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} is not writable."
            )
        if self.process is None or self.process.poll() is not None:
            return
        self._command_queue.put(payload)

    def _wait_for_event(
        self,
        *,
        expected_event: str,
        phase: str,
        task_id: str | None,
        allow_process_exit: bool = False,
    ) -> dict[str, object]:
        if self._event_queue is None:
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} is not readable."
            )
        while True:
            try:
                event = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                self._drain_log_queue()
                if self.process is not None and self.process.poll() is not None:
                    if allow_process_exit:
                        return {}
                    raise self._build_process_exit_error(phase=phase, task_id=task_id)
                continue
            self._drain_log_queue()
            event_name = str(event.get("event", ""))
            if event_name == expected_event:
                return event
            if event_name == "worker_failed":
                if event.get("task_id") is None and task_id is not None:
                    event["task_id"] = task_id
                self._state = "failed"
                raise self._build_worker_failed_error(event)
            if event_name == "shutdown" and expected_event == "shutdown":
                return event
            self._log_line(
                f"[worker][{self.model.name}][replica={self.replica_id}] supervisor ignored event={event_name}"
            )

    def _start_stderr_pump(self) -> None:
        if self.process is None or self.process.stderr is None:
            return

        def pump() -> None:
            assert self.process is not None and self.process.stderr is not None
            for raw_line in self.process.stderr:
                text = raw_line.rstrip("\n")
                if text:
                    self._log_line(text)

        self._stderr_thread = threading.Thread(
            target=pump,
            name=f"{_slugify(self.model.name)}-replica-{self.replica_id}-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _start_stdout_pump(self) -> None:
        if self.process is None or self.process.stdout is None:
            return

        def pump() -> None:
            assert self.process is not None and self.process.stdout is not None
            for raw_line in self.process.stdout:
                text = raw_line.rstrip("\n")
                if text:
                    self._log_line(
                        f"[worker][{self.model.name}][replica={self.replica_id}] stdout: {text}"
                    )

        self._stdout_thread = threading.Thread(
            target=pump,
            name=f"{_slugify(self.model.name)}-replica-{self.replica_id}-stdout",
            daemon=True,
        )
        self._stdout_thread.start()

    def _start_queue_manager(self) -> None:
        command_queue = queue.Queue()
        event_queue = queue.Queue()
        log_queue = queue.Queue()
        self._manager_socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._manager_socket_path.exists():
            self._manager_socket_path.unlink()
        register_parent_queues(
            command_method=self._command_method,
            event_method=self._event_method,
            log_method=self._log_method,
            command_queue=command_queue,
            event_queue=event_queue,
            log_queue=log_queue,
        )
        manager = PersistentWorkerQueueManager(
            address=str(self._manager_socket_path),
            authkey=bytes.fromhex(self._authkey_hex),
        )
        server = manager.get_server()
        thread = threading.Thread(
            target=server.serve_forever,
            name=f"{_slugify(self.model.name)}-replica-{self.replica_id}-ipc",
            daemon=True,
        )
        thread.start()
        self._manager = manager
        self._manager_server = server
        self._manager_thread = thread
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._log_queue = log_queue

    def _shutdown_queue_manager(self) -> None:
        self._drain_log_queue()
        if self._manager_server is not None:
            stop_event = getattr(self._manager_server, "stop_event", None)
            if stop_event is not None:
                stop_event.set()
            listener = getattr(self._manager_server, "listener", None)
            if listener is not None:
                with contextlib.suppress(Exception):
                    listener.close()
            self._manager_server = None
        if self._manager_thread is not None:
            self._manager_thread.join(timeout=2)
            self._manager_thread = None
        self._manager = None
        if self._manager_socket_path.exists():
            self._manager_socket_path.unlink()
        unregister_parent_queues(self._command_method, self._event_method, self._log_method)
        self._command_queue = None
        self._event_queue = None
        self._log_queue = None

    def _drain_log_queue(self) -> None:
        if self._log_queue is None:
            return
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._log_line(str(line))

    def _build_process_exit_error(self, *, phase: str, task_id: str | None) -> RunFlowError:
        exitcode = self.process.poll() if self.process is not None else None
        context = _WorkerCrashContext(
            phase=phase,
            task_id=task_id,
            exitcode=exitcode,
            error="worker exited unexpectedly",
            log_tail=list(self._recent_log_lines),
        )
        self._state = "failed"
        return RunFlowError(self._format_crash_context(context))

    def _build_worker_failed_error(self, event: dict[str, object]) -> RunFlowError:
        context = _WorkerCrashContext(
            phase=str(event.get("phase", "unknown")),
            task_id=str(event.get("task_id")) if event.get("task_id") is not None else None,
            exitcode=self.process.poll() if self.process is not None else None,
            error=str(event.get("error", "worker_failed")),
            traceback_text=str(event.get("traceback", "")),
            log_tail=list(self._recent_log_lines),
        )
        return RunFlowError(self._format_crash_context(context))

    def _format_crash_context(self, context: _WorkerCrashContext) -> str:
        lines = [
            f"Persistent worker for {self.model.name} replica={self.replica_id} failed during {context.phase}.",
        ]
        if context.task_id:
            if context.phase in {"accept_task", "task_dispatch"}:
                lines.append(f"Worker died before accepting task {context.task_id}.")
            elif context.phase == "task_execution":
                lines.append(f"Worker died while executing task {context.task_id}.")
            else:
                lines.append(f"Last task: {context.task_id}")
        if context.exitcode is not None:
            lines.append(f"Worker exit code: {context.exitcode}")
        if context.error:
            lines.append(f"Reason: {context.error}")
        if context.traceback_text.strip():
            lines.append(f"Traceback:\n{context.traceback_text.strip()}")
        if context.log_tail:
            lines.append("Recent worker logs:")
            lines.extend(context.log_tail[-10:])
        return "\n".join(lines)

    def _open_replica_log(self) -> None:
        if self.replica_log_path is None:
            return
        self.replica_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._replica_log_handle = self.replica_log_path.open("a", encoding="utf-8")

    def _close_replica_log(self) -> None:
        if self._replica_log_handle is None:
            return
        self._replica_log_handle.close()
        self._replica_log_handle = None

    def _log_line(self, line: str) -> None:
        text = line.rstrip("\n")
        if not text:
            return
        self._recent_log_lines.append(text)
        if self._replica_log_handle is not None:
            self._replica_log_handle.write(text + "\n")
            self._replica_log_handle.flush()
        if self.log_callback is not None:
            self.log_callback(text)


def _execute_prepared_task(
    *,
    prepared_task: PreparedTask,
    runner: Callable[[PreparedTask], tuple[int, str]],
    run_id: str,
    run_root: Path,
    failures: list[dict[str, object]],
    progress: RunProgress,
    ledger_writer: RunLedgerWriter,
    telemetry: RunTelemetry | None = None,
    state_lock: threading.Lock | None = None,
) -> TaskExecutionOutcome:
    task_started_at = time.monotonic()
    if telemetry is not None:
        telemetry.record_task_start(
            task_id=prepared_task.payload.task_id,
            model_name=prepared_task.model.name,
            replica_id=prepared_task.payload.runtime_config.get("replica_id"),
        )
    progress.task_start(
        current=prepared_task.batch_number,
        total=prepared_task.total_tasks_for_model,
        model_name=prepared_task.model.name,
        prompts=len(prepared_task.payload.prompts),
        execution_mode=prepared_task.payload.execution_mode,
    )

    try:
        returncode, logs = runner(prepared_task)
    except Exception as exc:
        error_message = str(exc).strip() or exc.__class__.__name__
        failure_category = _categorize_failure(error_message=error_message)
        synthetic_result = _build_task_failure_result_payload(
            prepared_task=prepared_task,
            error_message=error_message,
        )
        _record_task_failure(
            prepared_task=prepared_task,
            run_id=run_id,
            run_root=run_root,
            failures=failures,
            progress=progress,
            ledger_writer=ledger_writer,
            task_result=synthetic_result,
            error_message=error_message,
            failure_category=failure_category,
            state_lock=state_lock,
        )
        if telemetry is not None:
            telemetry.record_task_outcome(
                task_id=prepared_task.payload.task_id,
                model_name=prepared_task.model.name,
                replica_id=prepared_task.payload.runtime_config.get("replica_id"),
                successful_prompts=0,
                failed_prompts=len(prepared_task.payload.prompts),
                task_failed=True,
            )
        raise

    if not prepared_task.result_file.exists():
        error_message = "Worker did not produce a result file."
        failure_category = "worker_startup_error"
        synthetic_result = _build_task_failure_result_payload(
            prepared_task=prepared_task,
            error_message=error_message,
        )
        _record_task_failure(
            prepared_task=prepared_task,
            run_id=run_id,
            run_root=run_root,
            failures=failures,
            progress=progress,
            ledger_writer=ledger_writer,
            task_result=synthetic_result,
            error_message=error_message,
            failure_category=failure_category,
            state_lock=state_lock,
        )
        outcome = TaskExecutionOutcome(
            result_payload=synthetic_result,
            task_failed=True,
            failed_prompt_count=len(prepared_task.payload.prompts),
            successful_prompt_count=0,
            duration_sec=time.monotonic() - task_started_at,
            failure_message=f"Worker did not produce result file for {prepared_task.payload.task_id}.",
            failure_category=failure_category,
        )
        if telemetry is not None:
            telemetry.record_task_outcome(
                task_id=prepared_task.payload.task_id,
                model_name=prepared_task.model.name,
                replica_id=prepared_task.payload.runtime_config.get("replica_id"),
                successful_prompts=0,
                failed_prompts=outcome.failed_prompt_count,
                task_failed=True,
            )
        return outcome

    result_payload = json.loads(prepared_task.result_file.read_text(encoding="utf-8"))
    execution_logs = _merge_diagnostic_logs(
        result_payload.get("execution_result", {}).get("logs"),
        result_payload.get("model_result", {}).get("logs"),
        logs,
    )
    model_status = result_payload["model_result"]["status"]
    if returncode != 0 and model_status == "failed":
        failure_category = _categorize_failure(
            error_message=execution_logs,
            task_result=result_payload,
        )
        _record_task_failure(
            prepared_task=prepared_task,
            run_id=run_id,
            run_root=run_root,
            failures=failures,
            progress=progress,
            ledger_writer=ledger_writer,
            task_result=result_payload,
            error_message=execution_logs,
            failure_category=failure_category,
            state_lock=state_lock,
        )
        outcome = TaskExecutionOutcome(
            result_payload=result_payload,
            task_failed=True,
            failed_prompt_count=_count_failed_prompt_outputs(prepared_task.payload, result_payload),
            successful_prompt_count=_count_successful_prompt_outputs(result_payload),
            duration_sec=time.monotonic() - task_started_at,
            failure_message=f"Worker failed for {prepared_task.payload.task_id}:\n{execution_logs}",
            failure_category=failure_category,
        )
        if telemetry is not None:
            telemetry.record_task_outcome(
                task_id=prepared_task.payload.task_id,
                model_name=prepared_task.model.name,
                replica_id=prepared_task.payload.runtime_config.get("replica_id"),
                successful_prompts=outcome.successful_prompt_count,
                failed_prompts=outcome.failed_prompt_count,
                task_failed=True,
            )
        return outcome
    if model_status == "failed":
        failure_category = _categorize_failure(
            error_message=execution_logs,
            task_result=result_payload,
        )
        _record_task_failure(
            prepared_task=prepared_task,
            run_id=run_id,
            run_root=run_root,
            failures=failures,
            progress=progress,
            ledger_writer=ledger_writer,
            task_result=result_payload,
            error_message=execution_logs,
            failure_category=failure_category,
            state_lock=state_lock,
        )
        outcome = TaskExecutionOutcome(
            result_payload=result_payload,
            task_failed=True,
            failed_prompt_count=_count_failed_prompt_outputs(prepared_task.payload, result_payload),
            successful_prompt_count=_count_successful_prompt_outputs(result_payload),
            duration_sec=time.monotonic() - task_started_at,
            failure_message=f"Task {prepared_task.payload.task_id} failed:\n{execution_logs}",
            failure_category=failure_category,
        )
        if telemetry is not None:
            telemetry.record_task_outcome(
                task_id=prepared_task.payload.task_id,
                model_name=prepared_task.model.name,
                replica_id=prepared_task.payload.runtime_config.get("replica_id"),
                successful_prompts=outcome.successful_prompt_count,
                failed_prompts=outcome.failed_prompt_count,
                task_failed=True,
            )
        return outcome

    successful_artifacts = 0
    for batch_item in result_payload.get("model_result", {}).get("batch_items", []):
        if batch_item.get("status") != "success":
            continue
        successful_artifacts += len(batch_item.get("artifacts", []))

    progress.task_end(
        current=prepared_task.batch_number,
        total=prepared_task.total_tasks_for_model,
        model_name=prepared_task.model.name,
        status=model_status,
        artifacts=successful_artifacts or None,
    )
    _append_ledger_records(
        ledger_writer=ledger_writer,
        run_id=run_id,
        model=prepared_task.model,
        task_payload=prepared_task.payload,
        task_result=result_payload,
        state_lock=state_lock,
    )
    outcome = TaskExecutionOutcome(
        result_payload=result_payload,
        task_failed=False,
        failed_prompt_count=_count_failed_prompt_outputs(prepared_task.payload, result_payload),
        successful_prompt_count=_count_successful_prompt_outputs(result_payload),
        duration_sec=time.monotonic() - task_started_at,
        failure_message=None,
        failure_category=None,
    )
    if telemetry is not None:
        telemetry.record_task_outcome(
            task_id=prepared_task.payload.task_id,
            model_name=prepared_task.model.name,
            replica_id=prepared_task.payload.runtime_config.get("replica_id"),
            successful_prompts=outcome.successful_prompt_count,
            failed_prompts=outcome.failed_prompt_count,
            task_failed=False,
        )
    return outcome


def _run_worker_task(
    env_record: EnvironmentRecord,
    task_file: Path,
    result_file: Path,
    *,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
    worker_log_file = task_file.with_suffix(".worker.log")
    command, env = _build_worker_command_and_env(
        env_record=env_record,
        model_name=payload.model_name,
        execution_mode=payload.execution_mode,
        module_name="aigc.runtime.worker",
        extra_args=["--task-file", str(task_file), "--result-file", str(result_file)],
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        output_lines.append(raw_line)
        text = raw_line.rstrip("\n")
        if text and log_callback is not None:
            log_callback(text)
    returncode = process.wait()
    logs = "".join(output_lines).strip()
    if returncode != 0:
        diagnostics = [
            f"Worker command exit code: {returncode}",
            f"Command: {shlex.join(command)}",
            f"Task file: {task_file}",
            f"Result file: {result_file}",
            f"Worker log: {worker_log_file}",
        ]
        if logs:
            diagnostics.append(f"Command output:\n{logs}")
        else:
            diagnostics.append("Command output: <empty>")
        logs = "\n\n".join(diagnostics)
    if logs:
        worker_log_file.write_text(logs + "\n", encoding="utf-8")
    return returncode, logs


def _build_task_failure_result_payload(
    *,
    prepared_task: PreparedTask,
    error_message: str,
) -> dict[str, object]:
    return {
        "task_id": prepared_task.payload.task_id,
        "model_name": prepared_task.payload.model_name,
        "execution_mode": prepared_task.payload.execution_mode,
        "worker_strategy": prepared_task.payload.worker_strategy,
        "plan": None,
        "execution_result": {"exit_code": 1, "logs": error_message, "outputs": {}},
        "model_result": {
            "status": "failed",
            "batch_items": [],
            "logs": error_message,
            "metadata": {},
        },
    }


def _record_task_failure(
    *,
    prepared_task: PreparedTask,
    run_id: str,
    run_root: Path,
    failures: list[dict[str, object]],
    progress: RunProgress,
    ledger_writer: RunLedgerWriter,
    task_result: dict[str, object] | None,
    error_message: str,
    failure_category: str,
    state_lock: threading.Lock | None = None,
) -> None:
    _append_ledger_records(
        ledger_writer=ledger_writer,
        run_id=run_id,
        model=prepared_task.model,
        task_payload=prepared_task.payload,
        task_result=task_result,
        error_message=error_message,
        failure_category=failure_category,
        state_lock=state_lock,
    )
    failure_record = {
        "task_id": prepared_task.payload.task_id,
        "model_name": prepared_task.model.name,
        "status": "failed",
        "error": error_message,
        "category": failure_category,
        "execution_mode": prepared_task.payload.execution_mode,
        "worker_strategy": prepared_task.payload.worker_strategy,
        "batch_id": prepared_task.payload.batch_id,
        "prompt_count": len(prepared_task.payload.prompts),
        "prompt_ids": [prompt.prompt_id for prompt in prepared_task.payload.prompts],
        "replica_id": prepared_task.payload.runtime_config.get("replica_id"),
        "gpu_assignment": prepared_task.payload.runtime_config.get("gpu_assignment"),
    }
    if state_lock is not None:
        with state_lock:
            failures.append(failure_record)
            write_failures_summary(run_root, failures)
    else:
        failures.append(failure_record)
        write_failures_summary(run_root, failures)
    progress.task_end(
        current=prepared_task.batch_number,
        total=prepared_task.total_tasks_for_model,
        model_name=prepared_task.model.name,
        status="failed",
        artifacts=None,
    )


def _count_successful_prompt_outputs(task_result: dict[str, object]) -> int:
    count = 0
    for batch_item in task_result.get("model_result", {}).get("batch_items", []):
        if batch_item.get("status") == "success":
            count += 1
    return count


def _count_failed_prompt_outputs(task_payload: TaskPayload, task_result: dict[str, object]) -> int:
    batch_items = list(task_result.get("model_result", {}).get("batch_items", []))
    if batch_items:
        return sum(1 for batch_item in batch_items if batch_item.get("status") != "success")
    if task_result.get("model_result", {}).get("status") == "failed":
        return len(task_payload.prompts)
    return 0


def _categorize_failure(
    *,
    error_message: str | None,
    task_result: dict[str, object] | None = None,
) -> str:
    lower = (error_message or "").lower()
    if "conda env" in lower or ("environment for" in lower and "not available" in lower):
        return "env_error"
    if any(
        marker in lower
        for marker in (
            "failed during startup",
            "did not become ready",
            "worker exited unexpectedly",
            "worker did not produce result file",
            "failed to connect to persistent worker queue manager",
        )
    ):
        if any(
            marker in lower
            for marker in ("load_pipeline", "load_for_persistent_worker", "from_pretrained", "loading model")
        ):
            return "model_load_error"
        return "worker_startup_error"
    if any(
        marker in lower
        for marker in ("artifact", "collect(", "collect ", "export_to_video", "failed to export", "save image")
    ):
        return "artifact_collection_error"
    if task_result is not None:
        error_type = str(task_result.get("model_result", {}).get("metadata", {}).get("error_type") or "").strip()
        if error_type:
            return "task_execution_error"
    if lower:
        return "task_execution_error"
    return "unknown_error"


def _resolve_failure_policy(
    *,
    continue_on_error: bool | None,
    max_failures: int | None,
    max_failure_rate: float | None,
    profile_runtime: dict[str, object] | None,
) -> FailurePolicy:
    runtime = dict(profile_runtime or {})
    nested = runtime.get("failure_policy")
    nested_policy = dict(nested) if isinstance(nested, dict) else {}

    def _resolve(key: str, explicit: object | None) -> object | None:
        if explicit is not None:
            return explicit
        if key in nested_policy:
            return nested_policy.get(key)
        return runtime.get(key)

    resolved_max_failures = _coerce_optional_int(_resolve("max_failures", max_failures), field_name="max_failures")
    resolved_max_failure_rate = _coerce_optional_float(
        _resolve("max_failure_rate", max_failure_rate),
        field_name="max_failure_rate",
    )
    if resolved_max_failure_rate is not None and not 0.0 <= resolved_max_failure_rate <= 1.0:
        raise RunFlowError("max_failure_rate must be between 0.0 and 1.0.")

    raw_continue = _resolve("continue_on_error", continue_on_error)
    if raw_continue is None:
        resolved_continue_on_error = (
            resolved_max_failures is not None or resolved_max_failure_rate is not None
        )
    elif isinstance(raw_continue, bool):
        resolved_continue_on_error = raw_continue
    else:
        raise RunFlowError("continue_on_error must be a boolean if provided.")

    return FailurePolicy(
        continue_on_error=resolved_continue_on_error,
        max_failures=resolved_max_failures,
        max_failure_rate=resolved_max_failure_rate,
    )


def _coerce_optional_int(value: object | None, *, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise RunFlowError(f"{field_name} must be an integer.") from exc
    if resolved < 0:
        raise RunFlowError(f"{field_name} must be >= 0.")
    return resolved


def _coerce_optional_float(value: object | None, *, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RunFlowError(f"{field_name} must be a float.") from exc


def _apply_failure_policy_after_task(
    *,
    outcome: TaskExecutionOutcome,
    prepared_task: PreparedTask,
    failure_policy: FailurePolicy,
    failure_budget: FailureBudget,
    progress: RunProgress,
    model_name: str,
) -> None:
    failure_budget.record(
        failed_prompt_outputs=outcome.failed_prompt_count,
        task_failed=outcome.task_failed,
    )
    threshold_reason = failure_budget.check_thresholds(failure_policy)
    if threshold_reason:
        raise RunFlowError(threshold_reason)
    if outcome.task_failed:
        if failure_policy.continue_on_error:
            progress.env_message(
                f"[run][{model_name}] continuing after failed task {prepared_task.payload.task_id} "
                f"category={outcome.failure_category or 'unknown_error'}"
            )
            return
        raise RunFlowError(outcome.failure_message or f"Task {prepared_task.payload.task_id} failed.")


def _invoke_worker_runner(
    runner: Callable[..., tuple[int, str]],
    env_record: EnvironmentRecord,
    task_file: Path,
    result_file: Path,
    *,
    log_callback: Callable[[str], None] | None,
) -> tuple[int, str]:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "log_callback" in signature.parameters:
        return runner(
            env_record,
            task_file,
            result_file,
            log_callback=log_callback,
        )
    return runner(env_record, task_file, result_file)


def _build_worker_command_and_env(
    *,
    env_record: EnvironmentRecord,
    model_name: str,
    execution_mode: str,
    module_name: str,
    extra_args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    if execution_mode == "mock":
        command = [
            sys.executable,
            "-m",
            module_name,
            *extra_args,
        ]
        env = os.environ.copy()
    else:
        if env_record.state != "ready":
            raise RunFlowError(
                f"Environment for {model_name} is not ready for real execution: {env_record.state}"
            )
        manager = EnvManager()
        command = manager.wrap_command(
            env_record.env_id,
            ["python", "-m", module_name, *extra_args],
            foreground=False,
        )
        env = manager.build_model_process_env(model_name)

    pythonpath = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{pythonpath}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else pythonpath
    )
    if env_overrides:
        env.update(env_overrides)
    return command, env


def _build_replica_env_overrides(gpu_assignment: list[int]) -> dict[str, str]:
    if not gpu_assignment:
        return {}
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_assignment)}


def _parse_gpu_list(raw_value: str) -> list[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def _resolve_available_gpus() -> list[int]:
    explicit = os.environ.get("AIGC_AVAILABLE_GPUS")
    if explicit is not None:
        return _parse_gpu_list(explicit)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, ""):
        return _parse_gpu_list(visible)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if gpu_lines:
                return [int(line) for line in gpu_lines]
    except Exception:
        pass

    try:
        import torch

        count = int(torch.cuda.device_count())
        return list(range(count))
    except Exception:
        return []


def _plan_replicas_for_model(
    *,
    model: ModelInfo,
    prepared_tasks: list[PreparedTask],
    execution_mode: str,
) -> list[ReplicaPlan]:
    available_gpus = _limit_available_gpus_for_model(_resolve_available_gpus(), model)
    replica_count = _calculate_replica_count(
        total_available_gpus=len(available_gpus),
        gpus_per_replica=model.gpus_per_replica,
        supports_multi_replica=model.supports_multi_replica,
        execution_mode=execution_mode,
        task_count=len(prepared_tasks),
    )
    gpu_assignments = _assign_gpus_to_replicas(
        available_gpus=available_gpus,
        gpus_per_replica=model.gpus_per_replica,
        replica_count=replica_count,
    )
    sharded_tasks = _shard_tasks_across_replicas(
        prepared_tasks=prepared_tasks,
        replica_count=replica_count,
    )

    replica_plans: list[ReplicaPlan] = []
    for replica_id in range(replica_count):
        tasks = sharded_tasks[replica_id]
        gpu_assignment = gpu_assignments[replica_id] if replica_id < len(gpu_assignments) else []
        for prepared_task in tasks:
            _annotate_task_with_replica_context(
                prepared_task=prepared_task,
                replica_id=replica_id,
                gpu_assignment=gpu_assignment,
                replica_count=replica_count,
            )
        replica_plans.append(
            ReplicaPlan(
                replica_id=replica_id,
                gpu_assignment=gpu_assignment,
                tasks=tasks,
            )
        )
    return replica_plans


def _limit_available_gpus_for_model(available_gpus: list[int], model: ModelInfo) -> list[int]:
    if model.max_gpus is None:
        return list(available_gpus)
    return list(available_gpus[: model.max_gpus])


def _calculate_replica_count(
    *,
    total_available_gpus: int,
    gpus_per_replica: int,
    supports_multi_replica: bool,
    execution_mode: str,
    task_count: int,
) -> int:
    if task_count <= 0:
        return 0
    if execution_mode == "mock":
        if total_available_gpus <= 0:
            return 1
    if not supports_multi_replica or total_available_gpus <= 0:
        return 1
    planned = total_available_gpus // max(gpus_per_replica, 1)
    if planned <= 0:
        return 1
    return max(1, min(planned, task_count))


def _assign_gpus_to_replicas(
    *,
    available_gpus: list[int],
    gpus_per_replica: int,
    replica_count: int,
) -> list[list[int]]:
    if replica_count <= 0:
        return []
    if not available_gpus:
        return [[] for _ in range(replica_count)]

    assignments: list[list[int]] = []
    cursor = 0
    for _replica_id in range(replica_count):
        assignment = available_gpus[cursor : cursor + gpus_per_replica]
        if not assignment:
            assignment = list(available_gpus)
        assignments.append(list(assignment))
        cursor += gpus_per_replica
    return assignments


def _shard_tasks_across_replicas(
    *,
    prepared_tasks: list[PreparedTask],
    replica_count: int,
) -> list[list[PreparedTask]]:
    if replica_count <= 0:
        return []
    shards: list[list[PreparedTask]] = [[] for _ in range(replica_count)]
    for task_index, prepared_task in enumerate(prepared_tasks):
        shards[task_index % replica_count].append(prepared_task)
    return shards


def _annotate_task_with_replica_context(
    *,
    prepared_task: PreparedTask,
    replica_id: int,
    gpu_assignment: list[int],
    replica_count: int,
) -> None:
    prepared_task.payload.runtime_config.update(
        {
            "replica_id": replica_id,
            "gpu_assignment": list(gpu_assignment),
            "replica_count": replica_count,
        }
    )
    prepared_task.task_file.write_text(
        json.dumps(prepared_task.payload.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _log_replica_plan(
    *,
    progress: RunProgress,
    model: ModelInfo,
    replica_plans: list[ReplicaPlan],
) -> None:
    available_gpus = _limit_available_gpus_for_model(_resolve_available_gpus(), model)
    progress.env_message(f"[run][{model.name}] available_gpus={available_gpus}")
    if model.max_gpus is not None:
        progress.env_message(f"[run][{model.name}] max_gpus={model.max_gpus}")
    progress.env_message(f"[run][{model.name}] gpus_per_replica={model.gpus_per_replica}")
    progress.env_message(f"[run][{model.name}] starting {len(replica_plans)} replicas")
    for replica_plan in replica_plans:
        progress.env_message(
            f"[run][{model.name}] replica={replica_plan.replica_id} assigned {len(replica_plan.tasks)} tasks GPUs={replica_plan.gpu_assignment}"
        )


def _replica_log_path(workers_root: Path, model_name: str, replica_id: int) -> Path:
    return workers_root / _slugify(model_name) / f"replica_{replica_id}.log"


def _run_persistent_worker_replicas(
    *,
    model: ModelInfo,
    env_record: EnvironmentRecord,
    replica_plans: list[ReplicaPlan],
    execution_mode: str,
    run_id: str,
    run_root: Path,
    workers_root: Path,
    failures: list[dict[str, object]],
    progress: RunProgress,
    ledger_writer: RunLedgerWriter,
    telemetry: RunTelemetry | None,
    failure_policy: FailurePolicy,
    failure_budget: FailureBudget,
    task_results_out: list[tuple[ModelInfo, TaskPayload, dict]] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> list[tuple[ModelInfo, TaskPayload, dict]]:
    lock = threading.Lock()
    task_results = task_results_out if task_results_out is not None else []
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []
    stop_event = threading.Event()
    session_bindings: list[tuple[ReplicaPlan, _PersistentWorkerSession]] = []

    def run_replica(replica_plan: ReplicaPlan) -> None:
        try:
            session = next(
                bound_session
                for bound_plan, bound_session in session_bindings
                if bound_plan.replica_id == replica_plan.replica_id
            )
            for prepared_task in replica_plan.tasks:
                if stop_event.is_set():
                    break
                outcome = _execute_prepared_task(
                    prepared_task=prepared_task,
                    runner=lambda task, session=session: session.run_task(task),
                    run_id=run_id,
                    run_root=run_root,
                    failures=failures,
                    progress=progress,
                    ledger_writer=ledger_writer,
                    telemetry=telemetry,
                    state_lock=lock,
                )
                _apply_failure_policy_after_task(
                    outcome=outcome,
                    prepared_task=prepared_task,
                    failure_policy=failure_policy,
                    failure_budget=failure_budget,
                    progress=progress,
                    model_name=model.name,
                )
                with lock:
                    task_results.append((model, prepared_task.payload, outcome.result_payload))
        except BaseException as exc:  # pragma: no cover - exercised via thread/integration path
            with lock:
                errors.append(exc)
                stop_event.set()

    with contextlib.ExitStack() as stack:
        total_replicas = len(replica_plans)
        for warmup_index, replica_plan in enumerate(replica_plans, start=1):
            progress.env_message(
                f"[run][{model.name}] warming replica {warmup_index}/{total_replicas} "
                f"replica={replica_plan.replica_id} GPUs={replica_plan.gpu_assignment}"
            )
            session = stack.enter_context(
                _PersistentWorkerSession(
                    model=model,
                    env_record=env_record,
                    execution_mode=execution_mode,
                    replica_id=replica_plan.replica_id,
                    gpu_assignment=replica_plan.gpu_assignment,
                    replica_log_path=_replica_log_path(workers_root, model.name, replica_plan.replica_id),
                    log_callback=log_callback,
                )
            )
            session_bindings.append((replica_plan, session))
            progress.env_message(
                f"[run][{model.name}] replica={replica_plan.replica_id} ready "
                f"({warmup_index}/{total_replicas}) GPUs={replica_plan.gpu_assignment}"
            )

        progress.env_message(
            f"[run][{model.name}] all replicas ready, dispatching tasks"
        )

        for replica_plan, _session in session_bindings:
            thread = threading.Thread(
                target=run_replica,
                args=(replica_plan,),
                name=f"{_slugify(model.name)}-replica-{replica_plan.replica_id}",
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if errors:
            raise errors[0]
        return task_results


def _append_ledger_records(
    *,
    ledger_writer: RunLedgerWriter,
    run_id: str,
    model: ModelInfo,
    task_payload: TaskPayload,
    task_result: dict | None,
    error_message: str | None = None,
    failure_category: str | None = None,
    state_lock: threading.Lock | None = None,
) -> None:
    records = build_sample_ledger_records(
        run_id=run_id,
        model=model,
        task_payload=task_payload,
        task_result=task_result,
        error_message=error_message,
        failure_category=failure_category,
    )
    if state_lock is not None:
        with state_lock:
            ledger_writer.append_records(records)
    else:
        ledger_writer.append_records(records)


def _merge_diagnostic_logs(*parts: str | None) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = (part or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        merged.append(clean)
    return "\n\n".join(merged)


def _prompt_to_task_prompt(prompt: PromptRecord) -> TaskPrompt:
    return TaskPrompt(
        prompt_id=prompt.prompt_id,
        prompt=prompt.prompt,
        language=prompt.language,
        negative_prompt=prompt.negative_prompt,
        metadata=prompt.metadata,
    )


def _default_generation_params(
    model: ModelInfo,
    prompts: list[PromptRecord],
    *,
    generation_defaults: dict[str, object] | None = None,
) -> dict[str, object]:
    if not prompts:
        raise RunFlowError(f"Cannot build generation params for {model.name} with no prompts.")

    params = _model_default_generation_params(model)
    params = _merge_generation_param_overrides(params, generation_defaults)
    params = _merge_generation_param_overrides(params, prompts[0].parameters)

    if model.capabilities.get("supports_negative_prompt"):
        params["negative_prompts"] = [batch_prompt.negative_prompt or "" for batch_prompt in prompts]
    return params


def _model_default_generation_params(model: ModelInfo) -> dict[str, object]:
    default_seed = get_default_seed()
    if model.modality == "image":
        params: dict[str, object] = {
            "width": 1024,
            "height": 1024,
        }
    elif model.modality == "video":
        params = {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "num_frames": 121,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
        }
    else:
        params = {}

    if default_seed is not None:
        params["seed"] = default_seed

    if model.name == "Z-Image":
        params.update(
            {
                "cfg_normalization": False,
                "num_inference_steps": 50,
                "guidance_scale": 4.0,
            }
        )
    elif model.name == "Z-Image-Turbo":
        params.update({"num_inference_steps": 9, "guidance_scale": 0.0})
    elif model.name == "FLUX.1-dev":
        params.update(
            {
                "num_inference_steps": 50,
                "guidance_scale": 3.5,
                "max_sequence_length": 512,
            }
        )
    elif model.name == "stable-diffusion-xl-base-1.0":
        params.update({"num_inference_steps": 40, "guidance_scale": 5.0})
    elif model.name == "Qwen-Image-2512":
        params.update({"num_inference_steps": 50, "guidance_scale": 4.0})
    elif model.name == "HunyuanImage-3.0":
        params.update(
            {
                "num_inference_steps": 50,
                "guidance_scale": 1.0,
                "stream": True,
                "attn_implementation": "sdpa",
                "moe_impl": "eager",
            }
        )
    elif model.name == "Wan2.2-T2V-A14B-Diffusers":
        params.update(
            {
                "width": 1280,
                "height": 720,
                "fps": 16,
                "num_frames": 81,
                "num_inference_steps": 40,
                "guidance_scale": 4.0,
                "guidance_scale_2": 3.0,
            }
        )
    elif model.name == "CogVideoX-5B":
        params.update(
            {
                "width": 720,
                "height": 480,
                "fps": 8,
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "local_model_path": str(
                    model.weights.get("weights_path") or model.weights.get("local_path")
                )
                if (model.weights.get("weights_path") or model.weights.get("local_path"))
                else None,
            }
        )
    elif model.name == "Wan2.2-TI2V-5B":
        params.update(
            {
                "width": 1280,
                "height": 704,
                "fps": 24,
                "num_frames": 121,
                "num_inference_steps": 40,
                "guidance_scale": 4.0,
                "checkpoint_dir": str(
                    model.weights.get("weights_path")
                    or model.weights.get("local_path")
                    or "./Wan2.2-TI2V-5B"
                ),
            }
        )
    elif model.name == "LongCat-Video":
        params.update(
            {
                "width": 1280,
                "height": 720,
                "fps": 30,
                "num_frames": 121,
                "num_inference_steps": 50,
                "checkpoint_dir": str(
                    model.weights.get("weights_path")
                    or model.weights.get("local_path")
                    or "./weights/LongCat-Video"
                ),
            }
        )
    elif model.name == "HunyuanVideo-1.5":
        params.update(
            {
                "width": 1280,
                "height": 720,
                "fps": 24,
                "num_frames": 121,
                "num_inference_steps": 50,
                "guidance_scale": 4.0,
            }
        )
    elif model.name == "MOVA-720p":
        params.update(
            {
                "width": 1280,
                "height": 720,
                "fps": 24,
                "num_frames": 193,
                "num_inference_steps": 25,
                "guidance_scale": 5.0,
                "checkpoint_dir": str(
                    model.weights.get("weights_path")
                    or model.weights.get("local_path")
                    or "/path/to/MOVA-720p"
                ),
                "offload": "cpu",
                "cp_size": 1,
            }
        )

    return params


def _merge_generation_param_overrides(
    base_params: dict[str, object],
    overrides: dict[str, object] | None,
) -> dict[str, object]:
    params = dict(base_params)
    if not overrides:
        return params

    if "resolution" in overrides:
        width, height = _parse_resolution_value(str(overrides["resolution"]))
        params["width"] = width
        params["height"] = height

    for key, value in overrides.items():
        if key == "resolution":
            continue
        params[key] = value
    return params


def _parse_resolution_value(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("*", "x")
    width, height = re.split(r"\s*x\s*", normalized, maxsplit=1)
    return int(width), int(height)


def _prepare_recovery_tasks(
    *,
    models: list[ModelInfo],
    recovery_plan: RecoveryPlan,
    tasks_dir: Path,
    workdir_root: Path,
    artifacts_root: Path,
    batch_limit: int | None,
    worker_runner: Callable[[EnvironmentRecord, Path, Path], tuple[int, str]] | None,
    registry,
) -> tuple[dict[str, list[PreparedTask]], dict[str, str]]:
    prepared_tasks_by_model: dict[str, list[PreparedTask]] = {}
    worker_strategies: dict[str, str] = {}
    task_counter = 1

    for model in models:
        strategy = _resolve_worker_strategy(
            registry=registry,
            model=model,
            worker_runner=worker_runner,
        )
        worker_strategies[model.name] = strategy
        model_slug = _slugify(model.name)
        model_tasks_dir = tasks_dir / model_slug
        model_workdir_root = workdir_root / model_slug
        model_artifacts_root = artifacts_root / model_slug
        for directory in (model_tasks_dir, model_workdir_root, model_artifacts_root):
            directory.mkdir(parents=True, exist_ok=True)

        recovery_items = recovery_plan.items_by_model.get(model.name, [])
        grouped_items = _group_recovery_items_by_params(recovery_items)
        grouped_batches: list[tuple[dict[str, object], list[PromptRecord]]] = []
        for params_signature, items in grouped_items:
            del params_signature
            prompts = [item.prompt for item in items]
            for prompt_batch in _batch_prompts_for_model(
                model=model,
                prompts=prompts,
                batch_limit=batch_limit,
            ):
                batch_params = dict(items[0].params)
                if model.capabilities.get("supports_negative_prompt"):
                    batch_params["negative_prompts"] = [
                        batch_prompt.negative_prompt or "" for batch_prompt in prompt_batch
                    ]
                grouped_batches.append((batch_params, prompt_batch))

        total_tasks_for_model = len(grouped_batches)
        model_prepared_tasks: list[PreparedTask] = []
        for batch_number, (batch_params, prompt_batch) in enumerate(grouped_batches, start=1):
            task_id = f"task_{task_counter:06d}"
            batch_id = f"{model_slug}_batch_{batch_number:06d}"
            task_workdir = model_workdir_root / task_id
            payload = TaskPayload(
                task_id=task_id,
                model_name=model.name,
                execution_mode=recovery_plan.execution_mode,
                prompts=[_prompt_to_task_prompt(prompt) for prompt in prompt_batch],
                params=dict(batch_params),
                workdir=str(task_workdir),
                batch_id=batch_id,
                runtime_config={"worker_strategy": strategy},
                worker_strategy=strategy,
            )
            task_file = model_tasks_dir / f"{task_id}.json"
            result_file = model_tasks_dir / f"{task_id}.result.json"
            task_file.write_text(
                json.dumps(payload.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            model_prepared_tasks.append(
                PreparedTask(
                    model=model,
                    payload=payload,
                    task_file=task_file,
                    result_file=result_file,
                    batch_number=batch_number,
                    total_tasks_for_model=total_tasks_for_model,
                )
            )
            task_counter += 1

        prepared_tasks_by_model[model.name] = model_prepared_tasks

    return prepared_tasks_by_model, worker_strategies


def _group_recovery_items_by_params(
    recovery_items: list[RecoveryItem],
) -> list[tuple[str, list[RecoveryItem]]]:
    grouped: dict[str, list[RecoveryItem]] = {}
    for item in recovery_items:
        signature = json.dumps(item.params, sort_keys=True, ensure_ascii=False)
        grouped.setdefault(signature, []).append(item)
    return list(grouped.items())


def _generate_run_id(run_name: str | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if not run_name:
        return f"run_{timestamp}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name).strip("-").lower() or "run"
    return f"{slug}_{timestamp}"


def _resolve_environment_record(
    *,
    manager: EnvManager,
    model_name: str,
    execution_mode: str,
    progress: callable | None = None,
) -> EnvironmentRecord:
    if execution_mode == "mock" and hasattr(manager, "inspect_model_environment"):
        return manager.inspect_model_environment(model_name)
    if hasattr(manager, "ensure_ready"):
        return manager.ensure_ready(model_name, foreground=True, progress=progress)
    return manager.ensure_environment(model_name)


def _resolve_worker_strategy(
    *,
    registry,
    model: ModelInfo,
    worker_runner: Callable[[EnvironmentRecord, Path, Path], tuple[int, str]] | None,
) -> str:
    if model.execution_mode != "in_process":
        return "per_task_worker"
    if worker_runner is not None:
        return "per_task_worker"
    if model.worker_strategy in {"per_task_worker", "persistent_worker"}:
        return model.worker_strategy
    adapter_class = registry.resolve_adapter_class(model.name)
    capabilities = getattr(adapter_class, "capabilities", None)
    if capabilities is None:
        return "per_task_worker"
    if getattr(capabilities, "supports_persistent_worker", False):
        return str(getattr(capabilities, "preferred_worker_strategy", "persistent_worker"))
    return "per_task_worker"


def _resolve_execution_mode(*, execution_mode: str, mock_mode: bool | None) -> str:
    if mock_mode is True:
        return "mock"
    if execution_mode not in {"mock", "real"}:
        raise RunFlowError(
            f"Unsupported execution mode: {execution_mode}. Expected one of: mock, real."
        )
    return execution_mode


def _batch_prompts_for_model(
    *,
    model: ModelInfo,
    prompts: list[PromptRecord],
    batch_limit: int | None,
    generation_defaults: dict[str, object] | None = None,
) -> list[list[PromptRecord]]:
    batch_size = _resolve_batch_size(model=model, batch_limit=batch_limit)
    batches: list[list[PromptRecord]] = []
    current_batch: list[PromptRecord] = []
    current_signature: str | None = None

    for prompt in prompts:
        signature = _prompt_batch_signature(
            model=model,
            prompt=prompt,
            generation_defaults=generation_defaults,
        )
        if current_batch and (len(current_batch) >= batch_size or signature != current_signature):
            batches.append(current_batch)
            current_batch = []
            current_signature = None
        if not current_batch:
            current_signature = signature
        current_batch.append(prompt)

    if current_batch:
        batches.append(current_batch)
    return batches


def _resolve_batch_size(*, model: ModelInfo, batch_limit: int | None) -> int:
    if not model.capabilities.get("supports_batch_prompts"):
        return 1
    max_batch_size = int(model.capabilities.get("max_batch_size", 1))
    preferred_batch_size = int(model.capabilities.get("preferred_batch_size", 1))
    batch_size = min(max_batch_size, preferred_batch_size)
    if batch_limit is not None:
        batch_size = min(batch_size, batch_limit)
    return max(batch_size, 1)


def _prompt_batch_signature(
    *,
    model: ModelInfo,
    prompt: PromptRecord,
    generation_defaults: dict[str, object] | None = None,
) -> str:
    params = _default_generation_params(model, [prompt], generation_defaults=generation_defaults)
    params.pop("negative_prompts", None)
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug or "model"
