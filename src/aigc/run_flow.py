from __future__ import annotations

import json
import inspect
import os
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from aigc.env import EnvManager, EnvManagerError, EnvironmentRecord
from aigc.exporters import build_dataset_records, export_jsonl
from aigc.prompts import PromptRecord, load_prompts
from aigc.registry.models import ModelInfo
from aigc.registry import load_registry
from aigc.run_store import write_failures_summary, write_run_manifest
from aigc.runtime.payloads import TaskPayload, TaskPrompt
from aigc.settings import get_runs_root
from aigc.utils.progress import (
    LoggedRunProgress,
    RunProgress,
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
    run_id: str
    model_names: list[str]
    prompt_file: str
    output_dir: str
    tasks_scheduled: int
    records_exported: int
    export_path: str
    execution_mode: str = "real"
    running_log_path: str | None = None

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
    for directory in (run_root, tasks_dir, workdir_root, exports_dir, artifacts_root):
        directory.mkdir(parents=True, exist_ok=True)
    base_progress = progress or NullRunProgress()
    console_logging_enabled = not isinstance(base_progress, NullRunProgress)
    run_logger = RunLogger(log_path=run_root / "running.log")
    progress = LoggedRunProgress(base=base_progress, logger=run_logger)

    total_stages = 9
    stage_index = 1
    progress.env_message(
        f"[run] Starting run run_id={run_id} mode={resolved_execution_mode} models={','.join(model.name for model in models)}"
    )
    progress.env_message(f"[run] Output dir: {run_root}")
    progress.env_message(f"[run] Running log: {run_root / 'running.log'}")

    progress.stage_start(stage_index, total_stages, "Loading prompts")
    prompts = load_prompts(prompt_file)
    progress.stage_end(stage_index, total_stages, "Loading prompts")
    stage_index += 1

    progress.stage_start(stage_index, total_stages, "Validating prompts")
    if not prompts:
        raise RunFlowError("Prompt file did not produce any valid prompts.")
    progress.stage_end(stage_index, total_stages, "Validating prompts")
    stage_index += 1

    progress.stage_start(stage_index, total_stages, "Resolving models")

    manager = env_manager or EnvManager(registry=registry)
    progress.stage_end(stage_index, total_stages, "Resolving models")
    stage_index += 1
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
        "export_paths": {},
        "running_log_path": str(run_logger.log_path),
        "registry_path": str(registry.registry_path),
        "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
    }
    write_run_manifest(run_root, initial_manifest)
    failures: list[dict[str, object]] = []
    task_results: list[tuple[ModelInfo, TaskPayload, dict]] = []
    export_path: Path | None = None
    failures_path = run_root / "failures.json"
    manifest_path = run_root / "run_manifest.json"
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
                    params=_default_generation_params(model, prompt_batch),
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
                _log_replica_plan(progress=progress, model=model, replica_plans=replica_plans)
                if len(replica_plans) == 1:
                    replica_plan = replica_plans[0]
                    with _PersistentWorkerSession(
                        model=model,
                        env_record=env_records[model.name],
                        execution_mode=resolved_execution_mode,
                        replica_id=replica_plan.replica_id,
                        gpu_assignment=replica_plan.gpu_assignment,
                        log_callback=lambda line: run_logger.log(
                            line,
                            to_console=console_logging_enabled,
                            already_timestamped=True,
                        ),
                    ) as session:
                        for prepared_task in replica_plan.tasks:
                            result_payload = _execute_prepared_task(
                                prepared_task=prepared_task,
                                runner=lambda task, session=session: session.run_task(task),
                                run_root=run_root,
                                failures=failures,
                                progress=progress,
                            )
                            task_results.append((model, prepared_task.payload, result_payload))
                else:
                    task_results.extend(
                        _run_persistent_worker_replicas(
                            model=model,
                            env_record=env_records[model.name],
                            replica_plans=replica_plans,
                            execution_mode=resolved_execution_mode,
                            run_root=run_root,
                            failures=failures,
                            progress=progress,
                            log_callback=lambda line: run_logger.log(
                                line,
                                to_console=console_logging_enabled,
                                already_timestamped=True,
                            ),
                        )
                    )
            else:
                runner = worker_runner or _run_worker_task
                for prepared_task in prepared_tasks:
                    result_payload = _execute_prepared_task(
                        prepared_task=prepared_task,
                        runner=lambda task, env_record=env_records[model.name], runner=runner: _invoke_worker_runner(
                            runner,
                            env_record,
                            task.task_file,
                            task.result_file,
                            log_callback=lambda line: run_logger.log(
                                line,
                                to_console=console_logging_enabled,
                                already_timestamped=True,
                            ),
                        ),
                        run_root=run_root,
                        failures=failures,
                        progress=progress,
                    )
                    task_results.append((model, prepared_task.payload, result_payload))
                if prepared_tasks:
                    replica_plans_by_model[model.name] = [
                        ReplicaPlan(replica_id=0, gpu_assignment=[], tasks=list(prepared_tasks))
                    ]
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
        per_model_summary: dict[str, dict[str, object]] = {}
        for model in models:
            model_records = [record for record in all_records if record["model_name"] == model.name]
            per_model_summary[model.name] = {
                "record_count": len(model_records),
                "task_count": len([result for result in task_results if result[0].name == model.name]),
                "modality": model.modality,
                "task_type": model.task_type,
                "worker_strategy": worker_strategies.get(model.name, "per_task_worker"),
                "replica_count": len(replica_plans_by_model.get(model.name, [])) or 1,
                "replicas": [
                    {
                        "replica_id": replica_plan.replica_id,
                        "gpu_assignment": replica_plan.gpu_assignment,
                        "task_count": len(replica_plan.tasks),
                    }
                    for replica_plan in replica_plans_by_model.get(model.name, [])
                ],
            }

        run_manifest = {
            "run_id": run_id,
            "status": "completed",
            "created_at": created_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "models": [model.name for model in models],
            "prompt_source": str(prompt_file),
            "prompt_count": len(prompts),
            "execution_mode": resolved_execution_mode,
            "task_count": len(task_results),
            "records_exported": len(all_records),
            "output_dir": str(run_root),
            "export_paths": {"dataset_jsonl": str(export_path), "running_log": str(run_logger.log_path)},
            "export_path": str(export_path),
            "running_log_path": str(run_logger.log_path),
            "failures_path": str(failures_path),
            "per_model_summary": per_model_summary,
            "registry_path": str(registry.registry_path),
            "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
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
                status="completed",
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
            )
        )

        progress.stage_end(stage_index, total_stages, "Done")
    except Exception as exc:
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
            "running_log_path": str(run_logger.log_path),
            "failures_path": str(failures_path),
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
            )
        )
        raise
    finally:
        run_logger.close()

    return RunSummary(
        run_id=run_id,
        model_names=[model.name for model in models],
        prompt_file=str(prompt_file),
        output_dir=str(run_root),
        tasks_scheduled=len(task_results),
        records_exported=len(all_records),
        export_path=str(export_path),
        execution_mode=resolved_execution_mode,
        running_log_path=str(run_logger.log_path),
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
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.env_record = env_record
        self.execution_mode = execution_mode
        self.replica_id = replica_id
        self.gpu_assignment = list(gpu_assignment or [])
        self.log_callback = log_callback
        self.process: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> "_PersistentWorkerSession":
        command, env = _build_worker_command_and_env(
            env_record=self.env_record,
            model_name=self.model.name,
            execution_mode=self.execution_mode,
            module_name="aigc.runtime.persistent_worker",
            extra_args=[
                "--model-name",
                self.model.name,
                "--execution-mode",
                self.execution_mode,
                "--replica-id",
                str(self.replica_id),
                "--gpu-assignment",
                ",".join(str(gpu_id) for gpu_id in self.gpu_assignment),
            ],
            env_overrides=_build_replica_env_overrides(self.gpu_assignment),
        )
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_stderr_pump()
        event = self._read_event()
        if event.get("event") != "ready":
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} did not become ready: {event}"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_task(self, prepared_task: PreparedTask) -> tuple[int, str]:
        self._write_command(
            {
                "command": "run_task",
                "task_file": str(prepared_task.task_file),
                "result_file": str(prepared_task.result_file),
            }
        )
        event = self._read_event()
        if event.get("event") != "task_complete":
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} returned unexpected event: {event}"
            )
        status = str(event.get("status", "failed"))
        error = str(event.get("error", "")).strip()
        return (0 if status != "failed" else 1), error

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self._write_command({"command": "shutdown"})
                self._read_event(allow_eof=True)
                self.process.stdin.close()
        finally:
            returncode = self.process.wait(timeout=30)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=5)
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()
            self.process = None
            if returncode != 0:
                raise RunFlowError(
                    f"Persistent worker for {self.model.name} replica={self.replica_id} exited with code {returncode}."
                )

    def _write_command(self, payload: dict[str, object]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} is not writable."
            )
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _read_event(self, *, allow_eof: bool = False) -> dict[str, object]:
        if self.process is None or self.process.stdout is None:
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} is not readable."
            )
        line = self.process.stdout.readline()
        if not line:
            if allow_eof:
                return {}
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} exited before emitting the next event."
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunFlowError(
                f"Persistent worker for {self.model.name} replica={self.replica_id} emitted invalid JSON: {line!r}"
            ) from exc

    def _start_stderr_pump(self) -> None:
        if self.process is None or self.process.stderr is None:
            return

        def pump() -> None:
            assert self.process is not None and self.process.stderr is not None
            for raw_line in self.process.stderr:
                text = raw_line.rstrip("\n")
                if text and self.log_callback is not None:
                    self.log_callback(text)

        self._stderr_thread = threading.Thread(
            target=pump,
            name=f"{_slugify(self.model.name)}-replica-{self.replica_id}-stderr",
            daemon=True,
        )
        self._stderr_thread.start()


def _execute_prepared_task(
    *,
    prepared_task: PreparedTask,
    runner: Callable[[PreparedTask], tuple[int, str]],
    run_root: Path,
    failures: list[dict[str, object]],
    progress: RunProgress,
    state_lock: threading.Lock | None = None,
) -> dict:
    progress.task_start(
        current=prepared_task.batch_number,
        total=prepared_task.total_tasks_for_model,
        model_name=prepared_task.model.name,
        prompts=len(prepared_task.payload.prompts),
        execution_mode=prepared_task.payload.execution_mode,
    )

    returncode, logs = runner(prepared_task)
    if not prepared_task.result_file.exists():
        failure_record = {
            "task_id": prepared_task.payload.task_id,
            "model_name": prepared_task.model.name,
            "status": "failed",
            "error": "Worker did not produce a result file.",
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
        raise RunFlowError(f"Worker did not produce result file for {prepared_task.payload.task_id}.")

    result_payload = json.loads(prepared_task.result_file.read_text(encoding="utf-8"))
    execution_logs = _merge_diagnostic_logs(
        result_payload.get("execution_result", {}).get("logs"),
        result_payload.get("model_result", {}).get("logs"),
        logs,
    )
    model_status = result_payload["model_result"]["status"]
    if returncode != 0 and model_status == "failed":
        failure_record = {
            "task_id": prepared_task.payload.task_id,
            "model_name": prepared_task.model.name,
            "status": "failed",
            "error": execution_logs,
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
        raise RunFlowError(f"Worker failed for {prepared_task.payload.task_id}:\n{execution_logs}")
    if model_status == "failed":
        failure_record = {
            "task_id": prepared_task.payload.task_id,
            "model_name": prepared_task.model.name,
            "status": "failed",
            "error": execution_logs,
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
        raise RunFlowError(f"Task {prepared_task.payload.task_id} failed:\n{execution_logs}")

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
    return result_payload


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


def _run_persistent_worker_replicas(
    *,
    model: ModelInfo,
    env_record: EnvironmentRecord,
    replica_plans: list[ReplicaPlan],
    execution_mode: str,
    run_root: Path,
    failures: list[dict[str, object]],
    progress: RunProgress,
    log_callback: Callable[[str], None] | None = None,
) -> list[tuple[ModelInfo, TaskPayload, dict]]:
    lock = threading.Lock()
    task_results: list[tuple[ModelInfo, TaskPayload, dict]] = []
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def run_replica(replica_plan: ReplicaPlan) -> None:
        try:
            with _PersistentWorkerSession(
                model=model,
                env_record=env_record,
                execution_mode=execution_mode,
                replica_id=replica_plan.replica_id,
                gpu_assignment=replica_plan.gpu_assignment,
                log_callback=log_callback,
            ) as session:
                for prepared_task in replica_plan.tasks:
                    result_payload = _execute_prepared_task(
                        prepared_task=prepared_task,
                        runner=lambda task, session=session: session.run_task(task),
                        run_root=run_root,
                        failures=failures,
                        progress=progress,
                        state_lock=lock,
                    )
                    with lock:
                        task_results.append((model, prepared_task.payload, result_payload))
        except BaseException as exc:  # pragma: no cover - exercised via thread/integration path
            with lock:
                errors.append(exc)

    for replica_plan in replica_plans:
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


def _default_generation_params(model: ModelInfo, prompts: list[PromptRecord]) -> dict[str, object]:
    if not prompts:
        raise RunFlowError(f"Cannot build generation params for {model.name} with no prompts.")

    prompt = prompts[0]
    if model.modality == "image":
        params: dict[str, object] = {
            "width": 1024,
            "height": 1024,
            "seed": 42,
        }
    elif model.modality == "video":
        params = {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "num_frames": 121,
            "seed": 42,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
        }
    else:
        params = {"seed": 42}

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
                "max_gpus": model.max_gpus,
                "repo_dir": str(model.weights["repo_path"]) if model.weights.get("repo_path") else None,
                "local_model_path": str(
                    model.weights.get("weights_path") or model.weights.get("local_path")
                )
                if (model.weights.get("weights_path") or model.weights.get("local_path"))
                else None,
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

    if model.capabilities.get("supports_negative_prompt"):
        params["negative_prompts"] = [batch_prompt.negative_prompt or "" for batch_prompt in prompts]

    resolution = prompt.parameters.get("resolution")
    if isinstance(resolution, str) and "x" in resolution:
        width, height = resolution.lower().split("x", maxsplit=1)
        params["width"] = int(width)
        params["height"] = int(height)
    if "width" in prompt.parameters:
        params["width"] = int(prompt.parameters["width"])
    if "height" in prompt.parameters:
        params["height"] = int(prompt.parameters["height"])
    if "guidance_scale" in prompt.parameters:
        params["guidance_scale"] = float(prompt.parameters["guidance_scale"])
    if "num_inference_steps" in prompt.parameters:
        params["num_inference_steps"] = int(prompt.parameters["num_inference_steps"])
    if "seed" in prompt.parameters:
        params["seed"] = int(prompt.parameters["seed"])
    if "max_sequence_length" in prompt.parameters:
        params["max_sequence_length"] = int(prompt.parameters["max_sequence_length"])
    if "fps" in prompt.parameters:
        params["fps"] = int(prompt.parameters["fps"])
    if "num_frames" in prompt.parameters:
        params["num_frames"] = int(prompt.parameters["num_frames"])
    if "guidance_scale_2" in prompt.parameters:
        params["guidance_scale_2"] = float(prompt.parameters["guidance_scale_2"])
    if "stream" in prompt.parameters:
        params["stream"] = bool(prompt.parameters["stream"])
    if "attn_implementation" in prompt.parameters:
        params["attn_implementation"] = str(prompt.parameters["attn_implementation"])
    if "moe_impl" in prompt.parameters:
        params["moe_impl"] = str(prompt.parameters["moe_impl"])
    if "local_model_path" in prompt.parameters:
        params["local_model_path"] = str(prompt.parameters["local_model_path"])
    if "checkpoint_dir" in prompt.parameters:
        params["checkpoint_dir"] = str(prompt.parameters["checkpoint_dir"])
    if "repo_dir" in prompt.parameters:
        params["repo_dir"] = str(prompt.parameters["repo_dir"])
    if "offload" in prompt.parameters:
        params["offload"] = str(prompt.parameters["offload"])
    if "cp_size" in prompt.parameters:
        params["cp_size"] = int(prompt.parameters["cp_size"])
    if "context_parallel_size" in prompt.parameters:
        params["context_parallel_size"] = int(prompt.parameters["context_parallel_size"])
    if "image_path" in prompt.parameters:
        params["image_path"] = str(prompt.parameters["image_path"])
    if "ref_path" in prompt.parameters:
        params["ref_path"] = str(prompt.parameters["ref_path"])
    if "timeout_sec" in prompt.parameters:
        params["timeout_sec"] = int(prompt.parameters["timeout_sec"])
    return params


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
) -> list[list[PromptRecord]]:
    batch_size = _resolve_batch_size(model=model, batch_limit=batch_limit)
    batches: list[list[PromptRecord]] = []
    current_batch: list[PromptRecord] = []
    current_signature: str | None = None

    for prompt in prompts:
        signature = _prompt_batch_signature(model=model, prompt=prompt)
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


def _prompt_batch_signature(*, model: ModelInfo, prompt: PromptRecord) -> str:
    params = _default_generation_params(model, [prompt])
    params.pop("negative_prompts", None)
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug or "model"
