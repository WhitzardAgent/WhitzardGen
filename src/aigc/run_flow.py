from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
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
from aigc.utils.progress import (
    RunProgress,
    RunSummaryData,
    NullRunProgress,
    summarize_task_statuses,
)

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
    progress = progress or NullRunProgress()

    total_stages = 9
    stage_index = 1

    progress.stage_start(stage_index, total_stages, "Loading prompts")
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

    progress.stage_end(stage_index, total_stages, "Loading prompts")
    stage_index += 1

    progress.stage_start(stage_index, total_stages, "Validating prompts")
    prompts = load_prompts(prompt_file)
    if not prompts:
        raise RunFlowError("Prompt file did not produce any valid prompts.")
    progress.stage_end(stage_index, total_stages, "Validating prompts")
    stage_index += 1

    progress.stage_start(stage_index, total_stages, "Resolving models")
    run_id = _generate_run_id(run_name)
    run_root = Path(out_dir) if out_dir else REPO_ROOT / "runs" / run_id
    tasks_dir = run_root / "tasks"
    workdir_root = run_root / "workdir"
    exports_dir = run_root / "exports"
    artifacts_root = run_root / "artifacts"
    for directory in (tasks_dir, workdir_root, exports_dir, artifacts_root):
        directory.mkdir(parents=True, exist_ok=True)

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
        "registry_path": str(registry.registry_path),
        "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
    }
    write_run_manifest(run_root, initial_manifest)
    failures: list[dict[str, object]] = []
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
        task_results: list[tuple[ModelInfo, TaskPayload, dict]] = []
        runner = worker_runner or _run_worker_task
        task_counter = 1
        for model in models:
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
                    runtime_config={},
                )
                task_file = model_tasks_dir / f"{task_id}.json"
                result_file = model_tasks_dir / f"{task_id}.result.json"
                task_file.write_text(
                    json.dumps(payload.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                if resolved_execution_mode == "mock":
                    execution_label = "mock"
                else:
                    execution_label = "real"

                progress.task_start(
                    current=batch_number,
                    total=total_tasks_for_model,
                    model_name=model.name,
                    prompts=len(prompt_batch),
                    execution_mode=execution_label,
                )

                returncode, logs = runner(env_records[model.name], task_file, result_file)
                if not result_file.exists():
                    failures.append(
                        {
                            "task_id": task_id,
                            "model_name": model.name,
                            "status": "failed",
                            "error": "Worker did not produce a result file.",
                        }
                    )
                    write_failures_summary(run_root, failures)
                    progress.task_end(
                        current=batch_number,
                        total=total_tasks_for_model,
                        model_name=model.name,
                        status="failed",
                        artifacts=None,
                    )
                    raise RunFlowError(f"Worker did not produce result file for {task_id}.")

                result_payload = json.loads(result_file.read_text(encoding="utf-8"))
                execution_logs = _merge_diagnostic_logs(
                    result_payload.get("execution_result", {}).get("logs"),
                    result_payload.get("model_result", {}).get("logs"),
                    logs,
                )
                model_status = result_payload["model_result"]["status"]
                if returncode != 0 and model_status == "failed":
                    failures.append(
                        {
                            "task_id": task_id,
                            "model_name": model.name,
                            "status": "failed",
                            "error": execution_logs,
                        }
                    )
                    write_failures_summary(run_root, failures)
                    progress.task_end(
                        current=batch_number,
                        total=total_tasks_for_model,
                        model_name=model.name,
                        status="failed",
                        artifacts=None,
                    )
                    raise RunFlowError(f"Worker failed for {task_id}:\n{execution_logs}")
                if model_status == "failed":
                    failures.append(
                        {
                            "task_id": task_id,
                            "model_name": model.name,
                            "status": "failed",
                            "error": execution_logs,
                        }
                    )
                    write_failures_summary(run_root, failures)
                    progress.task_end(
                        current=batch_number,
                        total=total_tasks_for_model,
                        model_name=model.name,
                        status="failed",
                        artifacts=None,
                    )
                    raise RunFlowError(f"Task {task_id} failed:\n{execution_logs}")

                successful_artifacts = 0
                for batch_item in result_payload.get("model_result", {}).get("batch_items", []):
                    if batch_item.get("status") != "success":
                        continue
                    successful_artifacts += len(batch_item.get("artifacts", []))

                progress.task_end(
                    current=batch_number,
                    total=total_tasks_for_model,
                    model_name=model.name,
                    status=model_status,
                    artifacts=successful_artifacts or None,
                )
                task_results.append((model, payload, result_payload))
                task_counter += 1

        progress.stage_end(stage_index, total_stages, "Preparing tasks")
        stage_index += 1

        progress.stage_start(stage_index, total_stages, "Running tasks")
        # Task execution has already happened inline above; the stage marker
        # exists to keep the visible stage sequence aligned with the spec.
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
            "export_paths": {"dataset_jsonl": str(export_path)},
            "export_path": str(export_path),
            "per_model_summary": per_model_summary,
            "registry_path": str(registry.registry_path),
            "local_models_path": str(registry.local_models_path) if registry.local_models_path else None,
        }
        (run_root / "run.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        manifest_path = write_run_manifest(run_root, run_manifest)
        write_failures_summary(run_root, failures)
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
            )
        )

        progress.stage_end(stage_index, total_stages, "Done")
    except Exception as exc:
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
        }
        write_run_manifest(run_root, failure_manifest)
        write_failures_summary(run_root, failures)
        raise

    return RunSummary(
        run_id=run_id,
        model_names=[model.name for model in models],
        prompt_file=str(prompt_file),
        output_dir=str(run_root),
        tasks_scheduled=len(task_results),
        records_exported=len(all_records),
        export_path=str(export_path),
        execution_mode=resolved_execution_mode,
    )


def _run_worker_task(env_record: EnvironmentRecord, task_file: Path, result_file: Path) -> tuple[int, str]:
    payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
    worker_log_file = task_file.with_suffix(".worker.log")
    if payload.execution_mode == "mock":
        command = [
            sys.executable,
            "-m",
            "aigc.runtime.worker",
            "--task-file",
            str(task_file),
            "--result-file",
            str(result_file),
        ]
    else:
        if env_record.state != "ready":
            raise RunFlowError(
                f"Environment for {payload.model_name} is not ready for real execution: {env_record.state}"
            )
        manager = EnvManager()
        command = manager.wrap_command(
            env_record.env_id,
            [
                "python",
                "-m",
                "aigc.runtime.worker",
                "--task-file",
                str(task_file),
                "--result-file",
                str(result_file),
            ],
            foreground=False,
        )
        env = manager.build_model_process_env(payload.model_name)
    if payload.execution_mode == "mock":
        env = os.environ.copy()
    pythonpath = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{pythonpath}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else pythonpath
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    logs = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    if result.returncode != 0:
        diagnostics = [
            f"Worker command exit code: {result.returncode}",
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
    return result.returncode, logs


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
