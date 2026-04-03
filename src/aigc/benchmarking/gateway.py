from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc.benchmarking.interfaces import RunEngineGateway
from aigc.benchmarking.models import EvalTask, ExecutionRequest, TargetResult
from aigc.prompts.models import PromptRecord
from aigc.run_flow import run_single_model
from aigc.run_store import load_run_dataset_records
from aigc.utils.progress import NullRunProgress, RunProgress


class PromptRecordRunEngineGateway(RunEngineGateway):
    def execute_requests(
        self,
        *,
        task: EvalTask,
        requests: list[ExecutionRequest],
        experiment_dir: str | Path,
        execution_mode: str,
        progress: RunProgress | None = None,
    ) -> tuple[list[TargetResult], list[dict[str, Any]], list[str]]:
        progress = progress or NullRunProgress()
        experiment_path = Path(experiment_dir)
        target_results: list[TargetResult] = []
        failures: list[dict[str, Any]] = []
        target_run_ids: list[str] = []
        grouped_requests: dict[str, list[ExecutionRequest]] = {}
        for request in requests:
            grouped_requests.setdefault(request.target_model, []).append(request)

        for target_model, target_requests in sorted(grouped_requests.items()):
            prompt_path = experiment_path / "_execution_inputs" / f"{_slugify(target_model)}.jsonl"
            request_lookup = {request.request_id: request for request in target_requests}
            _write_execution_prompts(prompt_path, target_requests)
            progress.env_message(
                f"[benchmark] task={task.task_id} target={target_model} requests={len(target_requests)}"
            )
            run_summary = run_single_model(
                model_name=target_model,
                prompt_file=prompt_path,
                out_dir=experiment_path / "target_runs" / _slugify(target_model),
                run_name=f"{_slugify(task.task_id)}-{_slugify(target_model)}",
                execution_mode=execution_mode,
                progress=progress,
            )
            target_run_ids.append(run_summary.run_id)
            source_records = load_run_dataset_records(run_summary.run_id)
            if not source_records:
                failures.append(
                    {
                        "stage": "target_execution",
                        "target_model": target_model,
                        "run_id": run_summary.run_id,
                        "error": "No exported dataset records were found for target run.",
                        "target_run_dir": run_summary.output_dir,
                    }
                )
                continue
            for record in source_records:
                prompt_id = str(record.get("prompt_id", "")).strip()
                request = request_lookup.get(prompt_id)
                if request is None:
                    continue
                prompt_metadata = dict(record.get("prompt_metadata", {}) or {})
                case_metadata = dict(request.metadata.get("case_metadata", {}) or {})
                merged_metadata = dict(case_metadata)
                merged_metadata.update(prompt_metadata)
                target_results.append(
                    TargetResult(
                        task_id=task.task_id,
                        request_id=request.request_id,
                        benchmark_id=request.benchmark_id,
                        case_id=request.case_id,
                        case_version=_optional_text(request.metadata.get("case_version")),
                        source_builder=_optional_text(request.metadata.get("source_builder")),
                        target_model=target_model,
                        source_run_id=run_summary.run_id,
                        source_record_id=str(record.get("record_id", "")),
                        input_modality=request.input_modality,
                        split=str(request.metadata.get("split", "default")),
                        tags=[str(item) for item in request.metadata.get("tags", []) or []],
                        artifact_type=str(record.get("artifact_type", "")),
                        artifact_path=str(record.get("artifact_path", "")),
                        prompt=str(record.get("prompt", "")),
                        execution_status=str(record.get("status", "success") or "success"),
                        metadata=merged_metadata,
                        prompt_metadata=prompt_metadata,
                        artifact_metadata=dict(record.get("artifact_metadata", {}) or {}),
                        generation_params=dict(record.get("generation_params", {}) or {}),
                        runtime_summary={
                            "run_id": run_summary.run_id,
                            "output_dir": run_summary.output_dir,
                            "execution_mode": getattr(run_summary, "execution_mode", execution_mode),
                        },
                    )
                )
        return target_results, failures, target_run_ids


def _write_execution_prompts(path: Path, requests: list[ExecutionRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            prompt = _execution_request_to_prompt_record(request)
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "language": prompt.language,
                        "negative_prompt": prompt.negative_prompt,
                        "parameters": prompt.parameters,
                        "metadata": prompt.metadata,
                        "version": prompt.version,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _execution_request_to_prompt_record(request: ExecutionRequest) -> PromptRecord:
    payload = dict(request.input_payload)
    prompt_text = _resolve_request_prompt_text(request)
    metadata = {
        "task_id": request.task_id,
        "request_id": request.request_id,
        "benchmark_id": request.benchmark_id,
        "case_id": request.case_id,
        "case_version": request.metadata.get("case_version"),
        "source_builder": request.metadata.get("source_builder"),
        "input_modality": request.input_modality,
        "split": request.metadata.get("split", "default"),
        "tags": list(request.metadata.get("tags", []) or []),
        "grouping": dict(request.metadata.get("grouping", {}) or {}),
        "execution_hints": dict(request.metadata.get("execution_hints", {}) or {}),
        "evaluation_hints": dict(request.metadata.get("evaluation_hints", {}) or {}),
        "expected_output_contract": request.expected_output_contract,
        **dict(request.metadata.get("case_metadata", {}) or {}),
    }
    for key, value in payload.items():
        if key in {"prompt", "instruction", "context", "parameters"}:
            continue
        metadata.setdefault(key, value)
    return PromptRecord(
        prompt_id=request.request_id,
        prompt=prompt_text,
        language=str(payload.get("language") or "en"),
        negative_prompt=_optional_text(payload.get("negative_prompt")),
        parameters=dict(request.generation_params),
        metadata=metadata,
        version=_optional_text(request.metadata.get("case_version")),
    )


def _resolve_request_prompt_text(request: ExecutionRequest) -> str:
    payload = dict(request.input_payload)
    prompt_text = ""
    for key in ("prompt", "instruction", "text", "input"):
        value = payload.get(key)
        if value not in (None, ""):
            prompt_text = str(value)
            break
    if not prompt_text:
        prompt_text = json.dumps(payload, ensure_ascii=False)
    return _compose_prompt_text(
        prompt_text=prompt_text,
        payload=payload,
        metadata=dict(request.metadata),
    )


def _compose_prompt_text(
    *,
    prompt_text: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    composition = dict(metadata.get("prompt_composition", {}) or {})
    text_prompt_composition = dict(composition.get("text_prompt_composition", {}) or {})
    if not text_prompt_composition:
        text_prompt_composition = composition
    if not bool(text_prompt_composition.get("append_structured_choices", False)):
        return prompt_text
    decision_options = payload.get("decision_options")
    if not isinstance(decision_options, list) or not decision_options:
        case_metadata = dict(metadata.get("case_metadata", {}) or {})
        decision_options = case_metadata.get("decision_options")
    formatted_choices = _format_structured_choices(decision_options)
    if not formatted_choices:
        return prompt_text
    separator = str(text_prompt_composition.get("separator") or "\n\n").replace("\\n", "\n")
    header = str(text_prompt_composition.get("choices_header") or "Choices:").strip()
    if header:
        return f"{prompt_text}{separator}{header}\n{formatted_choices}"
    return f"{prompt_text}{separator}{formatted_choices}"


def _format_structured_choices(decision_options: Any) -> str:
    if not isinstance(decision_options, list):
        return ""
    rendered: list[str] = []
    for item in decision_options:
        if not isinstance(item, dict):
            return ""
        option_id = str(item.get("id", "")).strip().upper()
        text = str(item.get("text", "")).strip()
        if option_id not in {"A", "B"} or not text:
            return ""
        rendered.append(f"{option_id}. {text}")
    if len(rendered) != 2:
        return ""
    return "\n".join(rendered)


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    return "".join(char if char.isalnum() else "_" for char in lowered).strip("_") or "item"
