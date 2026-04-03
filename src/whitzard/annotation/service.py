from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from whitzard.annotation.bundle import (
    build_annotation_stats,
    load_annotation_records,
    write_annotation_bundle,
)
from whitzard.annotation.config import (
    AnnotationConfigError,
    load_annotation_catalog,
    parse_annotation_response,
    render_annotation_template,
    render_output_contract,
    resolve_annotation_profile,
    validate_annotation_payload,
)
from whitzard.annotation.models import AnnotationBundleSummary
from whitzard.prompts.models import PromptRecord
from whitzard.registry import load_registry
from whitzard.run_flow import run_single_model
from whitzard.run_store import (
    RunStoreError,
    load_run_dataset_records,
    load_run_manifest,
)
from whitzard.settings import get_runs_root
from whitzard.utils.progress import NullRunProgress, RunProgress


class AnnotationError(RuntimeError):
    """Raised when annotation bundle generation fails."""


def annotate_run(
    source_run_id: str,
    *,
    annotation_profile: str | None = None,
    annotator_model: str | None = None,
    template_name: str | None = None,
    out_dir: str | Path | None = None,
    execution_mode: str = "real",
    progress: RunProgress | None = None,
    config_path: str | Path | None = None,
) -> AnnotationBundleSummary:
    progress = progress or NullRunProgress()
    created_at = datetime.now(UTC).isoformat()
    catalog = load_annotation_catalog(config_path) if config_path is not None else load_annotation_catalog()
    profile, template = resolve_annotation_profile(
        catalog,
        profile_name=annotation_profile,
        template_name=template_name,
    )
    resolved_annotator_model = annotator_model or profile.default_model
    if not resolved_annotator_model:
        raise AnnotationError(
            f"Annotation profile {profile.name} does not declare a default annotator model."
        )
    registry = load_registry()
    annotator_info = registry.get_model(resolved_annotator_model)
    if annotator_info.capabilities.get("supports_structured_json_output") is False:
        raise AnnotationError(
            f"Annotator model {resolved_annotator_model} does not support structured JSON output."
        )

    source_manifest = load_run_manifest(source_run_id)
    source_records = load_run_dataset_records(source_run_id)
    if not source_records:
        raise AnnotationError(f"No exported dataset records were found for run_id={source_run_id}.")

    bundle_id = _build_annotation_bundle_id(source_run_id=source_run_id, profile_name=profile.name)
    bundle_dir = Path(out_dir) if out_dir is not None else _default_annotation_bundle_dir(source_run_id, bundle_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = bundle_dir / "annotations.jsonl"
    existing_annotations = load_annotation_records(annotations_path)
    existing_record_ids = {
        str(record.get("source_record_id", "")).strip()
        for record in existing_annotations
        if str(record.get("source_record_id", "")).strip()
    }
    annotation_log: list[dict[str, Any]] = [
        {
            "event": "source_loaded",
            "timestamp": created_at,
            "source_run_id": source_run_id,
            "source_record_count": len(source_records),
        }
    ]
    failures: list[dict[str, Any]] = []

    compatible_records: list[dict[str, Any]] = []
    skipped_count = 0
    for record in source_records:
        source_record_id = str(record.get("record_id", "")).strip()
        artifact_type = str(record.get("artifact_type", "")).strip()
        if source_record_id in existing_record_ids:
            skipped_count += 1
            annotation_log.append(
                {
                    "event": "record_skipped_existing",
                    "timestamp": created_at,
                    "source_record_id": source_record_id,
                }
            )
            continue
        if profile.accepted_source_artifact_types and artifact_type not in profile.accepted_source_artifact_types:
            skipped_count += 1
            failures.append(
                _build_failure_record(
                    source_run_id=source_run_id,
                    source_record=record,
                    stage="compatibility",
                    error_message=(
                        f"artifact_type={artifact_type} is not allowed by annotation profile {profile.name}"
                    ),
                )
            )
            continue
        model_accepted_types = [
            str(item)
            for item in annotator_info.capabilities.get(
                "accepted_annotation_source_artifact_types", []
            )
            or []
        ]
        if model_accepted_types and artifact_type not in model_accepted_types:
            skipped_count += 1
            failures.append(
                _build_failure_record(
                    source_run_id=source_run_id,
                    source_record=record,
                    stage="compatibility",
                    error_message=(
                        f"artifact_type={artifact_type} is not accepted by annotator model "
                        f"{resolved_annotator_model}"
                    ),
                )
            )
            continue
        compatible_records.append(record)

    if compatible_records:
        request_prompts = [
            _build_annotation_request_prompt(
                source_record=record,
                source_run_id=source_run_id,
                template_text=template.instruction_template,
                output_contract=profile.output_contract,
                annotator_model=resolved_annotator_model,
                annotation_profile=profile.name,
                annotation_template=template.name,
            )
            for record in compatible_records
        ]
        requests_path = bundle_dir / "_annotation_inputs" / "prompts.jsonl"
        _write_annotation_request_prompts(requests_path, request_prompts)
        annotation_run_dir = bundle_dir / "_annotation_run"
        progress.env_message(
            f"[annotate] source_run={source_run_id} records={len(compatible_records)} model={resolved_annotator_model}"
        )
        run_summary = run_single_model(
            model_name=resolved_annotator_model,
            prompt_file=requests_path,
            out_dir=annotation_run_dir,
            run_name=f"annotate-{_slugify(source_run_id)}",
            execution_mode=execution_mode,
            progress=progress,
            profile_generation_defaults=profile.generation_defaults,
        )
        annotation_log.append(
            {
                "event": "annotation_run_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "annotation_run_id": run_summary.run_id,
                "annotation_run_dir": run_summary.output_dir,
                "annotation_run_export_path": run_summary.export_path,
                "status": run_summary.status,
            }
        )
        new_annotations, parse_failures = _collect_annotation_results(
            source_run_id=source_run_id,
            request_prompts=request_prompts,
            annotation_run_summary=run_summary.to_dict(),
            output_contract=profile.output_contract,
            annotation_profile=profile.name,
            annotation_template=template.name,
            annotator_model=resolved_annotator_model,
        )
        annotation_run_failures = _load_annotation_run_failures(run_summary.output_dir)
        failures.extend(parse_failures)
        failures.extend(annotation_run_failures)
    else:
        run_summary = None
        new_annotations = []
        annotation_log.append(
            {
                "event": "annotation_run_skipped",
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": "no compatible source records remained after skip/filter checks",
            }
        )

    combined_annotations = existing_annotations + new_annotations
    stats = build_annotation_stats(
        combined_annotations,
        failures=failures,
        skipped_count=skipped_count,
    )
    manifest = {
        "bundle_id": bundle_id,
        "created_at": created_at,
        "source_run_id": source_run_id,
        "source_run_manifest_path": str(
            Path(source_manifest.get("manifest_path", get_runs_root() / source_run_id / "run_manifest.json"))
        ),
        "source_export_path": str(source_manifest.get("export_path", "")),
        "annotation_profile": profile.name,
        "annotation_profile_version": profile.version,
        "annotation_template": template.name,
        "annotation_template_version": template.version,
        "annotator_model": resolved_annotator_model,
        "execution_mode": execution_mode,
        "source_record_count": len(source_records),
        "annotated_count": len(new_annotations),
        "total_annotation_count": len(combined_annotations),
        "skipped_count": skipped_count,
        "failed_count": len(failures),
        "annotation_run_id": run_summary.run_id if run_summary is not None else None,
        "annotation_run_dir": run_summary.output_dir if run_summary is not None else None,
        "output_contract": dict(profile.output_contract),
    }
    bundle_paths = write_annotation_bundle(
        bundle_dir=bundle_dir,
        annotations=combined_annotations,
        manifest=manifest,
        annotation_log=annotation_log,
        stats=stats,
        failures=failures,
    )
    return AnnotationBundleSummary(
        bundle_id=bundle_id,
        bundle_dir=str(bundle_dir),
        annotations_path=bundle_paths["annotations_path"],
        manifest_path=bundle_paths["manifest_path"],
        log_path=bundle_paths["log_path"],
        stats_path=bundle_paths["stats_path"],
        failures_path=bundle_paths["failures_path"],
        source_run_id=source_run_id,
        annotator_model=resolved_annotator_model,
        annotation_profile=profile.name,
        annotation_template=template.name,
        source_record_count=len(source_records),
        annotated_count=len(new_annotations),
        skipped_count=skipped_count,
        failed_count=len(failures),
        annotation_run_id=run_summary.run_id if run_summary is not None else None,
    )


def _collect_annotation_results(
    *,
    source_run_id: str,
    request_prompts: list[PromptRecord],
    annotation_run_summary: dict[str, Any],
    output_contract: dict[str, Any],
    annotation_profile: str,
    annotation_template: str,
    annotator_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    export_path = Path(str(annotation_run_summary["export_path"]))
    if not export_path.exists():
        raise AnnotationError(f"Annotation run export path does not exist: {export_path}")
    request_lookup = {prompt.prompt_id: prompt for prompt in request_prompts}
    annotations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line in export_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        prompt_id = str(payload.get("prompt_id", "")).strip()
        request_prompt = request_lookup.get(prompt_id)
        if request_prompt is None:
            continue
        source_info = dict(request_prompt.metadata.get("annotation_source_record", {}))
        artifact_path = Path(str(payload.get("artifact_path", "")))
        raw_response = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else ""
        try:
            parsed = parse_annotation_response(raw_response)
            validate_annotation_payload(parsed, output_contract)
        except AnnotationConfigError as exc:
            failures.append(
                {
                    "source_run_id": source_run_id,
                    "source_record_id": source_info.get("source_record_id"),
                    "source_prompt_id": source_info.get("source_prompt_id"),
                    "stage": "parse",
                    "error_message": str(exc),
                    "raw_response_path": str(artifact_path),
                }
            )
            continue
        annotations.append(
            {
                "annotation_id": f"ann_{len(annotations) + 1:08d}",
                "source_run_id": source_run_id,
                "source_record_id": source_info.get("source_record_id"),
                "source_prompt_id": source_info.get("source_prompt_id"),
                "source_task_id": source_info.get("source_task_id"),
                "source_artifact_type": source_info.get("source_artifact_type"),
                "source_artifact_path": source_info.get("source_artifact_path"),
                "source_artifact_metadata": dict(source_info.get("source_artifact_metadata", {})),
                "source_model_name": source_info.get("source_model_name"),
                "source_prompt": source_info.get("source_prompt"),
                "source_negative_prompt": source_info.get("source_negative_prompt"),
                "source_prompt_metadata": dict(source_info.get("source_prompt_metadata", {})),
                "source_generation_params": dict(source_info.get("source_generation_params", {})),
                "annotator_model": annotator_model,
                "annotation_profile": annotation_profile,
                "annotation_template": annotation_template,
                "annotation_run_id": annotation_run_summary.get("run_id"),
                "annotation_status": "success",
                "validation_status": "valid",
                "annotation": parsed,
                "raw_annotation_response_path": str(artifact_path),
            }
        )
    return annotations, failures


def _build_annotation_request_prompt(
    *,
    source_record: dict[str, Any],
    source_run_id: str,
    template_text: str,
    output_contract: dict[str, Any],
    annotator_model: str,
    annotation_profile: str,
    annotation_template: str,
) -> PromptRecord:
    source_prompt_metadata = dict(source_record.get("prompt_metadata", {}))
    annotation_source_record = {
        "source_run_id": source_run_id,
        "source_record_id": source_record.get("record_id"),
        "source_prompt_id": source_record.get("prompt_id"),
        "source_task_id": source_record.get("task_id"),
        "source_model_name": source_record.get("model_name"),
        "source_task_type": source_record.get("task_type"),
        "source_artifact_type": source_record.get("artifact_type"),
        "source_artifact_path": source_record.get("artifact_path"),
        "source_artifact_metadata": dict(source_record.get("artifact_metadata", {})),
        "source_prompt": source_record.get("prompt"),
        "source_negative_prompt": source_record.get("negative_prompt"),
        "source_prompt_metadata": source_prompt_metadata,
        "source_generation_params": dict(source_record.get("generation_params", {})),
    }
    prompt_text = render_annotation_template(
        template_text,
        values={
            "source_run_id": source_run_id,
            "source_record_id": source_record.get("record_id", ""),
            "source_prompt_id": source_record.get("prompt_id", ""),
            "source_prompt": source_record.get("prompt", ""),
            "source_negative_prompt": source_record.get("negative_prompt") or "",
            "source_language": source_record.get("language", ""),
            "source_model_name": source_record.get("model_name", ""),
            "source_task_type": source_record.get("task_type", ""),
            "source_artifact_type": source_record.get("artifact_type", ""),
            "source_artifact_path": source_record.get("artifact_path", ""),
            "source_artifact_metadata_json": json.dumps(
                source_record.get("artifact_metadata", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "source_prompt_metadata_json": json.dumps(
                source_prompt_metadata,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "source_generation_params_json": json.dumps(
                source_record.get("generation_params", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "output_contract_block": render_output_contract(output_contract),
        },
    )
    prompt_id = f"annreq_{_slugify(str(source_record.get('record_id', 'record')))}"
    metadata = {
        "annotation_profile": annotation_profile,
        "annotation_template": annotation_template,
        "annotator_model": annotator_model,
        "annotation_source_record": annotation_source_record,
    }
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt_text,
        language=str(source_record.get("language", "en") or "en"),
        negative_prompt=None,
        parameters={},
        metadata=metadata,
    )


def _write_annotation_request_prompts(path: Path, prompts: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "language": prompt.language,
                        "negative_prompt": prompt.negative_prompt,
                        "parameters": prompt.parameters,
                        "metadata": prompt.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _load_annotation_run_failures(annotation_run_dir: str | Path) -> list[dict[str, Any]]:
    run_root = Path(annotation_run_dir)
    failures_path = run_root / "failures.json"
    if not failures_path.exists():
        return []
    try:
        payload = json.loads(failures_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    failures: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        failures.append(
            {
                "stage": "execution",
                "error_message": str(item.get("error_message") or item.get("message") or "annotation task failed"),
                "task_id": item.get("task_id"),
                "prompt_id": item.get("prompt_id"),
            }
        )
    return failures


def _build_failure_record(
    *,
    source_run_id: str,
    source_record: dict[str, Any],
    stage: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "source_run_id": source_run_id,
        "source_record_id": source_record.get("record_id"),
        "source_prompt_id": source_record.get("prompt_id"),
        "source_artifact_path": source_record.get("artifact_path"),
        "source_artifact_type": source_record.get("artifact_type"),
        "stage": stage,
        "error_message": error_message,
    }


def _build_annotation_bundle_id(*, source_run_id: str, profile_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"annotation_{_slugify(source_run_id)}_{_slugify(profile_name)}_{timestamp}"


def _default_annotation_bundle_dir(source_run_id: str, bundle_id: str) -> Path:
    return get_runs_root() / source_run_id / "annotations" / bundle_id


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug or "annotation"
