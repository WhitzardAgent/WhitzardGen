from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from whitzard.benchmarking.bundle import inspect_experiment_bundle, load_benchmark_cases
from whitzard.benchmarking.models import ExperimentExportSummary
from whitzard.exporters.bundle import ExportBundleError, load_jsonl_records
from whitzard.settings import get_experiments_root


class ExperimentExportError(RuntimeError):
    """Raised when an experiment bundle export cannot be materialized."""


def export_experiment_bundle(
    *,
    experiment: str | Path,
    output_dir: str | Path | None = None,
    export_format: str = "both",
) -> ExperimentExportSummary:
    if export_format not in {"jsonl", "csv", "both"}:
        raise ExperimentExportError(f"Unsupported experiment export format: {export_format}")

    experiment_dir = _resolve_experiment_dir(experiment)
    payload = inspect_experiment_bundle(experiment_dir)
    manifest = dict(payload.get("manifest") or {})
    experiment_id = str(manifest.get("experiment_id") or experiment_dir.name)

    cases_path = experiment_dir / "cases.jsonl"
    execution_requests_path = experiment_dir / "execution_requests.jsonl"
    target_results_path = experiment_dir / "target_results.jsonl"
    normalized_results_path = experiment_dir / "normalized_results.jsonl"
    score_records_path = experiment_dir / "score_records.jsonl"

    if not target_results_path.exists():
        raise ExperimentExportError(f"Missing target results for experiment export: {target_results_path}")

    export_dir = (
        Path(output_dir)
        if output_dir is not None
        else experiment_dir / "exports" / f"analysis_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    cases = load_benchmark_cases(cases_path) if cases_path.exists() else []
    try:
        execution_requests = load_jsonl_records(execution_requests_path) if execution_requests_path.exists() else []
        target_results = load_jsonl_records(target_results_path)
        normalized_results = load_jsonl_records(normalized_results_path) if normalized_results_path.exists() else []
        score_records = load_jsonl_records(score_records_path) if score_records_path.exists() else []
    except ExportBundleError as exc:
        raise ExperimentExportError(str(exc)) from exc

    merged_rows = _build_export_rows(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        cases=cases,
        execution_requests=execution_requests,
        target_results=target_results,
        normalized_results=normalized_results,
        score_records=score_records,
    )

    jsonl_path: Path | None = None
    csv_path: Path | None = None
    if export_format in {"jsonl", "both"}:
        jsonl_path = export_dir / "dataset.jsonl"
        _write_jsonl(jsonl_path, merged_rows)
    if export_format in {"csv", "both"}:
        csv_path = export_dir / "dataset.csv"
        _write_csv(csv_path, merged_rows)

    manifest_path = export_dir / "export_manifest.json"
    readme_path = export_dir / "README.md"

    export_manifest = {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "export_format": export_format,
        "record_count": len(merged_rows),
        "source_files": {
            "cases_path": str(cases_path) if cases_path.exists() else None,
            "execution_requests_path": str(execution_requests_path) if execution_requests_path.exists() else None,
            "target_results_path": str(target_results_path),
            "normalized_results_path": str(normalized_results_path) if normalized_results_path.exists() else None,
            "score_records_path": str(score_records_path) if score_records_path.exists() else None,
        },
        "output_files": {
            "jsonl_path": str(jsonl_path) if jsonl_path is not None else None,
            "csv_path": str(csv_path) if csv_path is not None else None,
            "manifest_path": str(manifest_path),
            "readme_path": str(readme_path),
        },
        "columns": sorted(merged_rows[0].keys()) if merged_rows else [],
    }
    manifest_path.write_text(json.dumps(export_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(
        _build_export_readme(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            export_format=export_format,
            record_count=len(merged_rows),
            jsonl_path=jsonl_path,
            csv_path=csv_path,
        ),
        encoding="utf-8",
    )

    return ExperimentExportSummary(
        experiment_id=experiment_id,
        experiment_dir=str(experiment_dir),
        export_dir=str(export_dir),
        export_format=export_format,
        record_count=len(merged_rows),
        jsonl_path=str(jsonl_path) if jsonl_path is not None else None,
        csv_path=str(csv_path) if csv_path is not None else None,
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
    )


def _resolve_experiment_dir(experiment: str | Path) -> Path:
    path = Path(experiment)
    if path.exists():
        return path.parent if path.is_file() else path
    candidate = get_experiments_root() / str(experiment)
    if candidate.exists():
        return candidate.parent if candidate.is_file() else candidate
    raise ExperimentExportError(f"Experiment bundle not found: {experiment}")


def _build_export_rows(
    *,
    experiment_id: str,
    experiment_dir: Path,
    cases: list[Any],
    execution_requests: list[dict[str, Any]],
    target_results: list[dict[str, Any]],
    normalized_results: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_by_id = {case.case_id: case for case in cases}
    request_by_id = {str(item.get("request_id", "")): item for item in execution_requests if item.get("request_id")}
    normalized_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    scores_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in normalized_results:
        normalized_by_key[(str(row.get("request_id", "")), str(row.get("target_model", "")))].append(row)
    for row in score_records:
        scores_by_key[(str(row.get("request_id", "")), str(row.get("target_model", "")))].append(row)

    merged_rows: list[dict[str, Any]] = []
    for row in target_results:
        request_id = str(row.get("request_id", ""))
        target_model = str(row.get("target_model", ""))
        case_id = str(row.get("case_id", ""))
        case = case_by_id.get(case_id)
        request = request_by_id.get(request_id, {})
        matched_normalized = normalized_by_key.get((request_id, target_model), [])
        matched_scores = scores_by_key.get((request_id, target_model), [])
        preferred_normalized = _pick_preferred_normalized_result(matched_normalized)
        artifact_path = Path(str(row.get("artifact_path", "")))
        output_text = _read_artifact_text(artifact_path)

        merged_rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_dir": str(experiment_dir),
                "benchmark_id": str(row.get("benchmark_id") or getattr(case, "benchmark_id", "") or ""),
                "case_id": case_id,
                "request_id": request_id,
                "target_model": target_model,
                "split": str(row.get("split") or getattr(case, "split", "default") or "default"),
                "case_version": _optional_text(row.get("case_version") or getattr(case, "case_version", None)),
                "source_builder": _optional_text(row.get("source_builder") or getattr(case, "source_builder", None)),
                "input_modality": str(row.get("input_modality") or getattr(case, "input_modality", "text") or "text"),
                "input_prompt": str(row.get("prompt", "")),
                "case_prompt": _optional_text(getattr(case, "prompt", None)),
                "case_instruction": _optional_text(getattr(case, "instruction", None)),
                "output_text": output_text,
                "artifact_type": str(row.get("artifact_type", "")),
                "artifact_path": str(artifact_path) if str(row.get("artifact_path", "")).strip() else "",
                "execution_status": str(row.get("execution_status", "")),
                "source_run_id": _optional_text(row.get("source_run_id")),
                "source_record_id": _optional_text(row.get("source_record_id")),
                "decision_text": _optional_text(preferred_normalized.get("decision_text") if preferred_normalized else None),
                "reasoning_trace_text": _optional_text(
                    preferred_normalized.get("reasoning_trace_text") if preferred_normalized else None
                ),
                "refusal_flag": preferred_normalized.get("refusal_flag") if preferred_normalized else None,
                "case_tags": list(getattr(case, "tags", []) or row.get("tags", []) or []),
                "case_grouping": dict(getattr(case, "grouping", {}) or {}),
                "case_metadata": dict(getattr(case, "metadata", {}) or {}),
                "case_input_payload": dict(getattr(case, "input_payload", {}) or {}),
                "execution_request": dict(request or {}),
                "target_result_metadata": dict(row.get("metadata", {}) or {}),
                "prompt_metadata": dict(row.get("prompt_metadata", {}) or {}),
                "artifact_metadata": dict(row.get("artifact_metadata", {}) or {}),
                "generation_params": dict(row.get("generation_params", {}) or {}),
                "runtime_summary": dict(row.get("runtime_summary", {}) or {}),
                "normalized_results": list(matched_normalized),
                "score_records": list(matched_scores),
                "normalized_result_count": len(matched_normalized),
                "score_record_count": len(matched_scores),
            }
        )
    return merged_rows


def _pick_preferred_normalized_result(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    successful = [row for row in rows if str(row.get("status", "")).lower() == "success"]
    ranked = successful or rows
    ranked.sort(key=lambda row: str(row.get("normalizer_id", "")))
    return ranked[0]


def _read_artifact_text(path: Path) -> str | None:
    if not str(path).strip() or not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    serialized_rows = [_serialize_csv_row(row) for row in rows]
    fieldnames = sorted({key for row in serialized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in serialized_rows:
            writer.writerow(row)


def _serialize_csv_row(row: dict[str, Any]) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for key, value in row.items():
        if value is None:
            serialized[key] = ""
        elif isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value, ensure_ascii=False)
        else:
            serialized[key] = str(value)
    return serialized


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _build_export_readme(
    *,
    experiment_id: str,
    experiment_dir: Path,
    export_format: str,
    record_count: int,
    jsonl_path: Path | None,
    csv_path: Path | None,
) -> str:
    lines = [
        f"# Experiment Export: {experiment_id}",
        "",
        f"- Source experiment: `{experiment_dir}`",
        f"- Formats: `{export_format}`",
        f"- Records: `{record_count}`",
    ]
    if jsonl_path is not None:
        lines.append(f"- JSONL: `{jsonl_path.name}`")
    if csv_path is not None:
        lines.append(f"- CSV: `{csv_path.name}`")
    lines.extend(
        [
            "",
            "Each exported row merges:",
            "- the selected benchmark case",
            "- the concrete execution request",
            "- the target model output artifact text when it is readable as UTF-8",
            "- matching normalized results",
            "- matching score records",
        ]
    )
    return "\n".join(lines) + "\n"
