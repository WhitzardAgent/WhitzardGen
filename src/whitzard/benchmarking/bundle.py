from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from whitzard.benchmarking.models import (
    AnalysisPluginResult,
    BenchmarkCase,
    CaseSet,
    CompiledTaskPlan,
    ExecutionRequest,
    ExperimentLogEvent,
    GroupAnalysisRecord,
    NormalizedResult,
    ScoreRecord,
    TargetResult,
)


def write_benchmark_bundle(
    *,
    benchmark_dir: str | Path,
    cases: list[BenchmarkCase],
    manifest: dict[str, Any],
    stats: dict[str, Any],
    extra_jsonl_files: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, str]:
    target = Path(benchmark_dir)
    target.mkdir(parents=True, exist_ok=True)
    cases_path = target / "cases.jsonl"
    manifest_path = target / "benchmark_manifest.json"
    stats_path = target / "stats.json"

    _write_jsonl(cases_path, (case.to_dict() for case in cases))
    extra_paths: dict[str, str] = {}
    for filename, rows in sorted((extra_jsonl_files or {}).items()):
        if not filename.endswith(".jsonl"):
            continue
        path = target / filename
        _write_jsonl(path, rows)
        extra_paths[filename] = str(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {
        "case_set_path": str(cases_path),
        "cases_path": str(cases_path),
        "manifest_path": str(manifest_path),
        "stats_path": str(stats_path),
    }
    payload.update(extra_paths)
    return payload


def inspect_benchmark_bundle(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_file():
        bundle_dir = target.parent
        cases_path = target
        manifest_path = bundle_dir / "benchmark_manifest.json"
    else:
        bundle_dir = target
        cases_path = target / "cases.jsonl"
        manifest_path = target / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    cases = load_benchmark_cases(cases_path) if cases_path.exists() else []
    stats = build_benchmark_stats(cases)
    return {
        "benchmark_dir": str(bundle_dir),
        "manifest": manifest,
        "case_count": len(cases),
        "counts_by_input_type": stats["counts_by_input_type"],
        "counts_by_split": stats["counts_by_split"],
        "counts_by_builder": stats["counts_by_builder"],
        "counts_by_family": stats["counts_by_family"],
        "sample_case_ids": [case.case_id for case in cases[:5]],
        "case_set_path": str(cases_path),
        "raw_realizations_path": str(bundle_dir / "raw_realizations.jsonl") if (bundle_dir / "raw_realizations.jsonl").exists() else None,
        "rejected_realizations_path": str(bundle_dir / "rejected_realizations.jsonl") if (bundle_dir / "rejected_realizations.jsonl").exists() else None,
        "request_previews_path": str(bundle_dir / "request_previews.jsonl") if (bundle_dir / "request_previews.jsonl").exists() else None,
        "request_preview_summary_path": str(bundle_dir / "request_preview_summary.json") if (bundle_dir / "request_preview_summary.json").exists() else None,
        "request_previews_markdown_path": str(bundle_dir / "request_previews.md") if (bundle_dir / "request_previews.md").exists() else None,
    }


def write_experiment_bundle(
    *,
    experiment_dir: str | Path,
    manifest: dict[str, Any],
    case_set: CaseSet,
    compiled_task_plan: CompiledTaskPlan,
    execution_requests: list[ExecutionRequest],
    target_results: list[TargetResult],
    normalized_results: list[NormalizedResult],
    score_records: list[ScoreRecord],
    group_analysis_records: list[GroupAnalysisRecord],
    analysis_plugin_results: list[AnalysisPluginResult],
    experiment_log: list[ExperimentLogEvent],
    summary: dict[str, Any],
    report_markdown: str,
    failures: list[dict[str, Any]],
) -> dict[str, str]:
    target = Path(experiment_dir)
    target.mkdir(parents=True, exist_ok=True)

    manifest_path = target / "experiment_manifest.json"
    cases_path = target / "cases.jsonl"
    execution_requests_path = target / "execution_requests.jsonl"
    target_results_path = target / "target_results.jsonl"
    normalized_results_path = target / "normalized_results.jsonl"
    score_records_path = target / "score_records.jsonl"
    group_analysis_records_path = target / "group_analysis_records.jsonl"
    analysis_plugin_results_path = target / "analysis_plugin_results.jsonl"
    experiment_log_path = target / "experiment_log.jsonl"
    compiled_task_plan_path = target / "compiled_task_plan.json"
    summary_path = target / "summary.json"
    report_path = target / "report.md"
    failures_path = target / "failures.json"

    manifest_payload = dict(manifest)
    manifest_payload.setdefault("compiled_task_plan_path", str(compiled_task_plan_path))
    manifest_payload.setdefault("experiment_log_path", str(experiment_log_path))

    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(cases_path, (case.to_dict() for case in case_set.cases))
    _write_jsonl(execution_requests_path, (request.to_dict() for request in execution_requests))
    _write_jsonl(target_results_path, (record.to_dict() for record in target_results))
    _write_jsonl(normalized_results_path, (record.to_dict() for record in normalized_results))
    _write_jsonl(score_records_path, (record.to_dict() for record in score_records))
    _write_jsonl(group_analysis_records_path, (record.to_dict() for record in group_analysis_records))
    _write_jsonl(analysis_plugin_results_path, (record.to_dict() for record in analysis_plugin_results))
    _write_jsonl(experiment_log_path, (event.to_dict() for event in experiment_log))
    compiled_task_plan_path.write_text(
        json.dumps(compiled_task_plan.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report_markdown, encoding="utf-8")
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "case_set_path": str(cases_path),
        "cases_path": str(cases_path),
        "execution_requests_path": str(execution_requests_path),
        "target_results_path": str(target_results_path),
        "normalized_results_path": str(normalized_results_path),
        "score_records_path": str(score_records_path),
        "group_analysis_records_path": str(group_analysis_records_path),
        "analysis_plugin_results_path": str(analysis_plugin_results_path),
        "experiment_log_path": str(experiment_log_path),
        "compiled_task_plan_path": str(compiled_task_plan_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "failures_path": str(failures_path),
        "evaluator_results_path": str(score_records_path),
        "group_analyses_path": str(group_analysis_records_path),
    }


def inspect_experiment_bundle(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    bundle_dir = target.parent if target.is_file() else target
    manifest_path = bundle_dir / "experiment_manifest.json"
    summary_path = bundle_dir / "summary.json"
    report_path = bundle_dir / "report.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "experiment_dir": str(bundle_dir),
        "manifest": manifest,
        "summary": summary,
        "report_path": str(report_path) if report_path.exists() else None,
        "request_previews_path": str(bundle_dir / "request_previews.jsonl") if (bundle_dir / "request_previews.jsonl").exists() else None,
        "request_preview_summary_path": str(bundle_dir / "request_preview_summary.json") if (bundle_dir / "request_preview_summary.json").exists() else None,
        "request_previews_markdown_path": str(bundle_dir / "request_previews.md") if (bundle_dir / "request_previews.md").exists() else None,
    }


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            input_modality = str(payload.get("input_modality") or payload.get("input_type") or "text")
            input_payload = dict(payload.get("input_payload", {}) or {})
            prompt = _optional_text(payload.get("prompt"))
            instruction = _optional_text(payload.get("instruction"))
            if not input_payload:
                if prompt not in (None, ""):
                    input_payload["prompt"] = prompt
                if instruction not in (None, ""):
                    input_payload["instruction"] = instruction
                if payload.get("context") not in (None, "", {}):
                    input_payload["context"] = payload.get("context")
                if payload.get("language") not in (None, ""):
                    input_payload["language"] = payload.get("language")
            expected_output_contract = payload.get("expected_output_contract", payload.get("expected_structure"))
            cases.append(
                BenchmarkCase(
                    benchmark_id=str(payload.get("benchmark_id", "")),
                    case_id=str(payload.get("case_id") or payload.get("prompt_id", "")),
                    input_modality=input_modality,
                    input_payload=input_payload,
                    metadata=dict(payload.get("metadata", {}) or {}),
                    tags=list(payload.get("tags", []) or []),
                    split=str(payload.get("split", "default")),
                    expected_output_contract=expected_output_contract,
                    expected_structure=payload.get("expected_structure"),
                    case_version=_optional_text(payload.get("case_version") or payload.get("version")),
                    source_builder=_optional_text(payload.get("source_builder")),
                    grouping=dict(payload.get("grouping", {}) or {}),
                    execution_hints=dict(payload.get("execution_hints", {}) or {}),
                    evaluation_hints=dict(payload.get("evaluation_hints", {}) or {}),
                    language=str(payload.get("language", "en")),
                    parameters=dict(payload.get("parameters", {}) or {}),
                    prompt=prompt,
                    instruction=instruction,
                    context=payload.get("context"),
                    input_type=_optional_text(payload.get("input_type")),
                )
            )
    return cases


def build_benchmark_stats(cases: list[BenchmarkCase]) -> dict[str, Any]:
    counts_by_input_type = Counter()
    counts_by_split = Counter()
    counts_by_builder = Counter()
    counts_by_family = Counter()
    for case in cases:
        counts_by_input_type[case.input_modality] += 1
        counts_by_split[case.split] += 1
        counts_by_builder[str(case.source_builder or "unknown")] += 1
        family_id = str(case.metadata.get("family_id") or case.metadata.get("template_id") or "unspecified")
        counts_by_family[family_id] += 1
    return {
        "case_count": len(cases),
        "counts_by_input_type": dict(sorted(counts_by_input_type.items())),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_builder": dict(sorted(counts_by_builder.items())),
        "counts_by_family": dict(sorted(counts_by_family.items())),
    }


def _write_jsonl(path: Path, rows: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None
