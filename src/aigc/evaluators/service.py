from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aigc.annotation import annotate_run
from aigc.benchmarking.models import CompiledTaskPlan, EvalTask, NormalizedResult, ScoreRecord, TargetResult
from aigc.evaluators.models import EvaluatorSpec
from aigc.utils.progress import NullRunProgress, RunProgress


class EvaluatorError(RuntimeError):
    """Raised when evaluator execution fails."""


def score_target_results(
    *,
    task: EvalTask,
    compiled_plan: CompiledTaskPlan,
    source_run_id: str,
    target_results: list[TargetResult],
    normalized_results: list[NormalizedResult],
    scorers: list[EvaluatorSpec | dict[str, Any]],
    out_dir: str | Path,
    execution_mode: str,
    progress: RunProgress | None = None,
) -> tuple[list[ScoreRecord], list[dict[str, Any]]]:
    progress = progress or NullRunProgress()
    del compiled_plan, normalized_results
    results: list[ScoreRecord] = []
    failures: list[dict[str, Any]] = []
    if not scorers:
        return results, failures

    target_results_by_record = {
        result.source_record_id: result for result in target_results
    }
    target_results_by_run_id: dict[str, list[TargetResult]] = {}
    for result in target_results:
        target_results_by_run_id.setdefault(result.source_run_id, []).append(result)

    for raw_scorer in scorers:
        scorer = raw_scorer if isinstance(raw_scorer, EvaluatorSpec) else EvaluatorSpec(**raw_scorer)
        if scorer.evaluator_type == "rule":
            evaluator_results, evaluator_failures = _run_rule_evaluator(
                task=task,
                target_results=target_results,
                scorer=scorer,
            )
            results.extend(evaluator_results)
            failures.extend(evaluator_failures)
            continue

        if scorer.evaluator_type == "judge":
            run_ids = sorted(target_results_by_run_id)
            if not run_ids and source_run_id:
                run_ids = [source_run_id]
            for current_run_id in run_ids:
                progress.env_message(
                    f"[evaluate] source_run={current_run_id} scorer={scorer.scorer_id} type=judge"
                )
                bundle_dir = Path(out_dir) / scorer.scorer_id / current_run_id
                annotation_summary = annotate_run(
                    current_run_id,
                    annotation_profile=scorer.annotation_profile,
                    annotator_model=scorer.judge_model,
                    template_name=scorer.annotation_template,
                    out_dir=bundle_dir,
                    execution_mode=execution_mode,
                    progress=progress,
                )
                annotation_rows = _load_jsonl_rows(annotation_summary.annotations_path)
                for row in annotation_rows:
                    source_record_id = str(row.get("source_record_id", "")).strip()
                    target_result = target_results_by_record.get(source_record_id)
                    if target_result is None:
                        continue
                    results.append(
                        _convert_annotation_row_to_score_record(
                            task=task,
                            row=row,
                            target_result=target_result,
                            scorer=scorer,
                        )
                    )
                failures.extend(_load_failures_json(annotation_summary.failures_path))
            continue

        failures.append(
            {
                "stage": "scorer_resolution",
                "scorer_id": scorer.scorer_id,
                "error": f"Unsupported scorer type: {scorer.evaluator_type}",
            }
        )
    return results, failures


def evaluate_target_run(
    *,
    task: EvalTask,
    compiled_plan: CompiledTaskPlan,
    source_run_id: str,
    target_results: list[TargetResult],
    normalized_results: list[NormalizedResult],
    evaluators: list[EvaluatorSpec | dict[str, Any]],
    out_dir: str | Path,
    execution_mode: str,
    progress: RunProgress | None = None,
) -> tuple[list[ScoreRecord], list[dict[str, Any]]]:
    return score_target_results(
        task=task,
        compiled_plan=compiled_plan,
        source_run_id=source_run_id,
        target_results=target_results,
        normalized_results=normalized_results,
        scorers=evaluators,
        out_dir=out_dir,
        execution_mode=execution_mode,
        progress=progress,
    )


def _run_rule_evaluator(
    *,
    task: EvalTask,
    target_results: list[TargetResult],
    scorer: EvaluatorSpec,
) -> tuple[list[ScoreRecord], list[dict[str, Any]]]:
    results: list[ScoreRecord] = []
    failures: list[dict[str, Any]] = []
    rule_type = str(scorer.rule_type or "").strip()
    patterns = [str(item) for item in scorer.rule_config.get("patterns", []) or []]
    labels = [str(item) for item in scorer.rule_config.get("labels", []) or []]
    score_name = str(scorer.rule_config.get("score_name", "matched"))
    for target_result in target_results:
        artifact_text = _read_text_artifact(target_result.artifact_path)
        matched = False
        if rule_type == "contains_any":
            matched = any(pattern.lower() in artifact_text.lower() for pattern in patterns)
        elif rule_type == "regex_any":
            matched = any(re.search(pattern, artifact_text, re.IGNORECASE) for pattern in patterns)
        elif rule_type == "exact_match":
            matched = artifact_text.strip() in {pattern.strip() for pattern in patterns}
        else:
            failures.append(
                {
                    "stage": "rule_scorer",
                    "scorer_id": scorer.scorer_id,
                    "source_record_id": target_result.source_record_id,
                    "error": f"Unsupported rule_type: {rule_type}",
                }
            )
            continue
        result_labels = list(labels)
        if matched and not result_labels:
            result_labels = [scorer.scorer_id]
        results.append(
            ScoreRecord(
                task_id=task.task_id,
                benchmark_id=target_result.benchmark_id,
                case_id=target_result.case_id,
                case_version=target_result.case_version,
                request_id=target_result.request_id,
                target_model=target_result.target_model,
                scorer_id=scorer.scorer_id,
                status="success",
                labels=result_labels if matched else [],
                scores={score_name: 1.0 if matched else 0.0},
                rationale=f"rule_type={rule_type}",
                raw_judgment={
                    "matched": matched,
                    "patterns": patterns,
                },
                scorer_metadata={
                    "type": "rule",
                    "rule_type": rule_type,
                },
                split=target_result.split,
                tags=list(target_result.tags),
                source_record_id=target_result.source_record_id,
            )
        )
    return results, failures


def _convert_annotation_row_to_score_record(
    *,
    task: EvalTask,
    row: dict[str, Any],
    target_result: TargetResult,
    scorer: EvaluatorSpec,
) -> ScoreRecord:
    annotation = dict(row.get("annotation", {}) or {})
    labels = _coerce_labels(annotation)
    scores = _coerce_scores(annotation)
    rationale = _coerce_rationale(annotation)
    return ScoreRecord(
        task_id=task.task_id,
        benchmark_id=target_result.benchmark_id,
        case_id=target_result.case_id,
        case_version=target_result.case_version,
        request_id=target_result.request_id,
        target_model=target_result.target_model,
        scorer_id=scorer.scorer_id,
        status=str(row.get("annotation_status", "success") or "success"),
        labels=labels,
        scores=scores,
        rationale=rationale,
        raw_judgment=annotation,
        scorer_metadata={
            "type": "judge",
            "judge_model": scorer.judge_model,
            "annotation_profile": scorer.annotation_profile,
            "annotation_template": scorer.annotation_template,
        },
        split=target_result.split,
        tags=list(target_result.tags),
        source_record_id=target_result.source_record_id,
    )


def _coerce_labels(annotation: dict[str, Any]) -> list[str]:
    if isinstance(annotation.get("labels"), list):
        return [str(item) for item in annotation.get("labels", [])]
    if isinstance(annotation.get("normative_labels"), list):
        return [str(item) for item in annotation.get("normative_labels", [])]
    if annotation.get("recommended_action") not in (None, ""):
        return [str(annotation.get("recommended_action"))]
    return []


def _coerce_scores(annotation: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    if annotation.get("confidence") not in (None, ""):
        scores["confidence"] = annotation.get("confidence")
    if annotation.get("score") not in (None, ""):
        scores["score"] = annotation.get("score")
    return scores


def _coerce_rationale(annotation: dict[str, Any]) -> str | None:
    for key in ("rationale", "primary_justification", "summary"):
        value = annotation.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _read_text_artifact(path: str | Path) -> str:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return ""
    if artifact_path.suffix.lower() not in {".txt", ".json", ".md"}:
        return ""
    return artifact_path.read_text(encoding="utf-8").strip()


def _load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_failures_json(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []
