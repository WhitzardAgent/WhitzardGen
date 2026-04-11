from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from whitzard.analysis import run_analysis_plugins
from whitzard.analysis.models import AnalysisPluginSpec
from whitzard.benchmarking.bundle import write_experiment_bundle
from whitzard.benchmarking.gateway import PromptRecordRunEngineGateway
from whitzard.benchmarking.interfaces import ExperimentRunner
from whitzard.benchmarking.models import (
    AnalysisPluginResult,
    CompiledTaskPlan,
    EvalTask,
    EvaluationExperimentSummary,
    ExperimentBundleManifest,
    ExperimentLogEvent,
    GroupAnalysisRecord,
    NormalizedResult,
    ScoreRecord,
    SummaryReport,
    TargetRunReference,
    TargetResult,
)
from whitzard.benchmarking.preview import PreviewCollector, parse_preview_stage, write_request_preview_bundle
from whitzard.evaluators.service import score_target_results
from whitzard.launching import plan_model_launch
from whitzard.normalizers.service import normalize_target_results
from whitzard.utils.progress import NullRunProgress, RunProgress


class DefaultExperimentRunner(ExperimentRunner):
    def __init__(self) -> None:
        self.gateway = PromptRecordRunEngineGateway()

    def run(
        self,
        *,
        task: EvalTask,
        compiled_plan: CompiledTaskPlan,
        experiment_dir: str | Path,
        execution_mode: str,
        progress: RunProgress | None = None,
    ) -> EvaluationExperimentSummary:
        progress = progress or NullRunProgress()
        experiment_path = Path(experiment_dir)
        experiment_path.mkdir(parents=True, exist_ok=True)
        benchmark_id = compiled_plan.case_set.benchmark_id
        started_at = time.monotonic()
        experiment_id = f"experiment_{_slugify(task.task_id)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        events: list[ExperimentLogEvent] = []
        failures: list[dict[str, Any]] = []
        total_expected_units = _estimate_total_evaluation_units(
            execution_request_count=len(compiled_plan.execution_requests),
            normalizer_count=len(compiled_plan.normalizer_specs),
            scorer_count=len(compiled_plan.scorer_specs),
            plugin_count=len(compiled_plan.plugin_specs),
        )
        _append_event(
            events,
            experiment_id=experiment_id,
            task_id=task.task_id,
            stage="planning",
            entity_type="task",
            entity_id=task.task_id,
            status="compiled",
            payload={"execution_request_count": len(compiled_plan.execution_requests)},
        )
        _emit_evaluation_eta(
            progress=progress,
            started_at=started_at,
            stage="planning",
            processed_units=0,
            total_units=total_expected_units,
            failed_units=0,
        )
        launch_plan = plan_model_launch(
            requested_models=task.target_models,
            auto_launch=bool(task.execution_policy.get("auto_launch", False)),
            config_path=task.execution_policy.get("launcher_config_path"),
        )

        failed_stages: list[str] = []
        _append_event(
            events,
            experiment_id=experiment_id,
            task_id=task.task_id,
            stage="execution",
            entity_type="gateway",
            entity_id="run_engine_gateway",
            status="started",
            payload={"target_models": list(task.target_models)},
        )
        preview_config = dict(task.execution_policy.get("request_preview", {}) or {})
        preview_collector = (
            PreviewCollector(
                enabled_stages=parse_preview_stage(
                    str(preview_config.get("preview_stage") or "all"),
                    allowed_stages={"target", "judge", "all"},
                ),
                source_context={
                    "task_id": task.task_id,
                    "benchmark_id": benchmark_id,
                    "benchmark_path": compiled_plan.case_set.case_set_path,
                    "target_models": list(task.target_models),
                    "preview_only": False,
                },
            )
            if bool(preview_config.get("enabled", False))
            else None
        )
        target_results: list[TargetResult] = []
        target_failures: list[dict[str, Any]] = []
        target_run_refs: list[TargetRunReference] = []
        try:
            target_results, target_failures, target_run_refs = self.gateway.execute_requests(
                task=task,
                requests=compiled_plan.execution_requests,
                experiment_dir=experiment_path,
                execution_mode=execution_mode,
                preview_collector=preview_collector,
                progress=progress,
            )
            failures.extend(target_failures)
            _append_event(
                events,
                experiment_id=experiment_id,
                task_id=task.task_id,
                stage="execution",
                entity_type="gateway",
                entity_id="run_engine_gateway",
                status="completed",
                payload={"target_result_count": len(target_results), "failure_count": len(target_failures)},
            )
        except Exception as exc:
            failed_stages.append("execution")
            failures.append(
                {
                    "stage": "execution",
                    "entity_id": "run_engine_gateway",
                    "error": str(exc),
                }
            )
            _append_event(
                events,
                experiment_id=experiment_id,
                task_id=task.task_id,
                stage="execution",
                entity_type="gateway",
                entity_id="run_engine_gateway",
                status="failed",
                payload={"error": str(exc)},
            )
        target_run_ids = [run_ref.run_id for run_ref in target_run_refs]
        _emit_evaluation_eta(
            progress=progress,
            started_at=started_at,
            stage="execution",
            processed_units=len(target_results),
            total_units=total_expected_units,
            failed_units=len(failures),
        )

        current_normalizers = list(compiled_plan.normalizer_specs)
        normalized_results: list[NormalizedResult] = []
        if current_normalizers and target_results:
            _append_event(
                events,
                experiment_id=experiment_id,
                task_id=task.task_id,
                stage="normalization",
                entity_type="normalizers",
                entity_id="normalization_stage",
                status="started",
                payload={"normalizer_count": len(current_normalizers)},
            )
            try:
                normalized_results, normalization_failures = normalize_target_results(
                    task=task,
                    compiled_plan=compiled_plan,
                    benchmark_id=benchmark_id,
                    benchmark_manifest=compiled_plan.case_set.manifest,
                    target_results=target_results,
                    normalizers=current_normalizers,
                )
                failures.extend(normalization_failures)
                _append_event(
                    events,
                    experiment_id=experiment_id,
                    task_id=task.task_id,
                    stage="normalization",
                    entity_type="normalizers",
                    entity_id="normalization_stage",
                    status="completed",
                    payload={"normalized_result_count": len(normalized_results), "failure_count": len(normalization_failures)},
                )
            except Exception as exc:
                failed_stages.append("normalization")
                failures.append(
                    {
                        "stage": "normalization",
                        "entity_id": "normalization_stage",
                        "error": str(exc),
                    }
                )
                _append_event(
                    events,
                    experiment_id=experiment_id,
                    task_id=task.task_id,
                    stage="normalization",
                    entity_type="normalizers",
                    entity_id="normalization_stage",
                    status="failed",
                    payload={"error": str(exc)},
                )
            _emit_evaluation_eta(
                progress=progress,
                started_at=started_at,
                stage="normalization",
                processed_units=len(target_results) + len(normalized_results),
                total_units=total_expected_units,
                failed_units=len(failures),
            )

        score_records: list[ScoreRecord] = []
        current_scorers = list(compiled_plan.scorer_specs)
        if current_scorers and target_results:
            _append_event(
                events,
                experiment_id=experiment_id,
                task_id=task.task_id,
                stage="scoring",
                entity_type="scorers",
                entity_id="scoring_stage",
                status="started",
                payload={"scorer_count": len(current_scorers)},
            )
            try:
                score_records, scoring_failures = score_target_results(
                    task=task,
                    compiled_plan=compiled_plan,
                    source_run_id=target_run_ids[0] if target_run_ids else "",
                    source_run_references=target_run_refs,
                    target_results=target_results,
                    normalized_results=normalized_results,
                    scorers=current_scorers,
                    out_dir=experiment_path / "scorers",
                    execution_mode=execution_mode,
                    preview_collector=preview_collector,
                    progress=progress,
                )
                failures.extend(scoring_failures)
                _append_event(
                    events,
                    experiment_id=experiment_id,
                    task_id=task.task_id,
                    stage="scoring",
                    entity_type="scorers",
                    entity_id="scoring_stage",
                    status="completed",
                    payload={"score_record_count": len(score_records), "failure_count": len(scoring_failures)},
                )
            except Exception as exc:
                failed_stages.append("scoring")
                failures.append(
                    {
                        "stage": "scoring",
                        "entity_id": "scoring_stage",
                        "error": str(exc),
                    }
                )
                _append_event(
                    events,
                    experiment_id=experiment_id,
                    task_id=task.task_id,
                    stage="scoring",
                    entity_type="scorers",
                    entity_id="scoring_stage",
                    status="failed",
                    payload={"error": str(exc)},
                )
            _emit_evaluation_eta(
                progress=progress,
                started_at=started_at,
                stage="scoring",
                processed_units=len(target_results) + len(normalized_results) + len(score_records),
                total_units=total_expected_units,
                failed_units=len(failures),
            )

        group_analysis_records = build_group_analysis_records(
            task=task,
            compiled_plan=compiled_plan,
            target_results=target_results,
            normalized_results=normalized_results,
            score_records=score_records,
        )
        _append_event(
            events,
            experiment_id=experiment_id,
            task_id=task.task_id,
            stage="group_analysis",
            entity_type="group_analyzers",
            entity_id="group_analysis_stage",
            status="completed",
            payload={"group_analysis_record_count": len(group_analysis_records)},
        )
        _emit_evaluation_eta(
            progress=progress,
            started_at=started_at,
            stage="group_analysis",
            processed_units=len(target_results) + len(normalized_results) + len(score_records),
            total_units=total_expected_units,
            failed_units=len(failures),
        )

        analysis_plugin_specs = list(compiled_plan.plugin_specs)
        analysis_plugin_results: list[AnalysisPluginResult] = []
        if analysis_plugin_specs:
            _append_event(
                events,
                experiment_id=experiment_id,
                task_id=task.task_id,
                stage="analysis_plugin",
                entity_type="plugins",
                entity_id="analysis_plugin_stage",
                status="started",
                payload={"plugin_count": len(analysis_plugin_specs)},
            )
            try:
                analysis_plugin_results, analysis_failures = run_analysis_plugins(
                    task=task,
                    compiled_plan=compiled_plan,
                    benchmark_id=benchmark_id,
                    benchmark_manifest=compiled_plan.case_set.manifest,
                    cases=compiled_plan.case_set.cases,
                    target_results=target_results,
                    normalized_results=normalized_results,
                    score_records=score_records,
                    plugin_specs=analysis_plugin_specs,
                )
                failures.extend(analysis_failures)
                _append_event(
                    events,
                    experiment_id=experiment_id,
                    task_id=task.task_id,
                    stage="analysis_plugin",
                    entity_type="plugins",
                    entity_id="analysis_plugin_stage",
                    status="completed",
                    payload={"analysis_plugin_result_count": len(analysis_plugin_results), "failure_count": len(analysis_failures)},
                )
            except Exception as exc:
                failed_stages.append("analysis_plugin")
                failures.append(
                    {
                        "stage": "analysis_plugin",
                        "entity_id": "analysis_plugin_stage",
                        "error": str(exc),
                    }
                )
                _append_event(
                    events,
                    experiment_id=experiment_id,
                    task_id=task.task_id,
                    stage="analysis_plugin",
                    entity_type="plugins",
                    entity_id="analysis_plugin_stage",
                    status="failed",
                    payload={"error": str(exc)},
                )
            _emit_evaluation_eta(
                progress=progress,
                started_at=started_at,
                stage="analysis_plugin",
                processed_units=total_expected_units,
                total_units=total_expected_units,
                failed_units=len(failures),
            )

        available_layers = ["cases", "execution_requests", "target_results"]
        if normalized_results:
            available_layers.append("normalized_results")
        if score_records:
            available_layers.append("score_records")
        if group_analysis_records:
            available_layers.append("group_analysis_records")
        if analysis_plugin_results:
            available_layers.append("analysis_plugin_results")
        bundle_completeness = "partial" if failed_stages else "complete"
        summary = build_summary_report(
            benchmark_id=benchmark_id,
            benchmark_path=compiled_plan.case_set.case_set_path or "",
            target_models=task.target_models,
            normalizer_ids=[str(item.get("normalizer_id", "")) for item in current_normalizers],
            scorer_ids=[str(item.get("scorer_id", item.get("evaluator_id", ""))) for item in current_scorers],
            analysis_plugin_ids=[
                item.plugin_id if isinstance(item, AnalysisPluginSpec) else str(item.get("plugin_id", ""))
                for item in analysis_plugin_specs
            ],
            target_results=target_results,
            normalized_results=normalized_results,
            score_records=score_records,
            group_analysis_records=group_analysis_records,
            analysis_plugin_results=analysis_plugin_results,
            failures=failures,
            bundle_completeness=bundle_completeness,
            available_layers=available_layers,
            failed_stages=failed_stages,
        )
        report_markdown = render_experiment_report(
            benchmark_id=benchmark_id,
            summary=summary,
            group_analysis_records=group_analysis_records,
        )
        selection_result = compiled_plan.case_selection_result
        selection_manifest_payload = None
        excluded_case_rows: list[dict[str, Any]] = []
        if selection_result is not None:
            selection_manifest_payload = {
                **selection_result.selection_manifest,
                "source_benchmark_path": compiled_plan.case_set.case_set_path,
                "source_benchmark_id": benchmark_id,
                "selected_case_count": selection_result.counts_after,
                "excluded_case_count": len(selection_result.excluded_cases),
            }
            excluded_case_rows = [case.to_dict() for case in selection_result.excluded_cases]
        bundle_paths = write_experiment_bundle(
            experiment_dir=experiment_path,
            manifest=ExperimentBundleManifest(
                experiment_id=experiment_id,
                task_id=task.task_id,
                task_version=task.task_version,
                benchmark_id=benchmark_id,
                benchmark_path=compiled_plan.case_set.case_set_path or "",
                target_models=list(task.target_models),
                normalizer_ids=[str(item.get("normalizer_id", "")) for item in current_normalizers],
                scorer_ids=[str(item.get("scorer_id", item.get("evaluator_id", ""))) for item in current_scorers],
                analysis_plugin_ids=[
                    item.plugin_id if isinstance(item, AnalysisPluginSpec) else str(item.get("plugin_id", ""))
                    for item in analysis_plugin_specs
                ],
                execution_mode=execution_mode,
                case_count=len(compiled_plan.case_set.cases),
                failure_count=len(failures),
                created_at=datetime.now(UTC).isoformat(),
                target_run_ids=target_run_ids,
                recipe_path=str(task.metadata.get("recipe_path")) if task.metadata.get("recipe_path") not in (None, "") else None,
                auto_launch=bool(task.execution_policy.get("auto_launch", False)),
                launch_plan=launch_plan.to_dict(),
                selection_applied=selection_result is not None,
                selection_spec=selection_result.spec.to_dict() if selection_result is not None else {},
                selected_case_count=selection_result.counts_after if selection_result is not None else None,
                source_case_count=selection_result.counts_before if selection_result is not None else None,
                excluded_case_count=len(selection_result.excluded_cases) if selection_result is not None else 0,
                bundle_completeness=bundle_completeness,
                available_layers=available_layers,
                failed_stages=list(failed_stages),
            ).to_dict(),
            case_set=compiled_plan.case_set,
            compiled_task_plan=compiled_plan,
            execution_requests=compiled_plan.execution_requests,
            target_results=target_results,
            normalized_results=normalized_results,
            score_records=score_records,
            group_analysis_records=group_analysis_records,
            analysis_plugin_results=analysis_plugin_results,
            experiment_log=events,
            summary=summary.to_dict(),
            report_markdown=report_markdown,
            failures=failures,
        )
        if selection_manifest_payload is not None:
            selection_manifest_path = experiment_path / "selection_manifest.json"
            selection_manifest_path.write_text(
                json.dumps(selection_manifest_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            excluded_cases_path = experiment_path / "excluded_cases.jsonl"
            with excluded_cases_path.open("w", encoding="utf-8") as handle:
                for row in excluded_case_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_path = Path(bundle_paths["manifest_path"])
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["selection_manifest_path"] = str(selection_manifest_path)
            manifest_payload["excluded_cases_path"] = str(excluded_cases_path)
            manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        preview_summary = None
        if preview_collector is not None and preview_collector.records:
            preview_summary = write_request_preview_bundle(
                preview_dir=experiment_path,
                bundle=preview_collector.to_bundle(),
                preview_only=False,
                preview_stage=str(preview_config.get("preview_stage") or "all"),
                preview_count=int(preview_config.get("preview_count", 5)),
            )
            manifest_path = Path(bundle_paths["manifest_path"])
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["request_previews_path"] = preview_summary.request_previews_path
            manifest_payload["request_preview_summary_path"] = preview_summary.request_preview_summary_path
            manifest_payload["request_previews_markdown_path"] = preview_summary.request_previews_markdown_path
            manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return EvaluationExperimentSummary(
            experiment_id=experiment_id,
            experiment_dir=str(experiment_path),
            task_id=task.task_id,
            benchmark_id=benchmark_id,
            benchmark_path=compiled_plan.case_set.case_set_path or "",
            target_models=list(task.target_models),
            normalizer_ids=[str(item.get("normalizer_id", "")) for item in current_normalizers],
            scorer_ids=[str(item.get("scorer_id", item.get("evaluator_id", ""))) for item in current_scorers],
            analysis_plugin_ids=[
                item.plugin_id if isinstance(item, AnalysisPluginSpec) else str(item.get("plugin_id", ""))
                for item in analysis_plugin_specs
            ],
            execution_mode=execution_mode,
            case_count=len(compiled_plan.case_set.cases),
            target_run_count=len(target_run_ids),
            normalized_result_count=len(normalized_results),
            score_record_count=len(score_records),
            group_analysis_record_count=len(group_analysis_records),
            analysis_plugin_result_count=len(analysis_plugin_results),
            source_case_count=selection_result.counts_before if selection_result is not None else None,
            excluded_case_count=len(selection_result.excluded_cases) if selection_result is not None else 0,
            execution_requests_path=bundle_paths["execution_requests_path"],
            target_results_path=bundle_paths["target_results_path"],
            normalized_results_path=bundle_paths["normalized_results_path"],
            score_records_path=bundle_paths["score_records_path"],
            group_analysis_records_path=bundle_paths["group_analysis_records_path"],
            analysis_plugin_results_path=bundle_paths["analysis_plugin_results_path"],
            experiment_log_path=bundle_paths["experiment_log_path"],
            compiled_task_plan_path=bundle_paths["compiled_task_plan_path"],
            manifest_path=bundle_paths["manifest_path"],
            summary_path=bundle_paths["summary_path"],
            report_path=bundle_paths["report_path"],
            failures_path=bundle_paths["failures_path"],
            selection_manifest_path=str(experiment_path / "selection_manifest.json") if selection_result is not None else None,
            excluded_cases_path=str(experiment_path / "excluded_cases.jsonl") if selection_result is not None else None,
            request_previews_path=preview_summary.request_previews_path if preview_summary is not None else None,
            request_preview_summary_path=preview_summary.request_preview_summary_path if preview_summary is not None else None,
            request_previews_markdown_path=preview_summary.request_previews_markdown_path if preview_summary is not None else None,
            bundle_completeness=bundle_completeness,
            available_layers=available_layers,
            failed_stages=list(failed_stages),
        )


def build_group_analysis_records(
    *,
    task: EvalTask,
    compiled_plan: CompiledTaskPlan,
    target_results: list[TargetResult],
    normalized_results: list[NormalizedResult],
    score_records: list[ScoreRecord],
) -> list[GroupAnalysisRecord]:
    grouped_target_results: dict[tuple[str, str, str], list[TargetResult]] = defaultdict(list)
    for result in target_results:
        group_key = str(
            result.metadata.get("group_key")
            or result.metadata.get("variant_group_id")
            or result.metadata.get("family_id")
            or result.metadata.get("template_id")
            or result.prompt_metadata.get("group_key")
            or result.prompt_metadata.get("variant_group_id")
            or result.prompt_metadata.get("family_id")
            or result.prompt_metadata.get("template_id")
            or "default"
        )
        grouped_target_results[(result.target_model, result.split, group_key)].append(result)

    analyses: list[GroupAnalysisRecord] = []
    for (target_model, split, group_key), items in sorted(grouped_target_results.items()):
        scorer_groups: dict[str, list[ScoreRecord]] = defaultdict(list)
        for item in items:
            for score_record in score_records:
                if score_record.target_model != target_model:
                    continue
                if score_record.case_id != item.case_id:
                    continue
                scorer_groups[score_record.scorer_id].append(score_record)
        if not scorer_groups:
            analyses.append(
                GroupAnalysisRecord(
                    task_id=task.task_id,
                    benchmark_id=compiled_plan.case_set.benchmark_id,
                    analysis_type="generic_group_summary",
                    target_model=target_model,
                    split=split,
                    group_key=group_key,
                    scorer_id=None,
                    case_count=len(items),
                    score_record_count=0,
                    output={},
                )
            )
            continue
        for scorer_id, rows in sorted(scorer_groups.items()):
            label_counter = Counter()
            score_totals: dict[str, float] = defaultdict(float)
            score_counts: dict[str, int] = defaultdict(int)
            for row in rows:
                for label in row.labels:
                    label_counter[label] += 1
                for key, value in row.scores.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        score_totals[key] += float(value)
                        score_counts[key] += 1
            analyses.append(
                GroupAnalysisRecord(
                    task_id=task.task_id,
                    benchmark_id=compiled_plan.case_set.benchmark_id,
                    analysis_type="generic_group_summary",
                    target_model=target_model,
                    split=split,
                    group_key=group_key,
                    scorer_id=scorer_id,
                    case_count=len(items),
                    score_record_count=len(rows),
                    labels=list(sorted(label_counter)),
                    scores={
                        key: (score_totals[key] / score_counts[key])
                        for key in sorted(score_totals)
                        if score_counts[key] > 0
                    },
                    output={"label_counts": dict(sorted(label_counter.items()))},
                )
            )
    return analyses


def build_summary_report(
    *,
    benchmark_id: str,
    benchmark_path: str | Path,
    target_models: list[str],
    normalizer_ids: list[str],
    scorer_ids: list[str],
    analysis_plugin_ids: list[str],
    target_results: list[TargetResult],
    normalized_results: list[NormalizedResult],
    score_records: list[ScoreRecord],
    group_analysis_records: list[GroupAnalysisRecord],
    analysis_plugin_results: list[AnalysisPluginResult],
    failures: list[dict[str, Any]],
    bundle_completeness: str = "complete",
    available_layers: list[str] | None = None,
    failed_stages: list[str] | None = None,
) -> SummaryReport:
    counts_by_target_model = Counter(result.target_model for result in target_results)
    counts_by_normalizer = Counter(result.normalizer_id for result in normalized_results)
    counts_by_scorer = Counter(result.scorer_id for result in score_records)
    counts_by_analysis_plugin = Counter(result.plugin_id for result in analysis_plugin_results)
    counts_by_split = Counter(result.split for result in target_results)
    label_counts_by_target: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_score_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    numeric_score_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in score_records:
        for label in result.labels:
            label_counts_by_target[result.target_model][label] += 1
        for key, value in result.scores.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_score_totals[result.scorer_id][key] += float(value)
                numeric_score_counts[result.scorer_id][key] += 1
    return SummaryReport(
        benchmark_id=benchmark_id,
        benchmark_path=str(benchmark_path),
        target_models=list(target_models),
        normalizer_ids=list(normalizer_ids),
        scorer_ids=list(scorer_ids),
        analysis_plugin_ids=list(analysis_plugin_ids),
        case_count=len({result.case_id for result in target_results}),
        target_result_count=len(target_results),
        normalized_result_count=len(normalized_results),
        score_record_count=len(score_records),
        group_analysis_record_count=len(group_analysis_records),
        analysis_plugin_result_count=len(analysis_plugin_results),
        failure_count=len(failures),
        counts_by_target_model=dict(sorted(counts_by_target_model.items())),
        counts_by_normalizer=dict(sorted(counts_by_normalizer.items())),
        counts_by_scorer=dict(sorted(counts_by_scorer.items())),
        counts_by_analysis_plugin=dict(sorted(counts_by_analysis_plugin.items())),
        counts_by_split=dict(sorted(counts_by_split.items())),
        label_counts_by_target_model={
            model: dict(sorted(counter.items()))
            for model, counter in sorted(label_counts_by_target.items())
        },
        average_numeric_scores_by_scorer={
            scorer_id: {
                score_name: (numeric_score_totals[scorer_id][score_name] / numeric_score_counts[scorer_id][score_name])
                for score_name in sorted(numeric_score_totals[scorer_id])
                if numeric_score_counts[scorer_id][score_name] > 0
            }
            for scorer_id in sorted(numeric_score_totals)
        },
        bundle_completeness=bundle_completeness,
        available_layers=list(available_layers or []),
        failed_stages=list(failed_stages or []),
    )


def render_experiment_report(
    *,
    benchmark_id: str,
    summary: SummaryReport,
    group_analysis_records: list[GroupAnalysisRecord],
) -> str:
    payload = summary.to_dict()
    lines = [
        f"# Experiment Report — {benchmark_id}",
        "",
        f"- Benchmark ID: {benchmark_id}",
        f"- Case Count: {payload.get('case_count', 0)}",
        f"- Target Result Count: {payload.get('target_result_count', 0)}",
        f"- Normalized Result Count: {payload.get('normalized_result_count', 0)}",
        f"- Score Record Count: {payload.get('score_record_count', 0)}",
        f"- Group Analysis Record Count: {payload.get('group_analysis_record_count', 0)}",
        f"- Analysis Plugin Result Count: {payload.get('analysis_plugin_result_count', 0)}",
        f"- Bundle Completeness: {payload.get('bundle_completeness', 'complete')}",
        f"- Available Layers: {', '.join(payload.get('available_layers', [])) or '-'}",
        "",
        "## Per-Target Counts",
        "",
        "| Target | Results |",
        "| --- | ---: |",
    ]
    failed_stages = list(payload.get("failed_stages", []) or [])
    if failed_stages:
        lines.extend(
            [
                "",
                "## Failed Stages",
                "",
                *[f"- {stage}" for stage in failed_stages],
            ]
        )
    for target_model, count in sorted(payload.get("counts_by_target_model", {}).items()):
        lines.append(f"| {target_model} | {count} |")
    lines.extend(
        [
            "",
            "## Per-Normalizer Counts",
            "",
            "| Normalizer | Results |",
            "| --- | ---: |",
        ]
    )
    for normalizer_id, count in sorted(payload.get("counts_by_normalizer", {}).items()):
        lines.append(f"| {normalizer_id} | {count} |")
    lines.extend(
        [
            "",
            "## Per-Scorer Counts",
            "",
            "| Scorer | Results |",
            "| --- | ---: |",
        ]
    )
    for scorer_id, count in sorted(payload.get("counts_by_scorer", {}).items()):
        lines.append(f"| {scorer_id} | {count} |")
    lines.extend(
        [
            "",
            "## Analysis Plugin Counts",
            "",
            "| Plugin | Results |",
            "| --- | ---: |",
        ]
    )
    for plugin_id, count in sorted(payload.get("counts_by_analysis_plugin", {}).items()):
        lines.append(f"| {plugin_id} | {count} |")
    lines.extend(
        [
            "",
            "## Group Analysis Records",
            "",
            "| Target | Split | Group | Scorer | Cases | Labels |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in group_analysis_records:
        lines.append(
            f"| {row.target_model or '-'} | {row.split or '-'} | "
            f"{row.group_key} | {row.scorer_id or '-'} | "
            f"{row.case_count} | {json.dumps(row.output.get('label_counts', {}), ensure_ascii=False)} |"
        )
    return "\n".join(lines) + "\n"


def _append_event(
    events: list[ExperimentLogEvent],
    *,
    experiment_id: str,
    task_id: str,
    stage: str,
    entity_type: str,
    entity_id: str,
    status: str,
    payload: dict[str, Any],
) -> None:
    events.append(
        ExperimentLogEvent(
            event_id=f"evt_{len(events) + 1:08d}",
            timestamp=datetime.now(UTC).isoformat(),
            experiment_id=experiment_id,
            task_id=task_id,
            stage=stage,
            entity_type=entity_type,
            entity_id=entity_id,
            status=status,
            payload=dict(payload),
        )
    )


def _estimate_total_evaluation_units(
    *,
    execution_request_count: int,
    normalizer_count: int,
    scorer_count: int,
    plugin_count: int,
) -> int:
    total = max(int(execution_request_count), 0)
    total += max(int(execution_request_count), 0) * max(int(normalizer_count), 0)
    total += max(int(execution_request_count), 0) * max(int(scorer_count), 0)
    total += max(int(plugin_count), 0)
    return max(total, max(int(execution_request_count), 1))


def _emit_evaluation_eta(
    *,
    progress: RunProgress,
    started_at: float,
    stage: str,
    processed_units: int,
    total_units: int,
    failed_units: int,
) -> None:
    processed = max(int(processed_units), 0)
    total = max(int(total_units), 1)
    elapsed_sec = max(time.monotonic() - started_at, 0.0)
    rate_per_min = 0.0 if processed <= 0 or elapsed_sec <= 0 else processed / (elapsed_sec / 60.0)
    parts = [
        "[THROUGHPUT]",
        "scope=evaluate",
        f"stage={stage}",
        f"prompts={processed}/{total}",
        f"rate={rate_per_min:.1f}/min",
        f"failed={max(int(failed_units), 0)}",
    ]
    eta = _format_eta_from_rate(processed=processed, total=total, rate_per_min=rate_per_min)
    if eta is not None:
        parts.append(f"eta={eta}")
    progress.env_message(" ".join(parts))


def _format_eta_from_rate(*, processed: int, total: int, rate_per_min: float) -> str | None:
    remaining = max(int(total) - int(processed), 0)
    if processed <= 0 or remaining <= 0 or rate_per_min <= 0:
        return None
    total_sec = int(round((remaining / rate_per_min) * 60.0))
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    return "".join(char if char.isalnum() else "_" for char in lowered).strip("_") or "item"
