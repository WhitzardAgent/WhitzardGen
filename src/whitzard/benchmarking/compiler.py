from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.benchmarking.bundle import build_benchmark_stats, inspect_benchmark_bundle, load_benchmark_cases
from whitzard.benchmarking.interfaces import TaskCompiler
from whitzard.benchmarking.models import (
    CaseSet,
    CaseSourceRef,
    CompiledTaskPlan,
    EvalTask,
    ExecutionRequest,
)
from whitzard.benchmarking.resolution import (
    resolve_runtime_analysis_plugins,
    resolve_runtime_analyzers,
    resolve_runtime_normalizers,
    resolve_runtime_scorers,
)


class BenchmarkingError(RuntimeError):
    """Raised when benchmark build or evaluation fails."""


class DefaultTaskCompiler(TaskCompiler):
    def __init__(
        self,
        *,
        normalizer_config_path: str | Path | None = None,
        scorer_config_path: str | Path | None = None,
        analysis_config_path: str | Path | None = None,
    ) -> None:
        self.normalizer_config_path = normalizer_config_path
        self.scorer_config_path = scorer_config_path
        self.analysis_config_path = analysis_config_path

    def compile(self, task: EvalTask) -> CompiledTaskPlan:
        case_set = _load_case_set_for_task(task)
        execution_requests = _build_execution_requests(task=task, case_set=case_set)
        normalizers = resolve_runtime_normalizers(
            normalizer_ids=task.normalizer_ids,
            normalizer_config_path=self.normalizer_config_path,
        )
        scorers = resolve_runtime_scorers(
            scorer_ids=task.scorer_ids,
            scorer_model=None,
            scorer_profile=None,
            scorer_template=None,
            scorer_config_path=self.scorer_config_path,
        )
        analyzers = resolve_runtime_analyzers(
            benchmark_manifest=case_set.manifest,
        )
        plugins = resolve_runtime_analysis_plugins(
            analysis_plugin_ids=task.plugin_ids,
            analysis_config_path=self.analysis_config_path,
        )
        return CompiledTaskPlan(
            task=task,
            case_set=case_set,
            execution_requests=execution_requests,
            normalizer_specs=[item.to_dict() for item in normalizers],
            scorer_specs=[item.to_dict() for item in scorers],
            analyzer_specs=analyzers,
            plugin_specs=[item.to_dict() for item in plugins],
            failure_policy=dict(task.execution_policy.get("failure_policy", {}) or {}),
            execution_defaults=dict(task.execution_policy),
        )


def _load_case_set_for_task(task: EvalTask) -> CaseSet:
    if not task.case_set_path:
        raise BenchmarkingError(f"EvalTask {task.task_id} is missing case_set_path.")
    benchmark_path = Path(task.case_set_path)
    payload = inspect_benchmark_bundle(benchmark_path)
    manifest = dict(payload.get("manifest") or {})
    bundle_dir = benchmark_path.parent if benchmark_path.is_file() else benchmark_path
    cases_path = bundle_dir / "cases.jsonl"
    cases = load_benchmark_cases(cases_path)
    if not cases:
        raise BenchmarkingError(f"No benchmark cases were found at {cases_path}")
    benchmark_id = str(manifest.get("benchmark_id") or bundle_dir.name)
    source = task.case_source or CaseSourceRef(
        source_type="benchmark_bundle",
        source_path=str(bundle_dir),
        builder_name=str(manifest.get("builder_name") or manifest.get("source_builder") or ""),
        metadata={"build_mode": manifest.get("build_mode")},
    )
    return CaseSet(
        benchmark_id=benchmark_id,
        cases=cases,
        source=source,
        manifest=manifest,
        stats=build_benchmark_stats(cases),
        case_set_path=str(cases_path),
    )


def _build_execution_requests(*, task: EvalTask, case_set: CaseSet) -> list[ExecutionRequest]:
    requests: list[ExecutionRequest] = []
    text_prompt_composition = dict(task.execution_policy.get("text_prompt_composition", {}) or {})
    target_prompt_template = dict(task.execution_policy.get("target_prompt_template", {}) or {})
    for target_model in task.target_models:
        for case in case_set.cases:
            request_id = f"{task.task_id}:{target_model}:{case.case_id}"
            metadata: dict[str, Any] = {
                "benchmark_id": case.benchmark_id,
                "case_id": case.case_id,
                "case_version": case.case_version,
                "source_builder": case.source_builder,
                "split": case.split,
                "tags": list(case.tags),
                "grouping": dict(case.grouping),
                "execution_hints": dict(case.execution_hints),
                "evaluation_hints": dict(case.evaluation_hints),
                "case_metadata": dict(case.metadata),
                "prompt_composition": text_prompt_composition,
                "prompt_template": target_prompt_template,
            }
            requests.append(
                ExecutionRequest(
                    task_id=task.task_id,
                    benchmark_id=case.benchmark_id,
                    case_id=case.case_id,
                    request_id=request_id,
                    target_model=target_model,
                    input_modality=case.input_modality,
                    input_payload=dict(case.input_payload),
                    generation_params=dict(case.parameters),
                    expected_output_contract=case.expected_output_contract,
                    metadata=metadata,
                    runtime_hints=dict(case.execution_hints),
                )
            )
    return requests
