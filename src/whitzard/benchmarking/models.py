from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CaseSourceRef:
    source_type: str
    source_path: str | None = None
    builder_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkCase:
    benchmark_id: str
    case_id: str
    input_modality: str = "text"
    input_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    split: str = "default"
    expected_output_contract: dict[str, Any] | list[Any] | str | None = None
    expected_structure: dict[str, Any] | list[Any] | str | None = None
    case_version: str | None = None
    source_builder: str | None = None
    grouping: dict[str, str] = field(default_factory=dict)
    execution_hints: dict[str, Any] = field(default_factory=dict)
    evaluation_hints: dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    parameters: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None
    instruction: str | None = None
    context: dict[str, Any] | str | None = None
    input_type: str | None = None

    def __post_init__(self) -> None:
        if not self.input_modality and self.input_type:
            self.input_modality = str(self.input_type)
        if not self.input_modality:
            self.input_modality = "text"
        if self.input_type in (None, ""):
            self.input_type = self.input_modality

        if not self.input_payload:
            payload: dict[str, Any] = {}
            if self.prompt not in (None, ""):
                payload["prompt"] = self.prompt
            if self.instruction not in (None, ""):
                payload["instruction"] = self.instruction
            if self.context not in (None, ""):
                payload["context"] = self.context
            if self.parameters:
                payload["parameters"] = dict(self.parameters)
            self.input_payload = payload
        else:
            self.input_payload = dict(self.input_payload)

        if self.expected_output_contract in (None, "") and self.expected_structure not in (None, ""):
            self.expected_output_contract = self.expected_structure
        if self.expected_structure in (None, "") and self.expected_output_contract not in (None, ""):
            self.expected_structure = self.expected_output_contract

        if self.prompt in (None, ""):
            prompt_value = self.input_payload.get("prompt")
            if prompt_value not in (None, ""):
                self.prompt = str(prompt_value)
        if self.instruction in (None, ""):
            instruction_value = self.input_payload.get("instruction")
            if instruction_value not in (None, ""):
                self.instruction = str(instruction_value)
        if self.context in (None, "") and "context" in self.input_payload:
            self.context = self.input_payload.get("context")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "case_id": self.case_id,
            "case_version": self.case_version,
            "input_modality": self.input_modality,
            "input_payload": dict(self.input_payload),
            "expected_output_contract": self.expected_output_contract,
            "expected_structure": self.expected_structure,
            "metadata": dict(self.metadata),
            "tags": list(self.tags),
            "split": self.split,
            "source_builder": self.source_builder,
            "grouping": dict(self.grouping),
            "execution_hints": dict(self.execution_hints),
            "evaluation_hints": dict(self.evaluation_hints),
            "language": self.language,
            "parameters": dict(self.parameters),
            "prompt": self.prompt,
            "instruction": self.instruction,
            "context": self.context,
        }

    @property
    def rendered_input(self) -> str:
        if self.prompt not in (None, ""):
            return str(self.prompt).strip()
        if self.instruction not in (None, ""):
            return str(self.instruction).strip()
        for key in ("prompt", "instruction", "text", "input"):
            value = self.input_payload.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""


@dataclass(slots=True)
class RealizationSpec:
    benchmark_id: str
    case_id: str
    source_builder: str
    input_modality: str = "text"
    language: str = "en"
    split: str = "default"
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    grouping: dict[str, str] = field(default_factory=dict)
    execution_hints: dict[str, Any] = field(default_factory=dict)
    evaluation_hints: dict[str, Any] = field(default_factory=dict)
    expected_output_contract: dict[str, Any] | list[Any] | str | None = None
    slot_assignments: dict[str, Any] = field(default_factory=dict)
    invariants: list[str] = field(default_factory=list)
    forbidden_transformations: list[str] = field(default_factory=list)
    prompt_template_name: str | None = None
    prompt_template_version: str | None = None
    synthesis_model: str | None = None
    synthesis_request_version: str | None = None
    prompt_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RealizationResult:
    benchmark_id: str
    case_id: str
    source_builder: str
    synthesized_text: str
    scene_description: str | None = None
    structured_output: dict[str, Any] = field(default_factory=dict)
    decision_frame: dict[str, Any] = field(default_factory=dict)
    decision_options: list[dict[str, Any]] = field(default_factory=list)
    prompt_template_name: str | None = None
    prompt_template_version: str | None = None
    synthesis_model: str | None = None
    synthesis_request_version: str | None = None
    request_prompt: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.validation_errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RealizationValidationResult:
    benchmark_id: str
    case_id: str
    valid: bool
    issues: list[str] = field(default_factory=list)
    feedback_for_retry: list[str] = field(default_factory=list)
    binary_frame_assessment: dict[str, Any] = field(default_factory=dict)
    conflict_preservation_assessment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaseSet:
    benchmark_id: str
    cases: list[BenchmarkCase]
    source: CaseSourceRef
    manifest: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    case_set_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "source": self.source.to_dict(),
            "manifest": dict(self.manifest),
            "stats": dict(self.stats),
            "case_set_path": self.case_set_path,
            "case_count": len(self.cases),
        }


@dataclass(slots=True)
class CaseSelectionSpec:
    seed: int = 42
    group_selector: str | None = None
    sample_size_per_group: int | None = None
    undersized_group_policy: str = "keep_all_warn"
    include_groups: list[str] = field(default_factory=list)
    exclude_groups: list[str] = field(default_factory=list)
    include_case_ids: list[str] = field(default_factory=list)
    exclude_case_ids: list[str] = field(default_factory=list)
    split_filter: list[str] = field(default_factory=list)
    tag_filter: list[str] = field(default_factory=list)
    max_cases: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaseSelectionResult:
    spec: CaseSelectionSpec
    selected_cases: list[BenchmarkCase] = field(default_factory=list)
    excluded_cases: list[BenchmarkCase] = field(default_factory=list)
    counts_before: int = 0
    counts_after: int = 0
    counts_by_group_before: dict[str, int] = field(default_factory=dict)
    counts_by_group_after: dict[str, int] = field(default_factory=dict)
    undersized_groups: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selection_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "counts_before": self.counts_before,
            "counts_after": self.counts_after,
            "counts_by_group_before": dict(self.counts_by_group_before),
            "counts_by_group_after": dict(self.counts_by_group_after),
            "undersized_groups": list(self.undersized_groups),
            "warnings": list(self.warnings),
            "selection_manifest": dict(self.selection_manifest),
            "selected_case_ids": [case.case_id for case in self.selected_cases],
            "excluded_case_ids": [case.case_id for case in self.excluded_cases],
        }


@dataclass(slots=True)
class EvalTask:
    task_id: str
    task_version: str = "v2"
    case_source: CaseSourceRef | None = None
    case_set_path: str | None = None
    target_models: list[str] = field(default_factory=list)
    case_selection: CaseSelectionSpec | None = None
    execution_policy: dict[str, Any] = field(default_factory=dict)
    normalizer_ids: list[str] = field(default_factory=list)
    scorer_ids: list[str] = field(default_factory=list)
    analyzer_refs: list[dict[str, Any]] = field(default_factory=list)
    plugin_ids: list[str] = field(default_factory=list)
    output_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "case_source": self.case_source.to_dict() if self.case_source is not None else None,
            "case_set_path": self.case_set_path,
            "target_models": list(self.target_models),
            "case_selection": self.case_selection.to_dict() if self.case_selection is not None else None,
            "execution_policy": dict(self.execution_policy),
            "normalizer_ids": list(self.normalizer_ids),
            "scorer_ids": list(self.scorer_ids),
            "analyzer_refs": list(self.analyzer_refs),
            "plugin_ids": list(self.plugin_ids),
            "output_policy": dict(self.output_policy),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ExecutionRequest:
    task_id: str
    benchmark_id: str
    case_id: str
    request_id: str
    target_model: str
    input_modality: str
    input_payload: dict[str, Any]
    generation_params: dict[str, Any] = field(default_factory=dict)
    expected_output_contract: dict[str, Any] | list[Any] | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompiledTaskPlan:
    task: EvalTask
    case_set: CaseSet
    execution_requests: list[ExecutionRequest]
    case_selection_result: CaseSelectionResult | None = None
    normalizer_specs: list[dict[str, Any]] = field(default_factory=list)
    scorer_specs: list[dict[str, Any]] = field(default_factory=list)
    analyzer_specs: list[dict[str, Any]] = field(default_factory=list)
    plugin_specs: list[dict[str, Any]] = field(default_factory=list)
    failure_policy: dict[str, Any] = field(default_factory=dict)
    execution_defaults: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "case_set": self.case_set.to_dict(),
            "execution_requests": [item.to_dict() for item in self.execution_requests],
            "case_selection_result": self.case_selection_result.to_dict() if self.case_selection_result is not None else None,
            "normalizer_specs": list(self.normalizer_specs),
            "scorer_specs": list(self.scorer_specs),
            "analyzer_specs": list(self.analyzer_specs),
            "plugin_specs": list(self.plugin_specs),
            "failure_policy": dict(self.failure_policy),
            "execution_defaults": dict(self.execution_defaults),
        }


@dataclass(slots=True)
class TargetResult:
    task_id: str
    request_id: str
    benchmark_id: str
    case_id: str
    case_version: str | None
    source_builder: str | None
    target_model: str
    source_run_id: str
    source_record_id: str
    input_modality: str
    split: str
    tags: list[str]
    artifact_type: str
    artifact_path: str
    prompt: str
    execution_status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    artifact_metadata: dict[str, Any] = field(default_factory=dict)
    generation_params: dict[str, Any] = field(default_factory=dict)
    runtime_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def input_type(self) -> str:
        return self.input_modality


@dataclass(slots=True)
class NormalizedResult:
    task_id: str
    benchmark_id: str
    case_id: str
    case_version: str | None
    request_id: str
    target_model: str
    normalizer_id: str
    status: str
    split: str = "default"
    tags: list[str] = field(default_factory=list)
    source_record_id: str | None = None
    decision_text: str | None = None
    refusal_flag: bool | None = None
    confidence_signal: float | str | None = None
    reasoning_trace_text: str | None = None
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    raw_normalized: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoreRecord:
    task_id: str
    benchmark_id: str
    case_id: str
    case_version: str | None
    request_id: str
    target_model: str
    scorer_id: str
    status: str
    labels: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None
    raw_judgment: Any = None
    scorer_metadata: dict[str, Any] = field(default_factory=dict)
    split: str = "default"
    tags: list[str] = field(default_factory=list)
    source_record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EvaluatorResult = ScoreRecord


@dataclass(slots=True)
class GroupAnalysisRecord:
    task_id: str
    benchmark_id: str
    analysis_type: str
    group_key: str
    target_model: str | None = None
    split: str | None = None
    scorer_id: str | None = None
    status: str = "success"
    case_count: int = 0
    score_record_count: int = 0
    labels: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisPluginResult:
    benchmark_id: str
    plugin_id: str
    plugin_version: str
    plugin_type: str
    task_id: str | None = None
    target_model: str | None = None
    case_id: str | None = None
    split: str | None = None
    status: str = "success"
    labels: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentLogEvent:
    event_id: str
    timestamp: str
    experiment_id: str
    task_id: str
    stage: str
    entity_type: str
    entity_id: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RequestPreviewRecord:
    stage: str
    entity_id: str
    case_id: str | None = None
    request_id: str | None = None
    target_model: str | None = None
    judge_model: str | None = None
    template_name: str | None = None
    template_version: str | None = None
    rendered_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RequestPreviewBundle:
    records: list[RequestPreviewRecord] = field(default_factory=list)
    counts_by_stage: dict[str, int] = field(default_factory=dict)
    source_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [item.to_dict() for item in self.records],
            "counts_by_stage": dict(self.counts_by_stage),
            "source_context": dict(self.source_context),
        }


@dataclass(slots=True)
class PreviewSummary:
    preview_dir: str
    request_previews_path: str
    request_preview_summary_path: str
    request_previews_markdown_path: str | None
    preview_only: bool
    preview_stage: str
    preview_count: int
    counts_by_stage: dict[str, int] = field(default_factory=dict)
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    source_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkBuildSummary:
    benchmark_id: str
    benchmark_dir: str
    builder_name: str
    source_path: str
    case_set_path: str
    manifest_path: str
    stats_path: str
    case_count: int
    build_mode: str
    raw_realizations_path: str | None = None
    rejected_realizations_path: str | None = None
    selection_manifest_path: str | None = None
    excluded_cases_path: str | None = None
    source_case_count: int | None = None
    excluded_case_count: int = 0
    request_previews_path: str | None = None
    request_preview_summary_path: str | None = None
    request_previews_markdown_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cases_path(self) -> str:
        return self.case_set_path


@dataclass(slots=True)
class EvaluationExperimentSummary:
    experiment_id: str
    experiment_dir: str
    task_id: str
    benchmark_id: str
    benchmark_path: str
    target_models: list[str]
    normalizer_ids: list[str]
    scorer_ids: list[str]
    analysis_plugin_ids: list[str]
    execution_mode: str
    case_count: int
    target_run_count: int
    normalized_result_count: int
    score_record_count: int
    group_analysis_record_count: int
    analysis_plugin_result_count: int
    execution_requests_path: str
    target_results_path: str
    normalized_results_path: str | None
    score_records_path: str
    group_analysis_records_path: str | None
    analysis_plugin_results_path: str | None
    experiment_log_path: str
    compiled_task_plan_path: str
    manifest_path: str
    summary_path: str
    report_path: str
    failures_path: str
    source_case_count: int | None = None
    excluded_case_count: int = 0
    selection_manifest_path: str | None = None
    excluded_cases_path: str | None = None
    request_previews_path: str | None = None
    request_preview_summary_path: str | None = None
    request_previews_markdown_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def evaluator_ids(self) -> list[str]:
        return self.scorer_ids

    @property
    def evaluator_result_count(self) -> int:
        return self.score_record_count

    @property
    def group_analysis_count(self) -> int:
        return self.group_analysis_record_count

    @property
    def evaluator_results_path(self) -> str:
        return self.score_records_path

    @property
    def group_analyses_path(self) -> str | None:
        return self.group_analysis_records_path


@dataclass(slots=True)
class ExperimentBundleManifest:
    experiment_id: str
    task_id: str
    task_version: str
    benchmark_id: str
    benchmark_path: str
    target_models: list[str]
    normalizer_ids: list[str]
    scorer_ids: list[str]
    analysis_plugin_ids: list[str]
    execution_mode: str
    case_count: int
    failure_count: int
    created_at: str
    target_run_ids: list[str] = field(default_factory=list)
    recipe_path: str | None = None
    auto_launch: bool = False
    launch_plan: dict[str, Any] = field(default_factory=dict)
    selection_applied: bool = False
    selection_spec: dict[str, Any] = field(default_factory=dict)
    selected_case_count: int | None = None
    source_case_count: int | None = None
    excluded_case_count: int = 0
    selection_manifest_path: str | None = None
    compiled_task_plan_path: str | None = None
    experiment_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def evaluator_ids(self) -> list[str]:
        return self.scorer_ids


ExperimentManifest = ExperimentBundleManifest


@dataclass(slots=True)
class SummaryReport:
    benchmark_id: str
    benchmark_path: str
    target_models: list[str]
    normalizer_ids: list[str]
    scorer_ids: list[str]
    analysis_plugin_ids: list[str]
    case_count: int
    target_result_count: int
    normalized_result_count: int
    score_record_count: int
    group_analysis_record_count: int
    analysis_plugin_result_count: int
    failure_count: int
    counts_by_target_model: dict[str, int] = field(default_factory=dict)
    counts_by_normalizer: dict[str, int] = field(default_factory=dict)
    counts_by_scorer: dict[str, int] = field(default_factory=dict)
    counts_by_analysis_plugin: dict[str, int] = field(default_factory=dict)
    counts_by_split: dict[str, int] = field(default_factory=dict)
    label_counts_by_target_model: dict[str, dict[str, int]] = field(default_factory=dict)
    average_numeric_scores_by_scorer: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def evaluator_ids(self) -> list[str]:
        return self.scorer_ids

    @property
    def evaluator_result_count(self) -> int:
        return self.score_record_count

    @property
    def group_analysis_count(self) -> int:
        return self.group_analysis_record_count

    @property
    def counts_by_evaluator(self) -> dict[str, int]:
        return self.counts_by_scorer

    @property
    def average_numeric_scores_by_evaluator(self) -> dict[str, dict[str, float]]:
        return self.average_numeric_scores_by_scorer


@dataclass(slots=True)
class ExperimentExportSummary:
    experiment_id: str
    experiment_dir: str
    export_dir: str
    export_format: str
    record_count: int
    jsonl_path: str | None = None
    csv_path: str | None = None
    manifest_path: str | None = None
    readme_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkBuilderSpec:
    builder: str
    description: str
    source: str = "core"
    manifest_path: str | None = None
    entrypoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NormalizerSpec:
    normalizer_id: str
    normalizer_type: str
    description: str = ""
    source: str = "core"
    accepted_input_types: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    entrypoint: str | None = None
    version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisPluginSpec:
    plugin_id: str
    plugin_type: str
    description: str = ""
    source: str = "core"
    entrypoint: str | None = None
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
