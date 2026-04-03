from __future__ import annotations

from aigc.benchmarking.bundle import inspect_benchmark_bundle, inspect_experiment_bundle
from aigc.benchmarking.interfaces import (
    AnalysisPlugin,
    AnalysisPluginRequest,
    BenchmarkBuildOutput,
    BenchmarkBuildRequest,
    BenchmarkBuilder,
    ExperimentRunner,
    Evaluator,
    GroupAnalysisRequest,
    GroupAnalyzer,
    NormalizationRequest,
    RunEngineGateway,
    ResultNormalizer,
    Scorer,
    TaskCompiler,
)
from aigc.benchmarking.models import (
    AnalysisPluginResult,
    AnalysisPluginSpec,
    BenchmarkBuildSummary,
    BenchmarkBuilderSpec,
    BenchmarkCase,
    CaseSet,
    CaseSourceRef,
    CompiledTaskPlan,
    EvalTask,
    ExecutionRequest,
    EvaluationExperimentSummary,
    EvaluatorResult,
    ExperimentLogEvent,
    ExperimentManifest,
    GroupAnalysisRecord,
    NormalizedResult,
    NormalizerSpec,
    ScoreRecord,
    SummaryReport,
    TargetResult,
)

__all__ = [
    "AnalysisPlugin",
    "AnalysisPluginRequest",
    "AnalysisPluginResult",
    "AnalysisPluginSpec",
    "BenchmarkBuildOutput",
    "BenchmarkBuildRequest",
    "BenchmarkBuildSummary",
    "BenchmarkBuilder",
    "BenchmarkBuilderSpec",
    "BenchmarkCase",
    "BenchmarkingError",
    "CaseSet",
    "CaseSourceRef",
    "CompiledTaskPlan",
    "EvaluationExperimentSummary",
    "EvalTask",
    "ExecutionRequest",
    "Evaluator",
    "EvaluatorResult",
    "ExperimentLogEvent",
    "ExperimentRecipeError",
    "ExperimentManifest",
    "ExperimentRunner",
    "GroupAnalysisRequest",
    "GroupAnalysisRecord",
    "GroupAnalyzer",
    "NormalizationRequest",
    "NormalizedResult",
    "NormalizerSpec",
    "ResultNormalizer",
    "RunEngineGateway",
    "ScoreRecord",
    "Scorer",
    "SummaryReport",
    "TaskCompiler",
    "TargetResult",
    "build_benchmark",
    "build_experiment_summary",
    "build_group_analyses",
    "evaluate_benchmark",
    "inspect_benchmark_bundle",
    "inspect_experiment",
    "inspect_experiment_bundle",
    "list_benchmark_builders",
    "list_experiments",
    "load_experiment_recipe",
    "load_yaml_file",
    "normalize_builder_output",
    "normalize_case_payload",
    "render_experiment_report",
    "slugify",
]


def __getattr__(name: str):
    if name == "ExperimentRecipeError":
        from aigc.benchmarking.recipes import ExperimentRecipeError

        return ExperimentRecipeError
    if name == "load_experiment_recipe":
        from aigc.benchmarking.recipes import load_experiment_recipe

        return load_experiment_recipe
    if name in {
        "BenchmarkingError",
        "build_benchmark",
        "build_experiment_summary",
        "build_group_analyses",
        "evaluate_benchmark",
        "inspect_experiment",
        "list_benchmark_builders",
        "list_experiments",
        "load_yaml_file",
        "normalize_builder_output",
        "normalize_case_payload",
        "render_experiment_report",
        "slugify",
    }:
        from aigc.benchmarking import service as _service

        return getattr(_service, name)
    raise AttributeError(name)
