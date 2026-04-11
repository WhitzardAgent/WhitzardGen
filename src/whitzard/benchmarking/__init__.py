from __future__ import annotations

from whitzard.benchmarking.bundle import inspect_benchmark_bundle, inspect_experiment_bundle
from whitzard.benchmarking.interfaces import (
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
    RealizationValidator,
    RunEngineGateway,
    ResultNormalizer,
    Scorer,
    TaskCompiler,
)
from whitzard.benchmarking.models import (
    AnalysisPluginResult,
    AnalysisPluginSpec,
    BenchmarkBuildSummary,
    BenchmarkBuilderSpec,
    BenchmarkCase,
    CaseSelectionResult,
    CaseSelectionSpec,
    CaseSet,
    CaseSourceRef,
    CompiledTaskPlan,
    EvalTask,
    ExecutionRequest,
    EvaluationExperimentSummary,
    ExperimentExportSummary,
    EvaluatorResult,
    ExperimentLogEvent,
    ExperimentManifest,
    GroupAnalysisRecord,
    NormalizedResult,
    NormalizerSpec,
    RealizationResult,
    RealizationSpec,
    RealizationValidationResult,
    ScoreRecord,
    SummaryReport,
    TargetRunReference,
    TargetResult,
)
from whitzard.benchmarking.packages import (
    BenchmarkPackageError,
    GenerativeBenchmarkPackage,
    SlotDefinition,
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
    "BenchmarkPackageError",
    "BenchmarkingError",
    "CaseSelectionResult",
    "CaseSelectionSpec",
    "CaseSet",
    "CaseSourceRef",
    "CompiledTaskPlan",
    "EvaluationExperimentSummary",
    "ExperimentExportSummary",
    "EvalTask",
    "ExecutionRequest",
    "Evaluator",
    "EvaluatorResult",
    "ExperimentLogEvent",
    "ExperimentRecipeError",
    "ExperimentManifest",
    "ExperimentRunner",
    "GenerativeBenchmarkPackage",
    "GroupAnalysisRequest",
    "GroupAnalysisRecord",
    "GroupAnalyzer",
    "NormalizationRequest",
    "NormalizedResult",
    "NormalizerSpec",
    "RealizationResult",
    "RealizationSpec",
    "RealizationValidationResult",
    "RealizationValidator",
    "ResultNormalizer",
    "RunEngineGateway",
    "ScoreRecord",
    "Scorer",
    "SlotDefinition",
    "SummaryReport",
    "TargetRunReference",
    "TaskCompiler",
    "TargetResult",
    "build_benchmark",
    "build_experiment_summary",
    "build_group_analyses",
    "evaluate_benchmark",
    "export_experiment",
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
    "sample_benchmark_bundle",
    "slugify",
]


def __getattr__(name: str):
    if name == "ExperimentRecipeError":
        from whitzard.benchmarking.recipes import ExperimentRecipeError

        return ExperimentRecipeError
    if name == "load_experiment_recipe":
        from whitzard.benchmarking.recipes import load_experiment_recipe

        return load_experiment_recipe
    if name in {
        "BenchmarkPackageError",
        "BenchmarkingError",
        "build_benchmark",
        "build_experiment_summary",
        "build_group_analyses",
        "evaluate_benchmark",
        "export_experiment",
        "inspect_experiment",
        "list_benchmark_builders",
        "list_experiments",
        "load_yaml_file",
        "normalize_builder_output",
        "normalize_case_payload",
        "render_experiment_report",
        "sample_benchmark_bundle",
        "slugify",
    }:
        if name == "BenchmarkPackageError":
            from whitzard.benchmarking.packages import BenchmarkPackageError as _pkg_err

            return _pkg_err
        from whitzard.benchmarking import service as _service

        return getattr(_service, name)
    raise AttributeError(name)
