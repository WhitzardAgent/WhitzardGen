from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aigc.benchmarking.models import (
    AnalysisPluginResult,
    BenchmarkCase,
    CaseSet,
    CompiledTaskPlan,
    EvalTask,
    ExecutionRequest,
    GroupAnalysisRecord,
    NormalizedResult,
    ScoreRecord,
    TargetResult,
)
from aigc.utils.progress import RunProgress


@dataclass(slots=True)
class BenchmarkBuildRequest:
    builder_name: str
    source_path: str | Path | None = None
    out_dir: str | Path | None = None
    benchmark_name: str | None = None
    seed: int = 42
    build_mode: str = "static"
    builder_config_path: str | Path | None = None
    count_config_path: str | Path | None = None
    llm_model: str | None = None
    execution_mode: str = "real"
    profile_path: str | Path | None = None
    template_name: str | None = None
    style_family_name: str | None = None
    target_model_name: str | None = None
    intended_modality: str | None = None
    entrypoint: str | None = None
    progress: RunProgress | None = None


@dataclass(slots=True)
class BenchmarkBuildOutput:
    cases: list[BenchmarkCase]
    source_path: str
    build_mode: str
    extra_manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroupAnalysisRequest:
    task: EvalTask
    compiled_plan: CompiledTaskPlan
    benchmark_id: str
    benchmark_manifest: dict[str, Any]
    target_results: list[TargetResult]
    normalized_results: list[NormalizedResult]
    score_records: list[ScoreRecord]

    @property
    def evaluator_results(self) -> list[ScoreRecord]:
        return self.score_records


class BenchmarkBuilder(ABC):
    builder_id: str
    description: str

    @abstractmethod
    def build(self, request: BenchmarkBuildRequest) -> BenchmarkBuildOutput:
        raise NotImplementedError


class Scorer(ABC):
    scorer_id: str


Evaluator = Scorer


@dataclass(slots=True)
class NormalizationRequest:
    task: EvalTask
    compiled_plan: CompiledTaskPlan
    benchmark_id: str
    benchmark_manifest: dict[str, Any]
    target_result: TargetResult


@dataclass(slots=True)
class AnalysisPluginRequest:
    task: EvalTask
    compiled_plan: CompiledTaskPlan
    benchmark_id: str
    benchmark_manifest: dict[str, Any]
    cases: list[BenchmarkCase]
    target_results: list[TargetResult]
    normalized_results: list[NormalizedResult]
    score_records: list[ScoreRecord]
    previous_outputs: dict[str, list[AnalysisPluginResult]] = field(default_factory=dict)

    @property
    def evaluator_results(self) -> list[ScoreRecord]:
        return self.score_records


class ResultNormalizer(ABC):
    normalizer_id: str

    @abstractmethod
    def normalize(self, request: NormalizationRequest) -> NormalizedResult:
        raise NotImplementedError


class AnalysisPlugin(ABC):
    plugin_id: str
    plugin_version: str = "v1"
    plugin_type: str = "comparative"

    @abstractmethod
    def execute(self, request: AnalysisPluginRequest) -> list[AnalysisPluginResult]:
        raise NotImplementedError


class GroupAnalyzer(ABC):
    analyzer_id: str

    @abstractmethod
    def analyze(self, request: GroupAnalysisRequest) -> list[GroupAnalysisRecord]:
        raise NotImplementedError


class TaskCompiler(ABC):
    @abstractmethod
    def compile(self, task: EvalTask) -> CompiledTaskPlan:
        raise NotImplementedError


class RunEngineGateway(ABC):
    @abstractmethod
    def execute_requests(
        self,
        *,
        task: EvalTask,
        requests: list[ExecutionRequest],
        experiment_dir: str | Path,
        execution_mode: str,
        progress: RunProgress | None = None,
    ) -> tuple[list[TargetResult], list[dict[str, Any]], list[str]]:
        raise NotImplementedError


class ExperimentRunner(ABC):
    @abstractmethod
    def run(
        self,
        *,
        task: EvalTask,
        compiled_plan: CompiledTaskPlan,
        experiment_dir: str | Path,
        execution_mode: str,
        progress: RunProgress | None = None,
    ):
        raise NotImplementedError
