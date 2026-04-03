from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from aigc.analysis.models import AnalysisPluginSpec
from aigc.benchmarking.discovery import BenchmarkDiscoveryError, load_analysis_plugin
from aigc.benchmarking.interfaces import AnalysisPluginRequest
from aigc.benchmarking.models import (
    AnalysisPluginResult,
    BenchmarkCase,
    CompiledTaskPlan,
    EvalTask,
    NormalizedResult,
    ScoreRecord,
    TargetResult,
)


class AnalysisError(RuntimeError):
    """Raised when analysis plugin execution fails."""


_PLUGIN_SPEC_KEYS = {
    "plugin_id",
    "plugin_type",
    "description",
    "dependencies",
    "config",
    "version",
}


def run_analysis_plugins(
    *,
    task: EvalTask,
    compiled_plan: CompiledTaskPlan,
    benchmark_id: str,
    benchmark_manifest: dict[str, Any],
    cases: list[BenchmarkCase],
    target_results: list[TargetResult],
    normalized_results: list[NormalizedResult],
    score_records: list[ScoreRecord],
    plugin_specs: list[AnalysisPluginSpec | dict[str, Any]],
) -> tuple[list[AnalysisPluginResult], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not plugin_specs:
        return [], failures
    normalized_specs = [
        spec
        if isinstance(spec, AnalysisPluginSpec)
        else AnalysisPluginSpec(**{key: value for key, value in spec.items() if key in _PLUGIN_SPEC_KEYS})
        for spec in plugin_specs
    ]
    ordered_specs = _sort_plugins(normalized_specs)
    results: list[AnalysisPluginResult] = []
    results_by_plugin: dict[str, list[AnalysisPluginResult]] = defaultdict(list)
    for spec in ordered_specs:
        try:
            plugin = load_analysis_plugin(spec.plugin_id)
        except BenchmarkDiscoveryError as exc:
            failures.append({"stage": "analysis_plugin", "plugin_id": spec.plugin_id, "error": str(exc)})
            continue
        try:
            plugin_results = plugin.execute(
                AnalysisPluginRequest(
                    task=task,
                    compiled_plan=compiled_plan,
                    benchmark_id=benchmark_id,
                    benchmark_manifest=benchmark_manifest,
                    cases=cases,
                    target_results=target_results,
                    normalized_results=normalized_results,
                    score_records=score_records,
                    previous_outputs=dict(results_by_plugin),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            failures.append({"stage": "analysis_plugin", "plugin_id": spec.plugin_id, "error": str(exc)})
            continue
        for result in plugin_results:
            results.append(result)
            results_by_plugin[result.plugin_id].append(result)
    return results, failures


def _sort_plugins(plugin_specs: list[AnalysisPluginSpec]) -> list[AnalysisPluginSpec]:
    spec_by_id = {spec.plugin_id: spec for spec in plugin_specs}
    indegree: dict[str, int] = {spec.plugin_id: 0 for spec in plugin_specs}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for spec in plugin_specs:
        for dependency in spec.dependencies:
            if dependency not in spec_by_id:
                continue
            outgoing[dependency].append(spec.plugin_id)
            indegree[spec.plugin_id] += 1
    queue = deque(sorted([plugin_id for plugin_id, degree in indegree.items() if degree == 0]))
    ordered_ids: list[str] = []
    while queue:
        plugin_id = queue.popleft()
        ordered_ids.append(plugin_id)
        for dependent in sorted(outgoing.get(plugin_id, [])):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if len(ordered_ids) != len(plugin_specs):
        raise AnalysisError("Analysis plugin dependency graph contains a cycle.")
    return [spec_by_id[plugin_id] for plugin_id in ordered_ids]
