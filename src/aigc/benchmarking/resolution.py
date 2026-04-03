from __future__ import annotations

from pathlib import Path

from aigc.analysis import resolve_analysis_plugins
from aigc.evaluators import resolve_evaluators
from aigc.evaluators.models import EvaluatorSpec
from aigc.normalizers import resolve_normalizers
from aigc.normalizers.models import NormalizerSpec


def resolve_runtime_normalizers(
    *,
    normalizer_ids: list[str] | None,
    normalizer_config_path: str | Path | None,
) -> list[NormalizerSpec]:
    if not normalizer_ids:
        return []
    return resolve_normalizers(
        normalizer_ids,
        path=normalizer_config_path if normalizer_config_path is not None else None,  # type: ignore[arg-type]
    )


def resolve_runtime_scorers(
    *,
    scorer_ids: list[str] | None,
    scorer_model: str | None,
    scorer_profile: str | None,
    scorer_template: str | None,
    scorer_config_path: str | Path | None,
) -> list[EvaluatorSpec]:
    resolved = resolve_evaluators(
        scorer_ids or [],
        path=scorer_config_path if scorer_config_path is not None else None,  # type: ignore[arg-type]
    ) if scorer_ids else []
    if scorer_model or scorer_profile or scorer_template:
        resolved.append(
            EvaluatorSpec(
                evaluator_id="legacy_judge",
                evaluator_type="judge",
                description="Implicit scorer built from legacy CLI flags.",
                judge_model=scorer_model,
                annotation_profile=scorer_profile,
                annotation_template=scorer_template,
            )
        )
    return resolved


def resolve_runtime_analysis_plugins(
    *,
    analysis_plugin_ids: list[str] | None,
    analysis_config_path: str | Path | None,
):
    if not analysis_plugin_ids:
        return []
    return resolve_analysis_plugins(
        analysis_plugin_ids,
        path=analysis_config_path if analysis_config_path is not None else None,  # type: ignore[arg-type]
    )


def resolve_runtime_analyzers(*, benchmark_manifest: dict[str, object]) -> list[dict[str, object]]:
    analyzers: list[dict[str, object]] = []
    for analyzer_spec in benchmark_manifest.get("group_analyzers", []) or []:
        if not isinstance(analyzer_spec, dict):
            continue
        analyzers.append(dict(analyzer_spec))
    return analyzers
