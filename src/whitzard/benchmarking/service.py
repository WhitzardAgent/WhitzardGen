from __future__ import annotations

import importlib
import importlib.util
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from whitzard.benchmarking.bundle import (
    build_benchmark_stats,
    inspect_benchmark_bundle,
    inspect_experiment_bundle,
    load_benchmark_cases,
    write_benchmark_bundle,
)
from whitzard.benchmarking.compiler import DefaultTaskCompiler
from whitzard.benchmarking.discovery import (
    BenchmarkDiscoveryError,
    discover_example_builder_specs,
    load_example_builder,
)
from whitzard.benchmarking.export import ExperimentExportError, export_experiment_bundle
from whitzard.benchmarking.interfaces import BenchmarkBuildOutput, BenchmarkBuildRequest
from whitzard.benchmarking.models import (
    BenchmarkBuildSummary,
    BenchmarkBuilderSpec,
    BenchmarkCase,
    CaseSet,
    CaseSourceRef,
    EvalTask,
    CaseSelectionSpec,
    ExperimentExportSummary,
    PreviewSummary,
    RequestPreviewRecord,
)
from whitzard.benchmarking.preview import PreviewCollector, parse_preview_stage, write_request_preview_bundle
from whitzard.benchmarking.prompt_templates import resolve_prompt_template_config
from whitzard.benchmarking.runner import (
    DefaultExperimentRunner,
    build_group_analysis_records,
    build_summary_report,
    render_experiment_report,
)
from whitzard.benchmarking.selection import (
    apply_case_selection,
    clone_case_set_with_selection,
    normalize_case_selection_spec,
)
from whitzard.settings import get_benchmarks_root, get_experiments_root
from whitzard.utils.progress import NullRunProgress, RunProgress


class BenchmarkingError(RuntimeError):
    """Raised when benchmark build or evaluation fails."""


def list_benchmark_builders() -> list[dict[str, Any]]:
    builders = [
        BenchmarkBuilderSpec(
            builder="static_jsonl",
            description="Build a benchmark bundle from a canonical benchmark-case JSONL file.",
            source="core",
        ),
        BenchmarkBuilderSpec(
            builder="python_custom",
            description="Build a benchmark bundle from a user-provided Python builder entrypoint.",
            source="core",
        ),
    ]
    try:
        builders.extend(discover_example_builder_specs().values())
    except BenchmarkDiscoveryError:
        pass
    return [item.to_dict() for item in sorted(builders, key=lambda item: (item.source, item.builder))]


def build_benchmark(
    *,
    builder_name: str,
    source_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    benchmark_name: str | None = None,
    seed: int = 42,
    build_mode: str = "static",
    builder_config_path: str | Path | None = None,
    count_config_path: str | Path | None = None,
    llm_model: str | None = None,
    synthesis_model: str | None = None,
    execution_mode: str = "real",
    profile_path: str | Path | None = None,
    template_name: str | None = None,
    style_family_name: str | None = None,
    target_model_name: str | None = None,
    intended_modality: str | None = None,
    entrypoint: str | None = None,
    preview_enabled: bool = False,
    preview_only: bool = False,
    preview_count: int = 5,
    preview_stage: str = "all",
    preview_format: str = "text",
    progress: RunProgress | None = None,
) -> BenchmarkBuildSummary | PreviewSummary:
    request = BenchmarkBuildRequest(
        builder_name=builder_name,
        source_path=source_path,
        out_dir=out_dir,
        benchmark_name=benchmark_name,
        seed=seed,
        build_mode=build_mode,
        builder_config_path=builder_config_path,
        count_config_path=count_config_path,
        llm_model=llm_model,
        synthesis_model=synthesis_model,
        execution_mode=execution_mode,
        profile_path=profile_path,
        template_name=template_name,
        style_family_name=style_family_name,
        target_model_name=target_model_name,
        intended_modality=intended_modality,
        entrypoint=entrypoint,
        preview_enabled=preview_enabled,
        preview_only=preview_only,
        preview_count=preview_count,
        preview_stage=preview_stage,
        preview_format=preview_format,
        progress=progress,
    )
    if preview_only:
        return _preview_benchmark_build(
            request=request,
            out_dir=out_dir,
            benchmark_name=benchmark_name,
            builder_name=builder_name,
        )
    if builder_name == "static_jsonl":
        output = _build_static_jsonl(request)
    elif builder_name == "python_custom":
        output = _build_python_custom(request)
    else:
        try:
            builder = load_example_builder(builder_name)
        except BenchmarkDiscoveryError as exc:
            raise BenchmarkingError(str(exc)) from exc
        output = builder.build(request)
    return _write_benchmark_output(
        benchmark_name=benchmark_name,
        output=output,
        out_dir=out_dir,
        builder_name=builder_name,
        preview_enabled=preview_enabled,
        preview_stage=preview_stage,
        preview_count=preview_count,
    )


def evaluate_benchmark(
    *,
    benchmark_path: str | Path,
    target_models: list[str],
    case_selection: dict[str, Any] | CaseSelectionSpec | None = None,
    normalizer_ids: list[str] | None = None,
    evaluator_ids: list[str] | None = None,
    analysis_plugin_ids: list[str] | None = None,
    evaluator_model: str | None = None,
    evaluator_profile: str | None = None,
    evaluator_template: str | None = None,
    out_dir: str | Path | None = None,
    execution_mode: str = "real",
    progress: RunProgress | None = None,
    normalizer_config_path: str | Path | None = None,
    evaluator_config_path: str | Path | None = None,
    analysis_config_path: str | Path | None = None,
    recipe_path: str | Path | None = None,
    auto_launch: bool = False,
    launcher_config_path: str | Path | None = None,
    execution_policy: dict[str, Any] | None = None,
    preview_enabled: bool = False,
    preview_only: bool = False,
    preview_count: int = 5,
    preview_stage: str = "all",
    preview_format: str = "text",
) -> Any:
    progress = progress or NullRunProgress()
    benchmark_bundle = inspect_benchmark_bundle(benchmark_path)
    benchmark_dir = Path(benchmark_path)
    if benchmark_dir.is_file():
        benchmark_dir = benchmark_dir.parent
    manifest = dict(benchmark_bundle.get("manifest") or {})
    benchmark_id = str(manifest.get("benchmark_id") or benchmark_dir.name)
    resolved_execution_policy = dict(execution_policy or {})
    resolved_execution_policy.setdefault("text_prompt_composition", {})
    recipe_base_dir = Path(recipe_path).resolve().parent if recipe_path not in (None, "") else None
    resolved_execution_policy["target_prompt_template"] = resolve_prompt_template_config(
        dict(resolved_execution_policy.get("target_prompt_template", {}) or {}),
        base_dir=recipe_base_dir,
    )
    resolved_execution_policy["judge_prompt_template"] = resolve_prompt_template_config(
        dict(resolved_execution_policy.get("judge_prompt_template", {}) or {}),
        base_dir=recipe_base_dir,
    )
    resolved_execution_policy["auto_launch"] = auto_launch
    resolved_execution_policy["launcher_config_path"] = (
        str(launcher_config_path) if launcher_config_path not in (None, "") else None
    )
    resolved_execution_policy["request_preview"] = {
        "enabled": bool(preview_enabled),
        "preview_only": bool(preview_only),
        "preview_count": int(preview_count),
        "preview_stage": str(preview_stage or "all"),
        "preview_format": str(preview_format or "text"),
    }
    task = EvalTask(
        task_id=f"task_{slugify(benchmark_id)}",
        case_source=CaseSourceRef(
            source_type="benchmark_bundle",
            source_path=str(benchmark_dir),
            builder_name=str(manifest.get("builder_name") or manifest.get("source_builder") or ""),
            metadata={"build_mode": manifest.get("build_mode")},
        ),
        case_set_path=str(benchmark_dir),
        target_models=list(target_models),
        case_selection=(
            case_selection
            if isinstance(case_selection, CaseSelectionSpec)
            else normalize_case_selection_spec(case_selection)
        ),
        execution_policy=resolved_execution_policy,
        normalizer_ids=list(normalizer_ids or []),
        scorer_ids=list(evaluator_ids or []),
        plugin_ids=list(analysis_plugin_ids or []),
        output_policy={},
        metadata={
            "recipe_path": str(recipe_path) if recipe_path not in (None, "") else None,
            "legacy_scorer_model": evaluator_model,
            "legacy_scorer_profile": evaluator_profile,
            "legacy_scorer_template": evaluator_template,
        },
    )
    compiler = DefaultTaskCompiler(
        normalizer_config_path=normalizer_config_path,
        scorer_config_path=evaluator_config_path,
        analysis_config_path=analysis_config_path,
    )
    compiled_plan = compiler.compile(task)
    if evaluator_model or evaluator_profile or evaluator_template:
        compiled_plan.scorer_specs.append(
            {
                "evaluator_id": "legacy_judge",
                "evaluator_type": "judge",
                "description": "Implicit scorer built from legacy CLI flags.",
                "judge_model": evaluator_model,
                "annotation_profile": evaluator_profile,
                "annotation_template": evaluator_template,
            }
        )
    experiment_id = f"experiment_{slugify(benchmark_id)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = Path(out_dir) if out_dir is not None else get_experiments_root() / experiment_id
    if preview_only:
        return _preview_evaluation(
            task=task,
            compiled_plan=compiled_plan,
            experiment_dir=experiment_dir,
        )
    runner = DefaultExperimentRunner()
    return runner.run(
        task=task,
        compiled_plan=compiled_plan,
        experiment_dir=experiment_dir,
        execution_mode=execution_mode,
        progress=progress,
    )


def sample_benchmark_bundle(
    *,
    benchmark_path: str | Path,
    case_selection: dict[str, Any] | CaseSelectionSpec,
    out_dir: str | Path | None = None,
    benchmark_name: str | None = None,
) -> BenchmarkBuildSummary:
    benchmark_bundle = inspect_benchmark_bundle(benchmark_path)
    benchmark_dir = Path(benchmark_path)
    if benchmark_dir.is_file():
        benchmark_dir = benchmark_dir.parent
    manifest = dict(benchmark_bundle.get("manifest") or {})
    benchmark_id = str(manifest.get("benchmark_id") or benchmark_dir.name)
    cases = load_benchmark_cases(benchmark_dir / "cases.jsonl")
    if not cases:
        raise BenchmarkingError(f"No benchmark cases were found at {benchmark_dir / 'cases.jsonl'}")
    spec = case_selection if isinstance(case_selection, CaseSelectionSpec) else normalize_case_selection_spec(case_selection)
    if spec is None:
        raise BenchmarkingError("benchmark sample requires a non-empty case selection config.")
    case_set = CaseSet(
        benchmark_id=benchmark_id,
        cases=cases,
        source=CaseSourceRef(
            source_type="benchmark_bundle",
            source_path=str(benchmark_dir),
            builder_name=str(manifest.get("builder_name") or manifest.get("source_builder") or ""),
            metadata={"build_mode": manifest.get("build_mode")},
        ),
        manifest=manifest,
        stats=build_benchmark_stats(cases),
        case_set_path=str(benchmark_dir / "cases.jsonl"),
    )
    selection_result = apply_case_selection(case_set=case_set, spec=spec)
    selected_case_set = clone_case_set_with_selection(case_set=case_set, selection_result=selection_result)
    sampled_benchmark_id = slugify(benchmark_name or benchmark_id)
    materialized_cases = [
        BenchmarkCase(
            benchmark_id=sampled_benchmark_id,
            case_id=case.case_id,
            input_modality=case.input_modality,
            input_payload=dict(case.input_payload),
            metadata=dict(case.metadata),
            tags=list(case.tags),
            split=case.split,
            expected_output_contract=case.expected_output_contract,
            expected_structure=case.expected_structure,
            case_version=case.case_version,
            source_builder=case.source_builder,
            grouping=dict(case.grouping),
            execution_hints=dict(case.execution_hints),
            evaluation_hints=dict(case.evaluation_hints),
            language=case.language,
            parameters=dict(case.parameters),
            prompt=case.prompt,
            instruction=case.instruction,
            context=case.context,
            input_type=case.input_type,
        )
        for case in selected_case_set.cases
    ]
    bundle_id = f"{sampled_benchmark_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    target_dir = Path(out_dir) if out_dir is not None else get_benchmarks_root() / bundle_id
    sampled_manifest = {
        **manifest,
        "benchmark_id": sampled_benchmark_id,
        "bundle_id": target_dir.name,
        "source_path": str(benchmark_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "build_mode": "sampled",
        "case_count": len(selected_case_set.cases),
        "selection_applied": True,
        "selection_spec": spec.to_dict(),
        "source_case_count": selection_result.counts_before,
        "selected_case_count": selection_result.counts_after,
        "excluded_case_count": len(selection_result.excluded_cases),
    }
    paths = write_benchmark_bundle(
        benchmark_dir=target_dir,
        cases=materialized_cases,
        manifest=sampled_manifest,
        stats=build_benchmark_stats(materialized_cases),
        extra_jsonl_files={"excluded_cases.jsonl": [case.to_dict() for case in selection_result.excluded_cases]},
    )
    selection_manifest_path = Path(target_dir) / "selection_manifest.json"
    selection_manifest_payload = {
        **selection_result.selection_manifest,
        "source_benchmark_path": str(benchmark_dir),
        "source_benchmark_id": benchmark_id,
        "selected_case_count": len(selected_case_set.cases),
        "excluded_case_count": len(selection_result.excluded_cases),
    }
    selection_manifest_path.write_text(
        json.dumps(selection_manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest_path = Path(paths["manifest_path"])
    sampled_manifest["selection_manifest_path"] = str(selection_manifest_path)
    sampled_manifest["excluded_cases_path"] = str(target_dir / "excluded_cases.jsonl")
    manifest_path.write_text(json.dumps(sampled_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return BenchmarkBuildSummary(
        benchmark_id=str(sampled_manifest["benchmark_id"]),
        benchmark_dir=str(target_dir),
        builder_name=str(sampled_manifest.get("builder_name") or manifest.get("builder_name") or "benchmark_sample"),
        source_path=str(benchmark_dir),
        case_set_path=paths["case_set_path"],
        manifest_path=paths["manifest_path"],
        stats_path=paths["stats_path"],
        case_count=len(materialized_cases),
        build_mode="sampled",
        selection_manifest_path=str(selection_manifest_path),
        excluded_cases_path=str(target_dir / "excluded_cases.jsonl"),
        source_case_count=selection_result.counts_before,
        excluded_case_count=len(selection_result.excluded_cases),
    )


def list_experiments() -> list[dict[str, Any]]:
    root = get_experiments_root()
    if not root.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/experiment_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["experiment_dir"] = str(manifest_path.parent)
        manifests.append(payload)
    return manifests


def inspect_experiment(path_or_id: str | Path) -> dict[str, Any]:
    path = Path(path_or_id)
    if path.exists():
        return inspect_experiment_bundle(path)
    return inspect_experiment_bundle(get_experiments_root() / str(path_or_id))


def export_experiment(
    *,
    experiment: str | Path,
    output_dir: str | Path | None = None,
    export_format: str = "both",
) -> ExperimentExportSummary:
    try:
        return export_experiment_bundle(
            experiment=experiment,
            output_dir=output_dir,
            export_format=export_format,
        )
    except ExperimentExportError as exc:
        raise BenchmarkingError(str(exc)) from exc


def build_group_analyses(*args: Any, **kwargs: Any):
    return build_group_analysis_records(*args, **kwargs)


def build_experiment_summary(*args: Any, **kwargs: Any):
    return build_summary_report(*args, **kwargs)


def normalize_case_payload(
    payload: dict[str, Any],
    *,
    benchmark_id: str,
    default_builder: str,
    default_split: str,
) -> BenchmarkCase:
    case_id = str(payload.get("case_id") or payload.get("prompt_id") or "").strip()
    if not case_id:
        raise BenchmarkingError("Benchmark case is missing case_id.")
    input_modality = str(payload.get("input_modality") or payload.get("input_type") or "text").strip() or "text"
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
    if not input_payload:
        raise BenchmarkingError(f"Benchmark case {case_id} is missing input payload.")
    return BenchmarkCase(
        benchmark_id=str(payload.get("benchmark_id", benchmark_id) or benchmark_id),
        case_id=case_id,
        input_modality=input_modality,
        input_payload=input_payload,
        metadata=dict(payload.get("metadata", {}) or {}),
        tags=[str(item) for item in payload.get("tags", []) or []],
        split=str(payload.get("split", default_split) or default_split),
        expected_output_contract=payload.get("expected_output_contract", payload.get("expected_structure")),
        expected_structure=payload.get("expected_structure"),
        case_version=_optional_text(payload.get("case_version") or payload.get("version")),
        source_builder=_optional_text(payload.get("source_builder")) or default_builder,
        grouping=dict(payload.get("grouping", {}) or {}),
        execution_hints=dict(payload.get("execution_hints", {}) or {}),
        evaluation_hints=dict(payload.get("evaluation_hints", {}) or {}),
        language=str(payload.get("language", "en") or "en"),
        parameters=dict(payload.get("parameters", {}) or {}),
        prompt=prompt,
        instruction=instruction,
        context=payload.get("context"),
        input_type=_optional_text(payload.get("input_type")),
    )


def normalize_builder_output(
    payload: Iterable[Any],
    *,
    benchmark_id: str,
    source_builder: str,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for item in payload:
        if isinstance(item, BenchmarkCase):
            if not item.benchmark_id:
                item.benchmark_id = benchmark_id  # type: ignore[misc]
            if item.source_builder is None:
                item.source_builder = source_builder  # type: ignore[misc]
            cases.append(item)
            continue
        if not isinstance(item, dict):
            raise BenchmarkingError("Python builder outputs must be BenchmarkCase objects or dicts.")
        cases.append(
            normalize_case_payload(
                item,
                benchmark_id=benchmark_id,
                default_builder=source_builder,
                default_split="default",
            )
        )
    return cases


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchmarkingError(f"Required YAML file does not exist: {path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise BenchmarkingError(f"PyYAML is required to parse benchmark YAML: {path}") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise BenchmarkingError(f"Expected YAML object in {path}, got {type(payload).__name__}.")
    return payload


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "item"


def _build_static_jsonl(request: BenchmarkBuildRequest) -> BenchmarkBuildOutput:
    if request.source_path is None:
        raise BenchmarkingError("static_jsonl builder requires --source.")
    source = Path(request.source_path)
    benchmark_id = slugify(request.benchmark_name or source.stem)
    cases = _load_static_cases(source_path=source, benchmark_id=benchmark_id)
    return BenchmarkBuildOutput(cases=cases, source_path=str(source), build_mode="static")


def _build_python_custom(request: BenchmarkBuildRequest) -> BenchmarkBuildOutput:
    resolved_entrypoint = request.entrypoint or _load_python_builder_entrypoint(request.builder_config_path)
    if not resolved_entrypoint:
        raise BenchmarkingError("python_custom builder requires --entrypoint or a builder config with entrypoint.")
    factory = _load_builder_callable(resolved_entrypoint)
    builder_config = _load_builder_config(request.builder_config_path)
    payload = factory(builder_config, request.seed)
    benchmark_id = slugify(request.benchmark_name or builder_config.get("benchmark_name") or "python_custom")
    cases = normalize_builder_output(payload, benchmark_id=benchmark_id, source_builder="python_custom")
    return BenchmarkBuildOutput(
        cases=cases,
        source_path=resolved_entrypoint,
        build_mode="dynamic",
        extra_manifest={"builder_config_path": str(request.builder_config_path) if request.builder_config_path else None},
    )


def _write_benchmark_output(
    *,
    benchmark_name: str | None,
    output: BenchmarkBuildOutput,
    out_dir: str | Path | None,
    builder_name: str,
    preview_enabled: bool,
    preview_stage: str,
    preview_count: int,
) -> BenchmarkBuildSummary:
    if output.cases:
        benchmark_id = output.cases[0].benchmark_id
    else:
        benchmark_id = slugify(benchmark_name or Path(output.source_path).stem)
    bundle_id = f"{benchmark_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    benchmark_dir = Path(out_dir) if out_dir is not None else get_benchmarks_root() / bundle_id
    stats = build_benchmark_stats(output.cases)
    manifest = {
        "benchmark_id": benchmark_id,
        "bundle_id": Path(benchmark_dir).name,
        "builder_name": builder_name,
        "source_path": str(output.source_path),
        "created_at": datetime.now(UTC).isoformat(),
        "build_mode": output.build_mode,
        "case_count": len(output.cases),
        **output.extra_manifest,
    }
    if output.build_artifacts:
        manifest["build_artifacts"] = dict(output.build_artifacts)
        manifest["raw_realization_count"] = int(output.build_artifacts.get("realization_raw_case_count", 0))
        manifest["rejected_case_count"] = int(output.build_artifacts.get("realization_rejected_case_count", 0))
    if output.build_manifest_overrides:
        manifest.update(dict(output.build_manifest_overrides))
    paths = write_benchmark_bundle(
        benchmark_dir=benchmark_dir,
        cases=output.cases,
        manifest=manifest,
        stats=stats,
        extra_jsonl_files=output.extra_jsonl_files,
    )
    preview_summary = _write_preview_bundle_if_needed(
        target_dir=benchmark_dir,
        preview_records=output.request_preview_records,
        source_context=output.request_preview_source_context,
        preview_only=False,
        preview_stage=preview_stage if preview_enabled else "all",
        preview_count=preview_count if preview_enabled else 0,
    )
    if preview_summary is not None:
        manifest_path = Path(paths["manifest_path"])
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_payload["request_previews_path"] = preview_summary.request_previews_path
        manifest_payload["request_preview_summary_path"] = preview_summary.request_preview_summary_path
        manifest_payload["request_previews_markdown_path"] = preview_summary.request_previews_markdown_path
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return BenchmarkBuildSummary(
        benchmark_id=benchmark_id,
        benchmark_dir=str(benchmark_dir),
        builder_name=builder_name,
        source_path=str(output.source_path),
        case_set_path=paths["case_set_path"],
        manifest_path=paths["manifest_path"],
        stats_path=paths["stats_path"],
        case_count=len(output.cases),
        build_mode=output.build_mode,
        raw_realizations_path=paths.get("raw_realizations.jsonl"),
        rejected_realizations_path=paths.get("rejected_realizations.jsonl"),
        request_previews_path=preview_summary.request_previews_path if preview_summary else None,
        request_preview_summary_path=preview_summary.request_preview_summary_path if preview_summary else None,
        request_previews_markdown_path=preview_summary.request_previews_markdown_path if preview_summary else None,
    )


def _preview_benchmark_build(
    *,
    request: BenchmarkBuildRequest,
    out_dir: str | Path | None,
    benchmark_name: str | None,
    builder_name: str,
) -> PreviewSummary:
    preview_dir = _resolve_preview_dir(
        out_dir=out_dir,
        fallback_root=get_benchmarks_root(),
        prefix=f"{slugify(benchmark_name or builder_name)}_preview",
    )
    if builder_name == "static_jsonl":
        return write_request_preview_bundle(
            preview_dir=preview_dir,
            bundle=PreviewCollector(
                enabled_stages={"all"},
                source_context={
                    "builder_name": builder_name,
                    "source_path": str(request.source_path or ""),
                    "note": "No request templates are produced by static_jsonl.",
                },
            ).to_bundle(),
            preview_only=True,
            preview_stage=request.preview_stage,
            preview_count=request.preview_count,
        )
    if builder_name == "python_custom":
        return write_request_preview_bundle(
            preview_dir=preview_dir,
            bundle=PreviewCollector(
                enabled_stages={"all"},
                source_context={
                    "builder_name": builder_name,
                    "source_path": str(request.source_path or request.entrypoint or ""),
                    "note": "python_custom preview requires a builder-specific preview implementation.",
                },
            ).to_bundle(),
            preview_only=True,
            preview_stage=request.preview_stage,
            preview_count=request.preview_count,
        )
    try:
        builder = load_example_builder(builder_name)
    except BenchmarkDiscoveryError as exc:
        raise BenchmarkingError(str(exc)) from exc
    preview_method = getattr(builder, "preview", None)
    if callable(preview_method):
        preview_bundle = preview_method(request)
        return write_request_preview_bundle(
            preview_dir=preview_dir,
            bundle=preview_bundle,
            preview_only=True,
            preview_stage=request.preview_stage,
            preview_count=request.preview_count,
        )
    return write_request_preview_bundle(
        preview_dir=preview_dir,
        bundle=PreviewCollector(
            enabled_stages={"all"},
            source_context={
                "builder_name": builder_name,
                "source_path": str(request.source_path or ""),
                "note": "Builder does not expose a preview implementation.",
            },
        ).to_bundle(),
        preview_only=True,
        preview_stage=request.preview_stage,
        preview_count=request.preview_count,
    )


def _preview_evaluation(
    *,
    task: EvalTask,
    compiled_plan,
    experiment_dir: str | Path,
) -> PreviewSummary:
    from whitzard.annotation.service import build_annotation_preview_prompts
    from whitzard.benchmarking.gateway import PromptRecordRunEngineGateway, build_prompt_record_from_execution_request
    from whitzard.evaluators.models import EvaluatorSpec
    from whitzard.evaluators.service import _resolve_judge_prompt_template

    preview_config = dict(task.execution_policy.get("request_preview", {}) or {})
    preview_stage = str(preview_config.get("preview_stage") or "all")
    preview_count = int(preview_config.get("preview_count", 5))
    collector = PreviewCollector(
        enabled_stages=parse_preview_stage(preview_stage, allowed_stages={"target", "judge", "all"}),
        source_context={
            "task_id": task.task_id,
            "benchmark_id": compiled_plan.case_set.benchmark_id,
            "benchmark_path": compiled_plan.case_set.case_set_path,
            "target_models": list(task.target_models),
            "preview_only": True,
        },
    )
    gateway = PromptRecordRunEngineGateway()
    gateway.preview_requests(
        task=task,
        requests=compiled_plan.execution_requests,
        preview_collector=collector,
    )
    synthetic_records = []
    for request in compiled_plan.execution_requests:
        prompt_record = build_prompt_record_from_execution_request(request)
        synthetic_records.append(
            {
                "record_id": request.request_id,
                "prompt_id": request.request_id,
                "task_id": request.task_id,
                "model_name": request.target_model,
                "task_type": "t2t",
                "artifact_type": "text",
                "artifact_path": "",
                "prompt": prompt_record.prompt,
                "negative_prompt": prompt_record.negative_prompt,
                "prompt_metadata": dict(prompt_record.metadata),
                "artifact_metadata": {},
                "generation_params": dict(prompt_record.parameters),
                "language": prompt_record.language,
            }
        )
    scorer_payloads: list[dict[str, Any]] = []
    for raw_scorer in list(compiled_plan.scorer_specs):
        scorer = raw_scorer if isinstance(raw_scorer, EvaluatorSpec) else EvaluatorSpec(**raw_scorer)
        scorer_payload = scorer.to_dict()
        scorer_payload["prompt_template"] = _resolve_judge_prompt_template(task=task, scorer=scorer)
        scorer_payloads.append(scorer_payload)
    build_annotation_preview_prompts(
        source_run_id="preview_only",
        source_records=synthetic_records,
        scorers=scorer_payloads,
        preview_collector=collector,
        extra_template_context_by_record_id={
            str(item["record_id"]): {
                "target_output_text": "<target output unavailable during preview-only>",
                "normalized_result": {},
                "case_metadata": dict(item.get("prompt_metadata", {}) or {}),
            }
            for item in synthetic_records
        },
    )
    return write_request_preview_bundle(
        preview_dir=experiment_dir,
        bundle=collector.to_bundle(),
        preview_only=True,
        preview_stage=preview_stage,
        preview_count=preview_count,
    )


def _write_preview_bundle_if_needed(
    *,
    target_dir: str | Path,
    preview_records: list[dict[str, Any]],
    source_context: dict[str, Any],
    preview_only: bool,
    preview_stage: str,
    preview_count: int,
) -> PreviewSummary | None:
    if not preview_records:
        return None
    collector = PreviewCollector(
        enabled_stages={"all"},
        source_context=source_context,
    )
    for item in preview_records:
        collector.collect(
            RequestPreviewRecord(
                stage=str(item.get("stage", "unknown")),
                entity_id=str(item.get("entity_id", "")),
                case_id=_optional_text(item.get("case_id")),
                request_id=_optional_text(item.get("request_id")),
                target_model=_optional_text(item.get("target_model")),
                judge_model=_optional_text(item.get("judge_model")),
                template_name=_optional_text(item.get("template_name")),
                template_version=_optional_text(item.get("template_version")),
                rendered_prompt=str(item.get("rendered_prompt", "")),
                metadata=dict(item.get("metadata", {}) or {}),
            )
        )
    return write_request_preview_bundle(
        preview_dir=target_dir,
        bundle=collector.to_bundle(),
        preview_only=preview_only,
        preview_stage=preview_stage,
        preview_count=preview_count,
    )


def _resolve_preview_dir(
    *,
    out_dir: str | Path | None,
    fallback_root: Path,
    prefix: str,
) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    return fallback_root / f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def _load_static_cases(*, source_path: Path, benchmark_id: str) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise BenchmarkingError(f"Line {line_number} of {source_path} must be a JSON object.")
            cases.append(
                normalize_case_payload(
                    payload,
                    benchmark_id=benchmark_id,
                    default_builder="static_jsonl",
                    default_split="default",
                )
            )
    return cases


def _load_builder_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_yaml_file(Path(path))


def _load_python_builder_entrypoint(path: str | Path | None) -> str | None:
    if path is None:
        return None
    payload = _load_builder_config(path)
    entrypoint = payload.get("entrypoint")
    if entrypoint in (None, ""):
        return None
    return str(entrypoint)


def _load_builder_callable(entrypoint: str):
    if ":" not in entrypoint:
        raise BenchmarkingError("Python builder entrypoint must look like module:function or /path/to/file.py:function")
    module_ref, function_name = entrypoint.split(":", 1)
    module_ref = module_ref.strip()
    function_name = function_name.strip()
    if not module_ref or not function_name:
        raise BenchmarkingError(f"Invalid python builder entrypoint: {entrypoint}")
    if module_ref.endswith(".py") or Path(module_ref).exists():
        module_path = Path(module_ref)
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise BenchmarkingError(f"Unable to load python builder module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise BenchmarkingError(f"Builder function {function_name} was not found in {module_ref}") from exc


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None
