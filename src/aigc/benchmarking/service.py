from __future__ import annotations

import importlib
import importlib.util
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aigc.benchmarking.bundle import (
    build_benchmark_stats,
    inspect_benchmark_bundle,
    inspect_experiment_bundle,
    load_benchmark_cases,
    write_benchmark_bundle,
)
from aigc.benchmarking.compiler import DefaultTaskCompiler
from aigc.benchmarking.discovery import (
    BenchmarkDiscoveryError,
    discover_example_builder_specs,
    load_example_builder,
)
from aigc.benchmarking.interfaces import BenchmarkBuildOutput, BenchmarkBuildRequest
from aigc.benchmarking.models import (
    BenchmarkBuildSummary,
    BenchmarkBuilderSpec,
    BenchmarkCase,
    CaseSourceRef,
    EvalTask,
)
from aigc.benchmarking.runner import (
    DefaultExperimentRunner,
    build_group_analysis_records,
    build_summary_report,
    render_experiment_report,
)
from aigc.settings import get_benchmarks_root, get_experiments_root
from aigc.utils.progress import NullRunProgress, RunProgress


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
    execution_mode: str = "real",
    profile_path: str | Path | None = None,
    template_name: str | None = None,
    style_family_name: str | None = None,
    target_model_name: str | None = None,
    intended_modality: str | None = None,
    entrypoint: str | None = None,
    progress: RunProgress | None = None,
) -> BenchmarkBuildSummary:
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
        execution_mode=execution_mode,
        profile_path=profile_path,
        template_name=template_name,
        style_family_name=style_family_name,
        target_model_name=target_model_name,
        intended_modality=intended_modality,
        entrypoint=entrypoint,
        progress=progress,
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
    )


def evaluate_benchmark(
    *,
    benchmark_path: str | Path,
    target_models: list[str],
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
):
    progress = progress or NullRunProgress()
    benchmark_bundle = inspect_benchmark_bundle(benchmark_path)
    benchmark_dir = Path(benchmark_path)
    if benchmark_dir.is_file():
        benchmark_dir = benchmark_dir.parent
    manifest = dict(benchmark_bundle.get("manifest") or {})
    benchmark_id = str(manifest.get("benchmark_id") or benchmark_dir.name)
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
        execution_policy={
            "auto_launch": auto_launch,
            "launcher_config_path": str(launcher_config_path) if launcher_config_path not in (None, "") else None,
        },
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
    runner = DefaultExperimentRunner()
    return runner.run(
        task=task,
        compiled_plan=compiled_plan,
        experiment_dir=experiment_dir,
        execution_mode=execution_mode,
        progress=progress,
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
    paths = write_benchmark_bundle(
        benchmark_dir=benchmark_dir,
        cases=output.cases,
        manifest=manifest,
        stats=stats,
    )
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
    )


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
