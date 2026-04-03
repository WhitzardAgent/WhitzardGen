from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from aigc.benchmarking.interfaces import AnalysisPlugin, BenchmarkBuilder, GroupAnalyzer, ResultNormalizer
from aigc.benchmarking.models import AnalysisPluginSpec, BenchmarkBuilderSpec, NormalizerSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"


class BenchmarkDiscoveryError(RuntimeError):
    """Raised when benchmark example discovery fails."""


def discover_example_builder_specs(examples_root: Path = EXAMPLES_ROOT) -> dict[str, BenchmarkBuilderSpec]:
    specs: dict[str, BenchmarkBuilderSpec] = {}
    for manifest_path in sorted((examples_root / "benchmarks").glob("*/builder.yaml")):
        payload = _load_manifest(manifest_path)
        builder_id = str(payload.get("builder_id") or "").strip()
        if not builder_id:
            raise BenchmarkDiscoveryError(f"Builder manifest missing builder_id: {manifest_path}")
        specs[builder_id] = BenchmarkBuilderSpec(
            builder=builder_id,
            description=str(payload.get("description") or ""),
            source="example",
            manifest_path=str(manifest_path),
            entrypoint=str(payload.get("entrypoint") or "").strip() or None,
        )
    return specs


def discover_example_evaluator_specs(examples_root: Path = EXAMPLES_ROOT) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted((examples_root / "evaluators").glob("*/evaluator.yaml")):
        payload = _load_manifest(manifest_path)
        evaluator_id = str(payload.get("evaluator_id") or "").strip()
        if not evaluator_id:
            raise BenchmarkDiscoveryError(f"Evaluator manifest missing evaluator_id: {manifest_path}")
        specs[evaluator_id] = {
            **payload,
            "source": "example",
            "manifest_path": str(manifest_path),
        }
    return specs


def discover_example_normalizer_specs(examples_root: Path = EXAMPLES_ROOT) -> dict[str, NormalizerSpec]:
    specs: dict[str, NormalizerSpec] = {}
    for manifest_path in sorted((examples_root / "normalizers").glob("*/normalizer.yaml")):
        payload = _load_manifest(manifest_path)
        normalizer_id = str(payload.get("normalizer_id") or "").strip()
        if not normalizer_id:
            raise BenchmarkDiscoveryError(f"Normalizer manifest missing normalizer_id: {manifest_path}")
        specs[normalizer_id] = NormalizerSpec(
            normalizer_id=normalizer_id,
            normalizer_type=str(payload.get("type") or "custom"),
            description=str(payload.get("description") or ""),
            source="example",
            accepted_input_types=[str(item) for item in payload.get("accepted_input_types", []) or []],
            config=dict(payload.get("config", {}) or {}),
            manifest_path=str(manifest_path),
            entrypoint=str(payload.get("entrypoint") or "").strip() or None,
            version=str(payload.get("version") or "v1"),
        )
    return specs


def discover_example_analysis_plugin_specs(examples_root: Path = EXAMPLES_ROOT) -> dict[str, AnalysisPluginSpec]:
    specs: dict[str, AnalysisPluginSpec] = {}
    for manifest_path in sorted((examples_root / "analysis_plugins").glob("*/plugin.yaml")):
        payload = _load_manifest(manifest_path)
        plugin_id = str(payload.get("plugin_id") or "").strip()
        if not plugin_id:
            raise BenchmarkDiscoveryError(f"Analysis plugin manifest missing plugin_id: {manifest_path}")
        specs[plugin_id] = AnalysisPluginSpec(
            plugin_id=plugin_id,
            plugin_type=str(payload.get("plugin_type") or "comparative"),
            description=str(payload.get("description") or ""),
            source="example",
            entrypoint=str(payload.get("entrypoint") or "").strip() or None,
            dependencies=[str(item) for item in payload.get("dependencies", []) or []],
            config=dict(payload.get("config", {}) or {}),
            manifest_path=str(manifest_path),
            version=str(payload.get("version") or "v1"),
        )
    return specs


def load_example_builder(builder_id: str, *, examples_root: Path = EXAMPLES_ROOT) -> BenchmarkBuilder:
    specs = discover_example_builder_specs(examples_root)
    try:
        spec = specs[builder_id]
    except KeyError as exc:
        raise BenchmarkDiscoveryError(f"Unknown example benchmark builder: {builder_id}") from exc
    if not spec.entrypoint:
        raise BenchmarkDiscoveryError(f"Example benchmark builder has no entrypoint: {builder_id}")
    builder = _load_entrypoint_object(spec.entrypoint)
    if isinstance(builder, type):
        builder = builder()
    if not isinstance(builder, BenchmarkBuilder):
        raise BenchmarkDiscoveryError(
            f"Builder entrypoint must resolve to a BenchmarkBuilder instance or class: {spec.entrypoint}"
        )
    return builder


def load_group_analyzer(entrypoint: str) -> GroupAnalyzer:
    analyzer = _load_entrypoint_object(entrypoint)
    if isinstance(analyzer, type):
        analyzer = analyzer()
    if not isinstance(analyzer, GroupAnalyzer):
        raise BenchmarkDiscoveryError(
            f"Group analyzer entrypoint must resolve to a GroupAnalyzer instance or class: {entrypoint}"
        )
    return analyzer


def load_example_normalizer(normalizer_id: str, *, examples_root: Path = EXAMPLES_ROOT) -> ResultNormalizer:
    specs = discover_example_normalizer_specs(examples_root)
    try:
        spec = specs[normalizer_id]
    except KeyError as exc:
        raise BenchmarkDiscoveryError(f"Unknown example normalizer: {normalizer_id}") from exc
    if not spec.entrypoint:
        raise BenchmarkDiscoveryError(f"Example normalizer has no entrypoint: {normalizer_id}")
    normalizer = _load_entrypoint_object(spec.entrypoint)
    if isinstance(normalizer, type):
        normalizer = normalizer()
    if not isinstance(normalizer, ResultNormalizer):
        raise BenchmarkDiscoveryError(
            f"Normalizer entrypoint must resolve to a ResultNormalizer instance or class: {spec.entrypoint}"
        )
    return normalizer


def load_analysis_plugin(plugin_id: str, *, examples_root: Path = EXAMPLES_ROOT) -> AnalysisPlugin:
    specs = discover_example_analysis_plugin_specs(examples_root)
    try:
        spec = specs[plugin_id]
    except KeyError as exc:
        raise BenchmarkDiscoveryError(f"Unknown analysis plugin: {plugin_id}") from exc
    if not spec.entrypoint:
        raise BenchmarkDiscoveryError(f"Analysis plugin has no entrypoint: {plugin_id}")
    plugin = _load_entrypoint_object(spec.entrypoint)
    if isinstance(plugin, type):
        plugin = plugin()
    if not isinstance(plugin, AnalysisPlugin):
        raise BenchmarkDiscoveryError(
            f"Analysis plugin entrypoint must resolve to an AnalysisPlugin instance or class: {spec.entrypoint}"
        )
    return plugin


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchmarkDiscoveryError(f"Manifest does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise BenchmarkDiscoveryError(f"PyYAML is required to parse manifest: {path}") from exc
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise BenchmarkDiscoveryError(f"Manifest must be an object: {path}")
    return payload


def _load_entrypoint_object(entrypoint: str) -> Any:
    if ":" not in entrypoint:
        raise BenchmarkDiscoveryError(
            f"Entrypoint must look like module:object or /path/to/file.py:object: {entrypoint}"
        )
    module_ref, object_name = entrypoint.split(":", 1)
    module_ref = module_ref.strip()
    object_name = object_name.strip()
    if not module_ref or not object_name:
        raise BenchmarkDiscoveryError(f"Invalid entrypoint: {entrypoint}")
    if module_ref.endswith(".py") or Path(module_ref).exists():
        module_path = Path(module_ref)
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise BenchmarkDiscoveryError(f"Unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise BenchmarkDiscoveryError(f"Entrypoint object not found: {entrypoint}") from exc
