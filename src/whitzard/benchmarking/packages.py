from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class BenchmarkPackageError(RuntimeError):
    """Raised when a benchmark package cannot be resolved or parsed."""


@dataclass(slots=True)
class SlotDefinition:
    slot_id: str
    layer: str
    type: str
    description: str
    analysis_contribution: list[str] = field(default_factory=list)
    theoretical_grounding: list[str] = field(default_factory=list)
    value_space: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    surface_realization: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerativeBenchmarkPackage:
    package_path: str
    canonical_package_path: str
    manifest: dict[str, Any]
    slot_library: dict[str, SlotDefinition]
    slot_library_payload: dict[str, Any]
    analysis_codebook: dict[str, Any]
    schema: dict[str, Any]
    templates: list[dict[str, Any]]
    template_paths: list[str]
    alias_path: str | None = None


def resolve_benchmark_package_path(package_path: str | Path) -> tuple[Path, Path | None]:
    target = Path(package_path).resolve()
    if not target.exists():
        raise BenchmarkPackageError(f"Benchmark package path does not exist: {target}")
    alias_manifest_path = target / "package_alias.yaml"
    if not alias_manifest_path.exists():
        return target, None
    payload = _load_yaml_file(alias_manifest_path)
    canonical_value = payload.get("canonical_package_path") or payload.get("alias_to")
    if canonical_value in (None, ""):
        raise BenchmarkPackageError(
            f"Benchmark package alias file is missing canonical_package_path: {alias_manifest_path}"
        )
    canonical_path = Path(str(canonical_value))
    if not canonical_path.is_absolute():
        canonical_path = (target / canonical_path).resolve()
    if not canonical_path.exists():
        raise BenchmarkPackageError(
            f"Benchmark package alias target does not exist: {canonical_path}"
        )
    return canonical_path, target


def load_generative_benchmark_package(package_path: str | Path) -> GenerativeBenchmarkPackage:
    canonical_path, alias_path = resolve_benchmark_package_path(package_path)
    manifest = _load_yaml_file(canonical_path / "manifest.yaml")
    slot_library_payload = _load_yaml_file(canonical_path / "slot_library.yaml")
    analysis_codebook = _load_yaml_file(canonical_path / "analysis_codebook.yaml")
    schema = _load_yaml_file(canonical_path / "schema.yaml")
    templates_dir = canonical_path / "templates"
    if not templates_dir.exists():
        raise BenchmarkPackageError(f"Benchmark package is missing templates/: {templates_dir}")
    template_paths = [str(path) for path in sorted(templates_dir.glob("*.yaml"))]
    templates = [_load_yaml_file(Path(path)) for path in template_paths]
    slot_library = _normalize_slot_library(slot_library_payload)
    return GenerativeBenchmarkPackage(
        package_path=str(alias_path or canonical_path),
        canonical_package_path=str(canonical_path),
        manifest=manifest,
        slot_library=slot_library,
        slot_library_payload=slot_library_payload,
        analysis_codebook=analysis_codebook,
        schema=schema,
        templates=templates,
        template_paths=template_paths,
        alias_path=str(alias_path) if alias_path is not None else None,
    )


def _normalize_slot_library(payload: dict[str, Any]) -> dict[str, SlotDefinition]:
    raw_slots = payload.get("slots", payload)
    normalized: dict[str, SlotDefinition] = {}
    if isinstance(raw_slots, list):
        raw_slots = {
            str(item.get("slot_id") or ""): dict(item or {})
            for item in raw_slots
            if isinstance(item, dict) and item.get("slot_id") not in (None, "")
        }
    if not isinstance(raw_slots, dict):
        raise BenchmarkPackageError("slot_library.yaml must define a slot mapping or slot list.")
    for slot_id, raw in raw_slots.items():
        if not isinstance(raw, dict):
            raise BenchmarkPackageError(f"Slot definition must be an object: {slot_id}")
        normalized[str(slot_id)] = SlotDefinition(
            slot_id=str(raw.get("slot_id") or slot_id),
            layer=str(raw.get("layer") or "structural"),
            type=str(raw.get("type") or "enum"),
            description=str(raw.get("description") or ""),
            analysis_contribution=[str(item) for item in raw.get("analysis_contribution", []) or []],
            theoretical_grounding=[str(item) for item in raw.get("theoretical_grounding", []) or []],
            value_space=dict(raw.get("value_space", {}) or {}),
            sampling=dict(raw.get("sampling", {}) or {}),
            surface_realization=dict(raw.get("surface_realization", {}) or {}),
            metadata={
                key: value
                for key, value in dict(raw).items()
                if key
                not in {
                    "slot_id",
                    "layer",
                    "type",
                    "description",
                    "analysis_contribution",
                    "theoretical_grounding",
                    "value_space",
                    "sampling",
                    "surface_realization",
                }
            },
        )
    return normalized


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise BenchmarkPackageError(f"YAML file must parse to an object: {path}")
    return payload
