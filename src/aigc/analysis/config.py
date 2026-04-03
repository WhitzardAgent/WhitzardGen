from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc.analysis.models import AnalysisPluginSpec
from aigc.benchmarking.discovery import discover_example_analysis_plugin_specs


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_CONFIG_PATH = REPO_ROOT / "configs" / "analysis_plugins"


class AnalysisConfigError(ValueError):
    """Raised when analysis plugin config cannot be loaded or resolved."""


def load_analysis_plugin_catalog(path: str | Path = DEFAULT_ANALYSIS_CONFIG_PATH) -> dict[str, AnalysisPluginSpec]:
    config_root = Path(path)
    profiles_path = config_root / "profiles.yaml"
    payload = _load_yaml_or_json(profiles_path)
    profiles_payload = payload.get("plugins", payload)
    if not isinstance(profiles_payload, dict):
        raise AnalysisConfigError(f"Analysis plugin config must define an object mapping: {profiles_path}")
    plugins: dict[str, AnalysisPluginSpec] = {}
    for plugin_id, config in profiles_payload.items():
        if not isinstance(config, dict):
            raise AnalysisConfigError(f"Analysis plugin {plugin_id} must be an object.")
        plugins[str(plugin_id)] = AnalysisPluginSpec(
            plugin_id=str(plugin_id),
            plugin_type=str(config.get("type", "comparative")),
            description=str(config.get("description", "")),
            dependencies=[str(item) for item in config.get("dependencies", []) or []],
            config=dict(config.get("config", {}) or {}),
            version=str(config.get("version", "v1")),
        )
    for plugin_id, spec in discover_example_analysis_plugin_specs().items():
        plugins[plugin_id] = AnalysisPluginSpec(
            plugin_id=spec.plugin_id,
            plugin_type=spec.plugin_type,
            description=spec.description,
            dependencies=list(spec.dependencies),
            config=dict(spec.config),
            version=spec.version,
        )
    return plugins


def resolve_analysis_plugins(
    plugin_ids: list[str] | None,
    *,
    path: str | Path | None = DEFAULT_ANALYSIS_CONFIG_PATH,
) -> list[AnalysisPluginSpec]:
    catalog = load_analysis_plugin_catalog(path or DEFAULT_ANALYSIS_CONFIG_PATH)
    if not plugin_ids:
        return []
    resolved: list[AnalysisPluginSpec] = []
    for plugin_id in plugin_ids:
        try:
            resolved.append(catalog[plugin_id])
        except KeyError as exc:
            raise AnalysisConfigError(f"Unknown analysis plugin: {plugin_id}") from exc
    return resolved


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise AnalysisConfigError(f"PyYAML is required to parse analysis config: {path}") from exc
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise AnalysisConfigError(f"Analysis config must be an object: {path}")
    return payload
