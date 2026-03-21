from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aigc.adapters import ADAPTER_REGISTRY
from aigc.registry.local_overrides import (
    LOCAL_MODELS_ENV_VAR,
    LOCAL_NESTED_OVERRIDE_FIELDS,
    LOCAL_OVERRIDE_FIELDS,
    LOCAL_RUNTIME_OVERRIDE_FIELDS,
    LocalOverrideError,
    load_local_model_overrides,
)
from aigc.registry.models import ModelInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "models"
DEFAULT_LOCAL_MODELS_PATH = REPO_ROOT / "configs" / "local_models"


class RegistryError(ValueError):
    """Raised when the model registry is invalid or cannot be loaded."""


class ModelRegistry:
    def __init__(
        self,
        models: dict[str, ModelInfo],
        *,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        local_models_path: Path | None = None,
    ) -> None:
        self._models = dict(sorted(models.items(), key=lambda item: item[0].lower()))
        self.registry_path = registry_path
        self.local_models_path = local_models_path

    def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def get_model(self, name: str) -> ModelInfo:
        try:
            return self._models[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown model: {name}") from exc

    def get_models_by_modality(self, modality: str) -> list[ModelInfo]:
        return [model for model in self._models.values() if model.modality == modality]

    def get_models_by_task(self, task_type: str) -> list[ModelInfo]:
        return [model for model in self._models.values() if model.task_type == task_type]

    def resolve_adapter_class(self, name: str) -> type:
        model = self.get_model(name)
        try:
            return ADAPTER_REGISTRY[model.adapter]
        except KeyError as exc:
            raise RegistryError(
                f"Model {name} references unknown adapter class: {model.adapter}"
            ) from exc

    def instantiate_adapter(self, name: str):
        return self.resolve_adapter_class(name)(model_config=self.get_model(name))


def load_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    local_models_path: str | Path | None = None,
) -> ModelRegistry:
    registry_path = Path(path)
    registry_source, models_payload = _load_registry_models(registry_path)
    resolved_local_models_path = _resolve_local_models_path(local_models_path)
    try:
        local_override_source, local_overrides = load_local_model_overrides(
            resolved_local_models_path
        )
    except LocalOverrideError as exc:
        raise RegistryError(str(exc)) from exc

    models: dict[str, ModelInfo] = {}
    for name, config in models_payload.items():
        if not isinstance(config, dict):
            raise RegistryError(f"Registry entry for {name} must be an object.")
        local_paths = _prune_redundant_local_overrides(
            config=config,
            overrides=dict(local_overrides.get(name, {})),
        )
        weights = dict(config.get("weights", {}))
        runtime = dict(config.get("runtime", {}))
        generation_defaults = dict(config.get("generation_defaults", {}))
        weights.update(
            {
                key: value
                for key, value in local_paths.items()
                if key in LOCAL_OVERRIDE_FIELDS
            }
        )
        runtime.update(
            {
                key: value
                for key, value in local_paths.items()
                if key in LOCAL_RUNTIME_OVERRIDE_FIELDS
            }
        )
        nested_override = local_paths.get("generation_defaults")
        if isinstance(nested_override, dict):
            generation_defaults.update(nested_override)
        model = ModelInfo(
            name=name,
            version=str(config.get("version", "")),
            adapter=str(config.get("adapter", "")),
            modality=str(config.get("modality", "")),
            task_type=str(config.get("task_type", "")),
            capabilities=dict(config.get("capabilities", {})),
            runtime=runtime,
            weights=weights,
            generation_defaults=generation_defaults,
            local_paths=local_paths,
            registry_source=str(registry_source),
            local_override_source=str(local_override_source) if local_paths else None,
        )
        _validate_model_info(model)
        models[name] = model

    registry = ModelRegistry(
        models,
        registry_path=registry_source,
        local_models_path=local_override_source,
    )
    for model_name in models:
        registry.resolve_adapter_class(model_name)
    return registry


def _resolve_local_models_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    env_override = os.environ.get(LOCAL_MODELS_ENV_VAR)
    if env_override:
        return Path(env_override)
    return DEFAULT_LOCAL_MODELS_PATH


def _load_registry_models(path: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    if not path.exists():
        raise RegistryError(f"Registry path does not exist: {path}")
    if path.is_dir():
        return path, _load_registry_directory(path)
    payload = _parse_registry_payload(path.read_text(encoding="utf-8"), source_path=path)
    return path, _extract_models_payload(payload, source_label=str(path))


def _load_registry_directory(path: Path) -> dict[str, dict[str, Any]]:
    files = _iter_registry_files(path)
    if not files:
        raise RegistryError(f"Registry directory contains no YAML/JSON fragments: {path}")
    models: dict[str, dict[str, Any]] = {}
    for file_path in files:
        payload = _parse_registry_payload(
            file_path.read_text(encoding="utf-8"),
            source_path=file_path,
        )
        fragment_models = _extract_models_payload(payload, source_label=str(file_path))
        for model_name, config in fragment_models.items():
            if model_name in models:
                raise RegistryError(
                    f"Duplicate model entry {model_name} found in registry fragment {file_path}."
                )
            models[model_name] = config
    return models


def _iter_registry_files(path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml", ".json"}
        ],
        key=lambda candidate: candidate.name.lower(),
    )


def _extract_models_payload(
    payload: dict[str, Any],
    *,
    source_label: str,
) -> dict[str, dict[str, Any]]:
    models_payload = payload.get("models")
    if not isinstance(models_payload, dict):
        raise RegistryError(
            f"Registry source must define a top-level 'models' mapping: {source_label}"
        )
    return models_payload


def _parse_registry_payload(raw: str, *, source_path: Path | None = None) -> dict:
    suffix = source_path.suffix.lower() if source_path is not None else ""
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml_registry_payload(raw)
    if suffix == ".json":
        return _parse_json_registry_payload(raw)
    try:
        return _parse_yaml_registry_payload(raw)
    except RegistryError:
        return _parse_json_registry_payload(raw)


def _parse_yaml_registry_payload(raw: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RegistryError("PyYAML is required to parse YAML registry files.") from exc
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise RegistryError("Registry payload must be an object.")
    return payload


def _parse_json_registry_payload(raw: str) -> dict:
    import json

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError("Registry file is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RegistryError("Registry payload must be an object.")
    return payload


def _validate_model_info(model: ModelInfo) -> None:
    required = {
        "version": model.version,
        "adapter": model.adapter,
        "modality": model.modality,
        "task_type": model.task_type,
    }
    missing = [field for field, value in required.items() if not value]
    if missing:
        raise RegistryError(f"Model {model.name} is missing required fields: {', '.join(missing)}")
    if model.adapter not in ADAPTER_REGISTRY:
        raise RegistryError(f"Model {model.name} references unknown adapter {model.adapter}")
    if "execution_mode" not in model.runtime:
        raise RegistryError(f"Model {model.name} is missing runtime.execution_mode")
    if "env_spec" not in model.runtime:
        raise RegistryError(f"Model {model.name} is missing runtime.env_spec")


def _prune_redundant_local_overrides(
    *,
    config: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    if not overrides:
        return {}
    runtime = dict(config.get("runtime", {})) if isinstance(config.get("runtime", {}), dict) else {}
    weights = dict(config.get("weights", {})) if isinstance(config.get("weights", {}), dict) else {}
    pruned: dict[str, object] = {}
    for key, value in overrides.items():
        if key in LOCAL_OVERRIDE_FIELDS:
            if weights.get(key) == value:
                continue
        elif key in LOCAL_RUNTIME_OVERRIDE_FIELDS:
            default_value = runtime.get(key)
            if key == "conda_env_name" and default_value in (None, ""):
                default_value = runtime.get("env_spec")
            if default_value == value:
                continue
        elif key in LOCAL_NESTED_OVERRIDE_FIELDS:
            default_value = config.get(key, {})
            if default_value == value:
                continue
        pruned[str(key)] = value
    return pruned
