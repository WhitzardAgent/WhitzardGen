from __future__ import annotations

import json
import os
from pathlib import Path

from aigc.adapters import ADAPTER_REGISTRY
from aigc.registry.local_overrides import (
    LOCAL_MODELS_ENV_VAR,
    LOCAL_OVERRIDE_FIELDS,
    LOCAL_RUNTIME_OVERRIDE_FIELDS,
    LocalOverrideError,
    load_local_model_overrides,
)
from aigc.registry.models import ModelInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "models.yaml"
DEFAULT_LOCAL_MODELS_PATH = REPO_ROOT / "configs" / "local_models.yaml"


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
    raw = registry_path.read_text(encoding="utf-8")
    payload = _parse_registry_payload(raw)
    resolved_local_models_path = _resolve_local_models_path(local_models_path)
    try:
        local_override_source, local_overrides = load_local_model_overrides(
            resolved_local_models_path
        )
    except LocalOverrideError as exc:
        raise RegistryError(str(exc)) from exc

    models_payload = payload.get("models")
    if not isinstance(models_payload, dict):
        raise RegistryError("Registry file must define a top-level 'models' mapping.")

    models: dict[str, ModelInfo] = {}
    for name, config in models_payload.items():
        if not isinstance(config, dict):
            raise RegistryError(f"Registry entry for {name} must be an object.")
        local_paths = dict(local_overrides.get(name, {}))
        weights = dict(config.get("weights", {}))
        runtime = dict(config.get("runtime", {}))
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
        model = ModelInfo(
            name=name,
            version=str(config.get("version", "")),
            adapter=str(config.get("adapter", "")),
            modality=str(config.get("modality", "")),
            task_type=str(config.get("task_type", "")),
            capabilities=dict(config.get("capabilities", {})),
            runtime=runtime,
            weights=weights,
            local_paths=local_paths,
            registry_source=str(registry_path),
            local_override_source=str(local_override_source) if local_paths else None,
        )
        _validate_model_info(model)
        models[name] = model

    registry = ModelRegistry(
        models,
        registry_path=registry_path,
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


def _parse_registry_payload(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RegistryError(
                "Registry file is not valid JSON and PyYAML is not installed for YAML parsing."
            ) from exc
        payload = yaml.safe_load(raw)
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
