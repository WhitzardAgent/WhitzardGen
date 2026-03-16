"""Model registry subsystem."""

from aigc.registry.loader import (
    DEFAULT_LOCAL_MODELS_PATH,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
    RegistryError,
    load_registry,
)
from aigc.registry.local_overrides import LOCAL_MODELS_ENV_VAR, LOCAL_OVERRIDE_FIELDS
from aigc.registry.models import ModelInfo

__all__ = [
    "DEFAULT_LOCAL_MODELS_PATH",
    "DEFAULT_REGISTRY_PATH",
    "LOCAL_MODELS_ENV_VAR",
    "LOCAL_OVERRIDE_FIELDS",
    "ModelInfo",
    "ModelRegistry",
    "RegistryError",
    "load_registry",
]
