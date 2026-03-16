from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelInfo:
    name: str
    version: str
    adapter: str
    modality: str
    task_type: str
    capabilities: dict[str, Any]
    runtime: dict[str, Any]
    weights: dict[str, Any]
    local_paths: dict[str, Any] = field(default_factory=dict)
    registry_source: str | None = None
    local_override_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def execution_mode(self) -> str:
        return str(self.runtime.get("execution_mode", "unknown"))

    @property
    def env_spec(self) -> str:
        return str(self.runtime.get("env_spec", ""))

    @property
    def backend_execution_mode(self) -> str:
        return self.execution_mode

    @property
    def has_local_overrides(self) -> bool:
        return bool(self.local_paths)
