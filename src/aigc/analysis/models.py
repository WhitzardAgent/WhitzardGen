from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisPluginSpec:
    plugin_id: str
    plugin_type: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
