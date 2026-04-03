from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from aigc.launching.models import LaunchPlan, ServiceHealth


@dataclass(slots=True)
class LaunchRequest:
    requested_models: list[str]
    config_path: str | Path | None = None
    auto_launch: bool = False


class LauncherBackend(ABC):
    backend_id: str

    @abstractmethod
    def plan(self, request: LaunchRequest) -> LaunchPlan:
        raise NotImplementedError

    @abstractmethod
    def health(self, request: LaunchRequest) -> list[ServiceHealth]:
        raise NotImplementedError
