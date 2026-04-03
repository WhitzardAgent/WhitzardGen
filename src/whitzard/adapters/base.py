from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from whitzard.registry.models import ModelInfo

ExecutionMode = Literal["external_process", "in_process"]
WorkerStrategy = Literal["per_task_worker", "persistent_worker"]
ArtifactType = Literal["image", "video", "audio", "text", "json"]
BatchStatus = Literal["success", "failed"]
ModelStatus = Literal["success", "partial_success", "failed"]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class ExecutionPlan:
    mode: ExecutionMode
    command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_sec: int | None = None
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactRecord:
    type: ArtifactType
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionResult:
    exit_code: int
    logs: str
    outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchItemResult:
    prompt_id: str
    artifacts: list[ArtifactRecord]
    status: BatchStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload


@dataclass(slots=True)
class ModelResult:
    status: ModelStatus
    batch_items: list[BatchItemResult]
    logs: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["batch_items"] = [item.to_dict() for item in self.batch_items]
        return payload


@dataclass(slots=True)
class AdapterCapabilities:
    supports_batch_prompts: bool = False
    max_batch_size: int = 1
    preferred_batch_size: int = 1
    supports_negative_prompt: bool = False
    supports_seed: bool = True
    output_types: list[str] = field(default_factory=list)
    supports_persistent_worker: bool = False
    preferred_worker_strategy: WorkerStrategy = "per_task_worker"


class BaseAdapter(ABC):
    capabilities = AdapterCapabilities()

    def __init__(self, model_config: "ModelInfo") -> None:
        self.model_config = model_config

    def load_for_persistent_worker(self) -> None:
        """Warm the adapter for a long-lived worker process."""

    def unload_persistent_worker(self) -> None:
        """Release any persistent-worker state."""

    @abstractmethod
    def prepare(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
    ) -> ExecutionPlan:
        raise NotImplementedError

    def execute(
        self,
        plan: ExecutionPlan,
        prompts: list[str],
        params: dict[str, Any],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        del progress_callback
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement in-process execution."
        )

    @abstractmethod
    def collect(
        self,
        plan: ExecutionPlan,
        exec_result: ExecutionResult,
        prompts: list[str],
        prompt_ids: list[str],
        workdir: str,
    ) -> ModelResult:
        raise NotImplementedError
