from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskPrompt:
    prompt_id: str
    prompt: str
    language: str
    negative_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskPrompt":
        return cls(
            prompt_id=str(payload["prompt_id"]),
            prompt=str(payload["prompt"]),
            language=str(payload["language"]),
            negative_prompt=payload.get("negative_prompt"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class TaskPayload:
    task_id: str
    model_name: str
    execution_mode: str
    prompts: list[TaskPrompt]
    params: dict[str, Any]
    workdir: str
    batch_id: str | None = None
    runtime_config: dict[str, Any] = field(default_factory=dict)
    worker_strategy: str = "per_task_worker"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prompts"] = [prompt.to_dict() for prompt in self.prompts]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskPayload":
        return cls(
            task_id=str(payload["task_id"]),
            model_name=str(payload["model_name"]),
            execution_mode=str(payload.get("execution_mode", "real")),
            worker_strategy=str(payload.get("worker_strategy", "per_task_worker")),
            prompts=[TaskPrompt.from_dict(item) for item in payload.get("prompts", [])],
            params=dict(payload.get("params", {})),
            workdir=str(payload["workdir"]),
            batch_id=payload.get("batch_id"),
            runtime_config=dict(payload.get("runtime_config", {})),
        )
