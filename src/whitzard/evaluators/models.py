from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluatorSpec:
    evaluator_id: str
    evaluator_type: str
    description: str = ""
    accepted_input_types: list[str] = field(default_factory=list)
    rule_type: str | None = None
    rule_config: dict[str, Any] = field(default_factory=dict)
    judge_model: str | None = None
    annotation_profile: str | None = None
    annotation_template: str | None = None
    prompt_template: dict[str, Any] = field(default_factory=dict)
    output_spec: dict[str, Any] = field(default_factory=dict)
    generation_defaults: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scorer_id(self) -> str:
        return self.evaluator_id
