from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ThemeNode:
    name: str
    count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    children: list["ThemeNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass(slots=True)
class ThemeTree:
    version: str
    name: str
    defaults: dict[str, Any]
    categories: list[ThemeNode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "defaults": dict(self.defaults),
            "categories": [category.to_dict() for category in self.categories],
        }


@dataclass(slots=True)
class SampledTheme:
    sample_id: str
    quota_source_path: tuple[str, ...]
    theme_path: tuple[str, ...]
    category: str
    subcategory: str
    subtopic: str
    theme: str
    resampled: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "quota_source_path": list(self.quota_source_path),
            "theme_path": list(self.theme_path),
            "category": self.category,
            "subcategory": self.subcategory,
            "subtopic": self.subtopic,
            "theme": self.theme,
            "resampled": self.resampled,
            "metadata": dict(self.metadata),
            "constraints": dict(self.constraints),
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class PromptGenerationSummary:
    bundle_id: str
    bundle_dir: str
    prompts_path: str
    manifest_path: str
    sampling_plan_path: str
    generation_log_path: str
    stats_path: str
    prompt_count: int
    llm_model: str | None
    execution_mode: str
    prompt_template: str | None = None
    prompt_style_family: str | None = None
    target_model_name: str | None = None
    few_shot_example_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptTemplateConfig:
    name: str
    version: str
    instruction_template: str
    default_style_family: str | None = None


@dataclass(slots=True)
class PromptStyleFamilyConfig:
    name: str
    version: str
    description: str
    style_instruction: str
    output_contract: dict[str, Any]
    few_shot_examples: list[dict[str, Any]]
    max_examples_per_request: int
    supported_modalities: list[str] = field(default_factory=list)
