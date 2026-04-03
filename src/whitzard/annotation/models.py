from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AnnotationTemplateConfig:
    name: str
    version: str
    instruction_template: str


@dataclass(slots=True)
class AnnotationProfileConfig:
    name: str
    version: str
    default_model: str | None
    default_template: str | None
    generation_defaults: dict[str, Any]
    output_contract: dict[str, Any]
    accepted_source_artifact_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnnotationBundleSummary:
    bundle_id: str
    bundle_dir: str
    annotations_path: str
    manifest_path: str
    log_path: str
    stats_path: str
    failures_path: str
    source_run_id: str
    annotator_model: str
    annotation_profile: str
    annotation_template: str
    source_record_count: int
    annotated_count: int
    skipped_count: int
    failed_count: int
    annotation_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
