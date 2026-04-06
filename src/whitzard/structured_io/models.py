from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TemplateSpec:
    name: str
    version: str = "v1"
    path: str | None = None
    template_text: str = ""
    variable_allowlist: list[str] = field(default_factory=list)
    helpers: list[str] = field(default_factory=list)
    missing_variable_policy: str = "warn_and_empty"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredOutputSpec:
    format_type: str = "plain_text"
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    fallback_patterns: dict[str, list[str]] = field(default_factory=dict)
    normalization_rules: dict[str, Any] = field(default_factory=dict)
    reasoning_capture: dict[str, Any] = field(default_factory=dict)
    parse_mode_map: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_type": self.format_type,
            "fields": dict(self.fields),
            "required_fields": list(self.required_fields),
            "aliases": {key: list(value) for key, value in self.aliases.items()},
            "fallback_patterns": {
                key: list(value) for key, value in self.fallback_patterns.items()
            },
            "normalization_rules": dict(self.normalization_rules),
            "reasoning_capture": dict(self.reasoning_capture),
            "parse_mode_map": dict(self.parse_mode_map),
            "raw": dict(self.raw),
        }


@dataclass(slots=True)
class StructuredParseResult:
    format_type: str
    fields: dict[str, Any] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    parse_status: str = "parsed"
    raw_hits: dict[str, Any] = field(default_factory=dict)
    fallback_source: dict[str, str] = field(default_factory=dict)
    reasoning_trace: str | None = None
    reasoning_source: str = "none"
    raw_payload: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
