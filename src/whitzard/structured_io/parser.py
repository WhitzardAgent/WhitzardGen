from __future__ import annotations

import json
import re
from typing import Any

from whitzard.structured_io.models import StructuredOutputSpec, StructuredParseResult


def resolve_output_spec(config: dict[str, Any] | None) -> StructuredOutputSpec:
    payload = dict(config or {})
    format_type = str(payload.get("format_type") or "").strip()
    if not format_type:
        format_type = _infer_format_type_from_legacy_payload(payload)

    raw_fields = payload.get("fields", {})
    fields: dict[str, dict[str, Any]] = {}
    aliases: dict[str, list[str]] = {}
    required_fields = [
        str(item).strip()
        for item in payload.get("required_fields", []) or []
        if str(item).strip()
    ]
    for field_name, config_value in dict(raw_fields or {}).items():
        if isinstance(config_value, dict):
            field_config = dict(config_value)
        else:
            field_config = {}
        fields[str(field_name)] = field_config
        field_aliases = [
            str(item).strip()
            for item in field_config.get("aliases", []) or []
            if str(item).strip()
        ]
        if field_aliases:
            aliases[str(field_name)] = field_aliases
        if field_config.get("required") is True and str(field_name) not in required_fields:
            required_fields.append(str(field_name))
    for field_name, raw_aliases in dict(payload.get("aliases", {}) or {}).items():
        aliases[str(field_name)] = [
            str(item).strip()
            for item in raw_aliases or []
            if str(item).strip()
        ]
    return StructuredOutputSpec(
        format_type=format_type or "plain_text",
        fields=fields,
        required_fields=required_fields,
        aliases=aliases,
        fallback_patterns={
            str(key): [str(item) for item in value or []]
            for key, value in dict(payload.get("fallback_patterns", {}) or {}).items()
        },
        normalization_rules=dict(payload.get("normalization_rules", {}) or {}),
        reasoning_capture=dict(payload.get("reasoning_capture", {}) or {}),
        parse_mode_map=dict(payload.get("parse_mode_map", {}) or {}),
        raw=payload,
    )


class StructuredParser:
    def resolve(self, config: dict[str, Any] | None) -> StructuredOutputSpec:
        return resolve_output_spec(config)

    def parse(
        self,
        raw_text: str,
        *,
        output_spec: StructuredOutputSpec | dict[str, Any] | None = None,
        artifact_metadata: dict[str, Any] | None = None,
        parse_mode: str | None = None,
    ) -> StructuredParseResult:
        return parse_structured_output(
            raw_text,
            output_spec=output_spec,
            artifact_metadata=artifact_metadata,
            parse_mode=parse_mode,
        )


class StructuredContractValidator:
    def validate(
        self,
        result: StructuredParseResult,
        *,
        output_spec: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> list[str]:
        return validate_structured_output(result, output_spec=output_spec)


def build_json_object_output_spec(
    required_keys: list[str] | None = None,
    *,
    aliases: dict[str, list[str]] | None = None,
) -> StructuredOutputSpec:
    return StructuredOutputSpec(
        format_type="json_object",
        fields={key: {} for key in list(required_keys or [])},
        required_fields=[str(key) for key in list(required_keys or [])],
        aliases=dict(aliases or {}),
    )


def output_contract_to_spec(contract: dict[str, Any] | None) -> StructuredOutputSpec:
    payload = dict(contract or {})
    format_name = str(payload.get("format") or "json").strip().lower()
    required_keys = [
        str(item).strip()
        for item in payload.get("required_keys", []) or []
        if str(item).strip()
    ]
    if format_name in {"json", "json_object"}:
        return build_json_object_output_spec(required_keys)
    return StructuredOutputSpec(format_type="plain_text", required_fields=required_keys)


def parse_structured_output(
    raw_text: str,
    *,
    output_spec: StructuredOutputSpec | dict[str, Any] | None = None,
    artifact_metadata: dict[str, Any] | None = None,
    parse_mode: str | None = None,
) -> StructuredParseResult:
    spec = output_spec if isinstance(output_spec, StructuredOutputSpec) else resolve_output_spec(output_spec)
    format_type = spec.format_type
    if format_type == "json_object":
        return _parse_json_object(raw_text, spec=spec, artifact_metadata=artifact_metadata)
    if format_type == "tag_blocks":
        return _parse_tag_blocks(
            raw_text,
            spec=spec,
            artifact_metadata=artifact_metadata,
            parse_mode=parse_mode,
        )
    if format_type == "markdown_sections":
        return _parse_markdown_sections(
            raw_text,
            spec=spec,
            artifact_metadata=artifact_metadata,
            parse_mode=parse_mode,
        )
    return _parse_plain_text(raw_text, spec=spec, artifact_metadata=artifact_metadata)


def validate_structured_output(
    result: StructuredParseResult,
    *,
    output_spec: StructuredOutputSpec | dict[str, Any] | None = None,
) -> list[str]:
    spec = output_spec if isinstance(output_spec, StructuredOutputSpec) else resolve_output_spec(output_spec)
    errors: list[str] = []
    for field_name in list(spec.required_fields):
        value = result.fields.get(field_name)
        if value in (None, "", []):
            errors.append(f"Missing required structured field: {field_name}")
    return errors


def normalize_choice(value: Any, *, rules: dict[str, Any] | None = None) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    choice_aliases = dict((rules or {}).get("choice_aliases", {}) or {})
    for canonical, aliases in choice_aliases.items():
        alias_set = {str(item).strip().lower() for item in aliases or [] if str(item).strip()}
        if normalized.lower() in alias_set:
            return str(canonical)
    compact = normalized.upper()
    if compact in {"A", "B"}:
        return compact
    return normalized or None


def extract_reasoning_trace(
    raw_text: str,
    *,
    artifact_metadata: dict[str, Any] | None = None,
    spec: StructuredOutputSpec | dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    resolved_spec = spec if isinstance(spec, StructuredOutputSpec) else resolve_output_spec(spec)
    capture = dict(resolved_spec.reasoning_capture or {})
    metadata_keys = [
        str(item).strip()
        for item in capture.get("metadata_keys", ["thinking_content"]) or []
        if str(item).strip()
    ]
    artifact_payload = dict(artifact_metadata or {})
    for key in metadata_keys:
        value = artifact_payload.get(key)
        if value not in (None, ""):
            normalized = str(value).strip()
            if normalized:
                return normalized, "artifact_metadata"

    tag_fields = [
        str(item).strip()
        for item in capture.get("tag_fields", ["thinking"]) or []
        if str(item).strip()
    ]
    for field_name in tag_fields:
        aliases = resolved_spec.aliases.get(field_name, [field_name])
        value = _extract_tag_by_aliases(raw_text, aliases)
        if value:
            return value, "tag"

    fallback_patterns = list(capture.get("fallback_patterns", []) or [])
    for pattern in fallback_patterns:
        match = re.search(str(pattern), raw_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:4000], "text_pattern"
    return None, "none"


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    result = _parse_json_object(raw_text, spec=StructuredOutputSpec(format_type="json_object"))
    if isinstance(result.raw_payload, dict):
        return dict(result.raw_payload)
    return None


def extract_text_value_from_json(
    raw_text: str,
    *,
    candidate_keys: list[str],
) -> str:
    payload = extract_json_object(raw_text)
    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return str(raw_text).strip()


def _parse_json_object(
    raw_text: str,
    *,
    spec: StructuredOutputSpec,
    artifact_metadata: dict[str, Any] | None = None,
) -> StructuredParseResult:
    text = str(raw_text).strip()
    payload = _extract_json_payload(text)
    if payload is None:
        return StructuredParseResult(
            format_type="json_object",
            fields={},
            missing_required=list(spec.required_fields),
            parse_status="invalid",
            raw_payload=text,
        )
    fields = dict(payload)
    for field_name, aliases in spec.aliases.items():
        if field_name in fields and fields.get(field_name) not in (None, ""):
            continue
        for alias in aliases:
            if alias in fields and fields.get(alias) not in (None, ""):
                fields[field_name] = fields.get(alias)
                break
    reasoning_trace, reasoning_source = extract_reasoning_trace(
        text,
        artifact_metadata=artifact_metadata,
        spec=spec,
    )
    if reasoning_trace and "thinking" not in fields:
        fields["thinking"] = reasoning_trace
    missing_required = [
        field_name
        for field_name in spec.required_fields
        if fields.get(field_name) in (None, "", [])
    ]
    return StructuredParseResult(
        format_type="json_object",
        fields=fields,
        missing_required=missing_required,
        parse_status="parsed" if not missing_required else "partial",
        raw_hits={key: fields.get(key) for key in fields},
        reasoning_trace=reasoning_trace,
        reasoning_source=reasoning_source,
        raw_payload=payload,
    )


def _parse_tag_blocks(
    raw_text: str,
    *,
    spec: StructuredOutputSpec,
    artifact_metadata: dict[str, Any] | None = None,
    parse_mode: str | None = None,
) -> StructuredParseResult:
    text = str(raw_text).strip()
    raw_hits: dict[str, Any] = {}
    fallback_source: dict[str, str] = {}
    fields: dict[str, Any] = {}
    for field_name in _iter_known_fields(spec):
        aliases = spec.aliases.get(field_name, [field_name])
        value = _extract_tag_by_aliases(text, aliases)
        if value is not None:
            fields[field_name] = value
            raw_hits[field_name] = value
            continue
        fallback_value = _extract_fallback_value(
            text=text,
            field_name=field_name,
            patterns=spec.fallback_patterns.get(field_name, []),
        )
        if fallback_value is not None:
            fields[field_name] = fallback_value
            fallback_source[field_name] = "pattern"
    if "final_choice" in fields:
        normalized_choice = normalize_choice(
            fields.get("final_choice"),
            rules=spec.normalization_rules,
        )
        if normalized_choice is not None:
            fields["final_choice"] = normalized_choice
    reasoning_trace, reasoning_source = extract_reasoning_trace(
        text,
        artifact_metadata=artifact_metadata,
        spec=spec,
    )
    if reasoning_trace is not None:
        fields.setdefault("thinking", reasoning_trace)
    missing_required = _compute_mode_required_fields(
        spec=spec,
        parse_mode=parse_mode,
        fields=fields,
    )
    return StructuredParseResult(
        format_type="tag_blocks",
        fields=fields,
        missing_required=missing_required,
        parse_status="parsed" if not missing_required else "partial",
        raw_hits=raw_hits,
        fallback_source=fallback_source,
        reasoning_trace=reasoning_trace,
        reasoning_source=reasoning_source,
        raw_payload=text,
    )


def _parse_markdown_sections(
    raw_text: str,
    *,
    spec: StructuredOutputSpec,
    artifact_metadata: dict[str, Any] | None = None,
    parse_mode: str | None = None,
) -> StructuredParseResult:
    text = str(raw_text).strip()
    fields: dict[str, Any] = {}
    raw_hits: dict[str, Any] = {}
    fallback_source: dict[str, str] = {}
    for field_name in _iter_known_fields(spec):
        aliases = spec.aliases.get(field_name, [field_name])
        value = _extract_markdown_section(text, aliases)
        if value is not None:
            fields[field_name] = value
            raw_hits[field_name] = value
            continue
        fallback_value = _extract_fallback_value(
            text=text,
            field_name=field_name,
            patterns=spec.fallback_patterns.get(field_name, []),
        )
        if fallback_value is not None:
            fields[field_name] = fallback_value
            fallback_source[field_name] = "pattern"
    reasoning_trace, reasoning_source = extract_reasoning_trace(
        text,
        artifact_metadata=artifact_metadata,
        spec=spec,
    )
    if reasoning_trace is not None:
        fields.setdefault("thinking", reasoning_trace)
    missing_required = _compute_mode_required_fields(
        spec=spec,
        parse_mode=parse_mode,
        fields=fields,
    )
    return StructuredParseResult(
        format_type="markdown_sections",
        fields=fields,
        missing_required=missing_required,
        parse_status="parsed" if not missing_required else "partial",
        raw_hits=raw_hits,
        fallback_source=fallback_source,
        reasoning_trace=reasoning_trace,
        reasoning_source=reasoning_source,
        raw_payload=text,
    )


def _parse_plain_text(
    raw_text: str,
    *,
    spec: StructuredOutputSpec,
    artifact_metadata: dict[str, Any] | None = None,
) -> StructuredParseResult:
    text = str(raw_text).strip()
    reasoning_trace, reasoning_source = extract_reasoning_trace(
        text,
        artifact_metadata=artifact_metadata,
        spec=spec,
    )
    fields = {"text": text}
    if reasoning_trace is not None:
        fields["thinking"] = reasoning_trace
    missing_required = [
        field_name
        for field_name in spec.required_fields
        if fields.get(field_name) in (None, "", [])
    ]
    return StructuredParseResult(
        format_type="plain_text",
        fields=fields,
        missing_required=missing_required,
        parse_status="parsed" if text else "invalid",
        reasoning_trace=reasoning_trace,
        reasoning_source=reasoning_source,
        raw_payload=text,
    )


def _iter_known_fields(spec: StructuredOutputSpec) -> list[str]:
    ordered: list[str] = []
    for field_name in spec.fields:
        if field_name not in ordered:
            ordered.append(field_name)
    for field_name in spec.aliases:
        if field_name not in ordered:
            ordered.append(field_name)
    for field_name in spec.required_fields:
        if field_name not in ordered:
            ordered.append(field_name)
    return ordered


def _compute_mode_required_fields(
    *,
    spec: StructuredOutputSpec,
    parse_mode: str | None,
    fields: dict[str, Any],
) -> list[str]:
    required = list(spec.required_fields)
    normalized_mode = str(parse_mode or "").strip()
    for field_name, config in spec.fields.items():
        required_by_modes = {
            str(item).strip()
            for item in dict(config or {}).get("required_by_modes", []) or []
            if str(item).strip()
        }
        preferred_by_modes = {
            str(item).strip()
            for item in dict(config or {}).get("preferred_by_modes", []) or []
            if str(item).strip()
        }
        if normalized_mode and normalized_mode in required_by_modes and field_name not in required:
            required.append(field_name)
        if (
            normalized_mode
            and normalized_mode in preferred_by_modes
            and field_name == "final_answer"
            and fields.get("final_answer") in (None, "")
            and fields.get("final_choice") in (None, "")
            and field_name not in required
        ):
            required.append(field_name)
    return [field_name for field_name in required if fields.get(field_name) in (None, "", [])]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())
    object_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if object_match:
        candidates.insert(0, object_match.group(1).strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_tag_by_aliases(text: str, aliases: list[str]) -> str | None:
    for alias in aliases:
        pattern = rf"<{re.escape(alias)}>\s*(.*?)\s*</{re.escape(alias)}>"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_fallback_value(
    *,
    text: str,
    field_name: str,
    patterns: list[str],
) -> str | None:
    del field_name
    for pattern in patterns:
        match = re.search(str(pattern), text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:4000]
    return None


def _extract_markdown_section(text: str, aliases: list[str]) -> str | None:
    for alias in aliases:
        header_pattern = rf"(?:^|\n)#+\s*{re.escape(alias)}\s*\n(.*?)(?=\n#+\s|\Z)"
        header_match = re.search(header_pattern, text, re.IGNORECASE | re.DOTALL)
        if header_match:
            value = header_match.group(1).strip()
            if value:
                return value
        label_pattern = rf"(?:^|\n){re.escape(alias)}\s*:\s*(.*?)(?=\n[A-Z][A-Za-z0-9 _-]*\s*:|\Z)"
        label_match = re.search(label_pattern, text, re.IGNORECASE | re.DOTALL)
        if label_match:
            value = label_match.group(1).strip()
            if value:
                return value
    return None


def _infer_format_type_from_legacy_payload(payload: dict[str, Any]) -> str:
    if payload.get("format") in {"json", "json_object"}:
        return "json_object"
    if payload.get("tags") or payload.get("choice_aliases"):
        return "tag_blocks"
    return "plain_text"
