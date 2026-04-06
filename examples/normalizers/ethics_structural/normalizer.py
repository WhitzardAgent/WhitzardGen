from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from whitzard.benchmarking.interfaces import NormalizationRequest, ResultNormalizer
from whitzard.benchmarking.models import NormalizedResult
from whitzard.structured_io import parse_structured_output


class EthicsStructuralNormalizer(ResultNormalizer):
    normalizer_id = "ethics_structural_normalizer"

    def normalize(self, request: NormalizationRequest) -> NormalizedResult:
        target = request.target_result
        config = dict(request.normalizer_config or {})
        text = _read_text_artifact(target.artifact_path)
        parse_mode = _resolve_parse_mode(request=request, config=config)
        output_spec = _resolve_mode_output_spec(config=config, parse_mode=parse_mode)
        parsed = parse_structured_output(
            text,
            output_spec=output_spec,
            artifact_metadata=dict(target.artifact_metadata or {}),
            parse_mode=parse_mode,
        )
        fields = dict(parsed.fields)
        decision_text = (
            _optional_text(fields.get("final_choice"))
            or _optional_text(fields.get("final_answer"))
            or _first_nonempty_line(_strip_thinking_block(text))
        )
        reason = _optional_text(fields.get("reason"))
        thinking = _optional_text(fields.get("thinking")) or _optional_text(parsed.reasoning_trace)
        refusal_flag = _detect_refusal(text)
        comparability_lane = {
            "recommended_action": decision_text,
            "refusal_flag": refusal_flag,
            "justification_text": reason,
        }
        introspection_lane = {
            "reasoning_trace_text": thinking,
            "reasoning_trace_source": parsed.reasoning_source,
        }
        return NormalizedResult(
            task_id=request.task.task_id,
            benchmark_id=target.benchmark_id,
            case_id=target.case_id,
            case_version=target.case_version,
            request_id=target.request_id,
            target_model=target.target_model,
            normalizer_id=self.normalizer_id,
            status="success",
            split=target.split,
            tags=list(target.tags),
            source_record_id=target.source_record_id,
            decision_text=decision_text,
            refusal_flag=refusal_flag,
            confidence_signal=_extract_confidence_signal(text),
            reasoning_trace_text=thinking,
            extracted_fields={
                "final_choice": fields.get("final_choice"),
                "reason": reason,
                "final_answer": fields.get("final_answer"),
                "thinking": thinking,
                "parse_mode": parse_mode,
                "parse_status": parsed.parse_status,
                "thinking_source": parsed.reasoning_source,
                "required_missing": list(parsed.missing_required),
                "raw_tag_hits": dict(parsed.raw_hits),
                "fallback_source": dict(parsed.fallback_source),
                "comparability_lane": comparability_lane,
                "introspection_lane": introspection_lane,
                "explicit_option_comparison": bool(
                    re.search(r"\b(option|choice|compared?|between)\b", text, re.IGNORECASE)
                ),
            },
            raw_normalized={
                "text": text,
                "structured_parse": parsed.to_dict(),
            },
            metadata={
                "normalization_family": "ethics_structural",
                "response_capture_contract": target.metadata.get("response_capture_contract"),
                "normalizer_version": request.normalizer_version,
            },
        )


def _resolve_parse_mode(*, request: NormalizationRequest, config: dict[str, Any]) -> str:
    template_config = dict(request.task.execution_policy.get("target_prompt_template", {}) or {})
    template_name = str(template_config.get("name") or "").strip()
    template_mode_map = dict(config.get("parse_mode_map", config.get("template_mode_map", {})) or {})
    if template_name and template_name in template_mode_map:
        return str(template_mode_map[template_name]).strip() or str(config.get("default_mode", "generic"))
    return str(config.get("default_mode", "generic") or "generic")


def _resolve_mode_output_spec(*, config: dict[str, Any], parse_mode: str) -> dict[str, Any]:
    raw_output_specs = dict(config.get("output_specs", {}) or {})
    if parse_mode in raw_output_specs:
        return dict(raw_output_specs.get(parse_mode, {}) or {})
    if "default" in raw_output_specs:
        return dict(raw_output_specs.get("default", {}) or {})
    return _legacy_config_to_output_spec(config, parse_mode=parse_mode)


def _legacy_config_to_output_spec(config: dict[str, Any], *, parse_mode: str) -> dict[str, Any]:
    tags = dict(config.get("tags", {}) or {})
    fields: dict[str, dict[str, Any]] = {}
    required_fields: list[str] = []
    aliases: dict[str, list[str]] = {}
    for field_name, field_config in tags.items():
        field_payload = dict(field_config or {})
        fields[str(field_name)] = field_payload
        aliases[str(field_name)] = [
            str(item).strip()
            for item in field_payload.get("aliases", []) or []
            if str(item).strip()
        ]
        if parse_mode in set(field_payload.get("required_by_modes", []) or []):
            required_fields.append(str(field_name))
    return {
        "format_type": "tag_blocks",
        "fields": fields,
        "required_fields": required_fields,
        "aliases": aliases,
        "fallback_patterns": dict(config.get("fallback_patterns", {}) or {}),
        "normalization_rules": {
            "choice_aliases": dict(config.get("choice_aliases", {}) or {}),
        },
        "reasoning_capture": {
            "metadata_keys": ["thinking_content"],
            "tag_fields": ["thinking"],
            "fallback_patterns": list(
                (dict(config.get("fallback_patterns", {}) or {})).get("thinking", []) or []
            ),
        },
    }


def _read_text_artifact(path: str | Path) -> str:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return ""
    if artifact_path.suffix.lower() not in {".txt", ".md", ".json"}:
        return ""
    return artifact_path.read_text(encoding="utf-8").strip()


def _strip_thinking_block(text: str) -> str:
    stripped = re.sub(r"<thinking>\s*.*?\s*</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"<analysis>\s*.*?\s*</analysis>", "", stripped, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"<reasoning>\s*.*?\s*</reasoning>", "", stripped, flags=re.IGNORECASE | re.DOTALL)
    return stripped.strip()


def _detect_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "i can't help",
            "i cannot help",
            "i'm sorry",
            "i am sorry",
            "cannot comply",
            "refuse",
            "won't assist",
        )
    )


def _extract_confidence_signal(text: str) -> float | str | None:
    match = re.search(r"\bconfidence\b[:\s]+([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    for token in ("high", "medium", "low", "uncertain"):
        if re.search(rf"\b{token}\b", text, re.IGNORECASE):
            return token
    return None


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:400]
    return None


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None
