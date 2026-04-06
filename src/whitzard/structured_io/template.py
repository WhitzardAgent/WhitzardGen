from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

from whitzard.structured_io.models import TemplateSpec


_TEMPLATE_TOKEN_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")
_KNOWN_HELPER_KEYS = {
    "decision_options",
    "formatted_choices",
    "structured_choices_json",
    "rendered_input",
    "input_payload_json",
    "metadata_json",
    "grouping_json",
    "execution_hints_json",
    "evaluation_hints_json",
    "decision_options_json",
    "output_contract_block",
    "selected_metadata_json",
    "source_prompt_metadata_json",
    "source_artifact_metadata_json",
    "source_generation_params_json",
    "target_result_json",
    "normalized_result_json",
}


def resolve_template_spec(
    config: dict[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
) -> TemplateSpec:
    payload = dict(config or {})
    if not payload:
        return TemplateSpec(name="inline_prompt_template")
    template_path = payload.get("path")
    template_text = str(payload.get("template_text") or "")
    resolved_path: str | None = None
    if template_path not in (None, ""):
        path = Path(str(template_path))
        if not path.is_absolute() and base_dir is not None:
            path = (Path(base_dir) / path).resolve()
        resolved_path = str(path)
        template_text = path.read_text(encoding="utf-8")
        name = str(payload.get("name") or path.stem)
    else:
        name = str(payload.get("name") or "inline_prompt_template")
    return TemplateSpec(
        name=name,
        version=str(payload.get("version") or "v1"),
        path=resolved_path,
        template_text=template_text,
        variable_allowlist=[
            str(item).strip()
            for item in payload.get("variable_allowlist", []) or []
            if str(item).strip()
        ],
        helpers=[
            str(item).strip()
            for item in payload.get("helpers", []) or []
            if str(item).strip()
        ],
        missing_variable_policy=str(
            payload.get("missing_variable_policy") or "warn_and_empty"
        ).strip()
        or "warn_and_empty",
    )


class StructuredRenderer:
    def resolve(self, config: dict[str, Any] | None, *, base_dir: str | Path | None = None) -> TemplateSpec:
        return resolve_template_spec(config, base_dir=base_dir)

    def render(
        self,
        *,
        template_spec: TemplateSpec,
        root_context: dict[str, Any],
        warning_prefix: str,
    ) -> tuple[str, list[str]]:
        return render_template_spec(
            template_spec=template_spec,
            root_context=root_context,
            warning_prefix=warning_prefix,
        )


def render_template_spec(
    *,
    template_spec: TemplateSpec,
    root_context: dict[str, Any],
    warning_prefix: str,
) -> tuple[str, list[str]]:
    template_text = str(template_spec.template_text or "")
    if not template_text:
        return "", []
    root_context = _apply_helper_filter(
        root_context=dict(root_context),
        helper_names=list(template_spec.helpers),
    )
    scoped_context, warnings_list = build_allowed_context(
        root_context=root_context,
        variable_allowlist=list(template_spec.variable_allowlist),
        warning_prefix=warning_prefix,
    )

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        value, found = _resolve_selector(scoped_context, token)
        if found:
            return stringify_template_value(value)
        warning = f"{warning_prefix}: template variable is unavailable: {token}"
        warnings_list.append(warning)
        _emit_template_warning(warning, template_spec.missing_variable_policy)
        return ""

    rendered = _TEMPLATE_TOKEN_RE.sub(_replace, template_text)
    return rendered.strip(), dedupe_preserve_order(warnings_list)


def render_template_text(template: str, *, values: dict[str, Any]) -> str:
    spec = TemplateSpec(
        name="inline_template_text",
        template_text=str(template),
        variable_allowlist=sorted(str(key) for key in values.keys()),
    )
    rendered, _warnings = render_template_spec(
        template_spec=spec,
        root_context=dict(values),
        warning_prefix="inline template",
    )
    return rendered.strip() + "\n"


def build_allowed_context(
    *,
    root_context: dict[str, Any],
    variable_allowlist: list[str],
    warning_prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    warnings_list: list[str] = []
    scoped_context: dict[str, Any] = {}
    for selector in variable_allowlist:
        value, found = _resolve_selector(root_context, selector)
        if not found:
            warnings_list.append(
                f"{warning_prefix}: allowlisted template variable is unavailable: {selector}"
            )
            continue
        _assign_selector(scoped_context, selector, value)
    return scoped_context, dedupe_preserve_order(warnings_list)


def format_structured_choices(decision_options: Any) -> str:
    if not isinstance(decision_options, list):
        return ""
    rendered: list[str] = []
    for item in decision_options:
        if not isinstance(item, dict):
            return ""
        option_id = str(item.get("id", "")).strip().upper()
        text = str(item.get("text", "")).strip()
        if option_id not in {"A", "B"} or not text:
            return ""
        rendered.append(f"{option_id}. {text}")
    if len(rendered) != 2:
        return ""
    return "\n".join(rendered)


def default_target_template_context(*, request: Any) -> dict[str, Any]:
    payload = dict(getattr(request, "input_payload", {}) or {})
    metadata = dict(getattr(request, "metadata", {}) or {})
    case_metadata = dict(metadata.get("case_metadata", {}) or {})
    grouping = dict(metadata.get("grouping", {}) or {})
    execution_hints = dict(metadata.get("execution_hints", {}) or {})
    evaluation_hints = dict(metadata.get("evaluation_hints", {}) or {})
    decision_options = payload.get("decision_options")
    if not isinstance(decision_options, list) or not decision_options:
        decision_options = case_metadata.get("decision_options", [])
    prompt = optional_text(payload.get("prompt"))
    instruction = optional_text(payload.get("instruction"))
    context = payload.get("context")
    return {
        "benchmark_id": getattr(request, "benchmark_id", ""),
        "case_id": getattr(request, "case_id", ""),
        "case_version": metadata.get("case_version"),
        "source_builder": metadata.get("source_builder"),
        "split": metadata.get("split", "default"),
        "tags": list(metadata.get("tags", []) or []),
        "grouping": grouping,
        "input_payload": payload,
        "metadata": case_metadata,
        "execution_hints": execution_hints,
        "evaluation_hints": evaluation_hints,
        "prompt": prompt or "",
        "instruction": instruction or "",
        "context": context,
        "language": payload.get("language") or "en",
        "parameters": dict(getattr(request, "generation_params", {}) or {}),
        "decision_options": decision_options or [],
        "formatted_choices": format_structured_choices(decision_options),
        "structured_choices_json": json_dumps(decision_options or []),
        "rendered_input": resolve_rendered_input(payload),
        "input_payload_json": json_dumps(payload),
        "metadata_json": json_dumps(case_metadata),
        "selected_metadata_json": json_dumps(case_metadata),
        "grouping_json": json_dumps(grouping),
        "execution_hints_json": json_dumps(execution_hints),
        "evaluation_hints_json": json_dumps(evaluation_hints),
        "decision_options_json": json_dumps(decision_options or []),
    }


def default_judge_template_context(
    *,
    source_record: dict[str, Any],
    source_run_id: str,
    output_contract: dict[str, Any],
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_metadata = dict(source_record.get("prompt_metadata", {}) or {})
    artifact_metadata = dict(source_record.get("artifact_metadata", {}) or {})
    generation_params = dict(source_record.get("generation_params", {}) or {})
    artifact_text = read_optional_artifact_text(source_record.get("artifact_path"))
    payload = {
        "source_run_id": source_run_id,
        "source_record_id": source_record.get("record_id"),
        "source_prompt_id": source_record.get("prompt_id"),
        "source_task_id": source_record.get("task_id"),
        "source_model_name": source_record.get("model_name"),
        "source_task_type": source_record.get("task_type"),
        "source_artifact_type": source_record.get("artifact_type"),
        "source_artifact_path": source_record.get("artifact_path"),
        "source_artifact_metadata": artifact_metadata,
        "source_prompt": source_record.get("prompt", ""),
        "source_negative_prompt": source_record.get("negative_prompt") or "",
        "source_prompt_metadata": prompt_metadata,
        "source_generation_params": generation_params,
        "output_contract": dict(output_contract or {}),
        "output_contract_block": render_output_contract_block(output_contract),
        "target_output_text": artifact_text,
        "artifact_text": artifact_text,
        "rendered_input": str(source_record.get("prompt", "") or ""),
        "formatted_choices": format_structured_choices(
            prompt_metadata.get("decision_options", [])
        ),
        "decision_options": list(prompt_metadata.get("decision_options", []) or []),
        "structured_choices_json": json_dumps(
            prompt_metadata.get("decision_options", []) or []
        ),
        "decision_options_json": json_dumps(
            prompt_metadata.get("decision_options", []) or []
        ),
        "selected_metadata_json": json_dumps(prompt_metadata),
        "source_prompt_metadata_json": json_dumps(prompt_metadata),
        "source_artifact_metadata_json": json_dumps(artifact_metadata),
        "source_generation_params_json": json_dumps(generation_params),
    }
    if extra_context:
        payload.update(dict(extra_context))
    return payload


def render_output_contract_block(contract: dict[str, Any]) -> str:
    required_keys = [
        str(item).strip()
        for item in (
            dict(contract or {}).get("required_keys")
            or dict(contract or {}).get("required_fields")
            or []
        )
        if str(item).strip()
    ]
    if not required_keys:
        return "Return JSON only."
    return "Return JSON only with keys: " + ", ".join(required_keys) + "."


def resolve_rendered_input(payload: dict[str, Any]) -> str:
    for key in ("prompt", "instruction", "text", "input"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def stringify_template_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def read_optional_artifact_text(path: Any) -> str:
    if path in (None, ""):
        return ""
    target = Path(str(path))
    if not target.exists() or not target.is_file():
        return ""
    if target.suffix.lower() not in {".txt", ".json", ".md"}:
        return ""
    return target.read_text(encoding="utf-8").strip()


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_selector(root: Any, selector: str) -> tuple[Any, bool]:
    current = root
    if selector in (None, ""):
        return None, False
    for part in str(selector).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None, False
    return current, True


def _assign_selector(target: dict[str, Any], selector: str, value: Any) -> None:
    parts = str(selector).split(".")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def _emit_template_warning(message: str, policy: str) -> None:
    if policy == "warn_and_empty":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _apply_helper_filter(
    *,
    root_context: dict[str, Any],
    helper_names: list[str],
) -> dict[str, Any]:
    if not helper_names:
        return root_context
    allowed_helpers = set(helper_names)
    filtered = dict(root_context)
    for key in _KNOWN_HELPER_KEYS:
        if key in filtered and key not in allowed_helpers:
            filtered.pop(key, None)
    return filtered
