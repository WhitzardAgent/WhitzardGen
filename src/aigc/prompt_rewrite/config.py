from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_REWRITE_CONFIG_PATH = REPO_ROOT / "configs" / "prompt_rewrite"
DEFAULT_TEMPLATE_NAME = "model_rewrite_v1"
DEFAULT_STYLE_FAMILY_NAME = "detailed_sentence"


class PromptRewriteConfigError(ValueError):
    """Raised when prompt-rewrite config is invalid."""


@dataclass(slots=True)
class PromptRewriteTemplateConfig:
    name: str
    version: str
    instruction_template: str
    default_style_family: str | None = None


@dataclass(slots=True)
class PromptRewriteStyleFamilyConfig:
    name: str
    version: str
    description: str
    style_instruction: str
    output_contract: dict[str, Any]
    few_shot_examples: list[dict[str, Any]]
    max_examples_per_request: int
    supported_modalities: list[str] = field(default_factory=list)


def load_prompt_rewrite_catalog(
    path: str | Path = DEFAULT_PROMPT_REWRITE_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_dir():
        raise PromptRewriteConfigError(
            f"Prompt rewrite config directory does not exist: {config_path}"
        )

    templates = _load_templates(config_path / "templates")
    style_families = _load_style_families(config_path / "style_families")
    return {
        "root": config_path,
        "templates": templates,
        "style_families": style_families,
    }


def resolve_prompt_rewrite_config(
    *,
    catalog: dict[str, Any],
    template_name: str | None,
    style_family_name: str | None,
) -> tuple[PromptRewriteTemplateConfig, PromptRewriteStyleFamilyConfig]:
    templates: dict[str, PromptRewriteTemplateConfig] = dict(catalog.get("templates", {}))
    style_families: dict[str, PromptRewriteStyleFamilyConfig] = dict(catalog.get("style_families", {}))

    resolved_template_name = str(template_name or DEFAULT_TEMPLATE_NAME)
    try:
        template = templates[resolved_template_name]
    except KeyError as exc:
        raise PromptRewriteConfigError(
            f"Unknown prompt rewrite template: {resolved_template_name}"
        ) from exc

    resolved_style_family = str(
        style_family_name or template.default_style_family or DEFAULT_STYLE_FAMILY_NAME
    )
    try:
        style_family = style_families[resolved_style_family]
    except KeyError as exc:
        raise PromptRewriteConfigError(
            f"Unknown prompt rewrite style family: {resolved_style_family}"
        ) from exc

    return template, style_family


def select_few_shot_examples(
    *,
    style_family: PromptRewriteStyleFamilyConfig,
    target_model_name: str,
) -> list[dict[str, Any]]:
    exact_matches: list[dict[str, Any]] = []
    generic_matches: list[dict[str, Any]] = []
    normalized_target = str(target_model_name)

    for example in style_family.few_shot_examples:
        if not isinstance(example, dict):
            continue
        applicability = example.get("applicability")
        target_models: list[str] = []
        if isinstance(applicability, dict):
            raw_target_models = applicability.get("target_models")
            if isinstance(raw_target_models, list):
                target_models = [str(item) for item in raw_target_models if str(item).strip()]
            elif raw_target_models not in (None, ""):
                target_models = [str(raw_target_models)]
        if target_models:
            if normalized_target in target_models:
                exact_matches.append(dict(example))
        else:
            generic_matches.append(dict(example))

    selected: list[dict[str, Any]] = []
    for bucket in (exact_matches, generic_matches):
        for example in bucket:
            selected.append(example)
            if len(selected) >= style_family.max_examples_per_request:
                return selected
    return selected


def render_rewrite_instruction(
    *,
    template: PromptRewriteTemplateConfig,
    values: dict[str, Any],
) -> str:
    rendered = str(template.instruction_template)
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return re.sub(r"\{\{[a-zA-Z0-9_]+\}\}", "", rendered).strip() + "\n"


def render_output_contract(style_family: PromptRewriteStyleFamilyConfig) -> str:
    contract = dict(style_family.output_contract)
    if not contract:
        return "Return JSON only with key: prompt."
    return json.dumps(contract, ensure_ascii=False, sort_keys=True)


def render_few_shot_block(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "No few-shot examples selected."
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        example_id = str(example.get("id", f"example_{index}"))
        input_payload = example.get("input", {})
        output_payload = example.get("output", {})
        blocks.append(
            "\n".join(
                [
                    f"Example {index} ({example_id})",
                    f"Input: {json.dumps(input_payload, ensure_ascii=False, sort_keys=True)}",
                    f"Output JSON: {json.dumps(output_payload, ensure_ascii=False, sort_keys=True)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _load_templates(path: Path) -> dict[str, PromptRewriteTemplateConfig]:
    if not path.is_dir():
        raise PromptRewriteConfigError(f"Prompt rewrite template directory does not exist: {path}")
    templates: dict[str, PromptRewriteTemplateConfig] = {}
    for file_path in _iter_config_files(path):
        payload = _load_yaml_or_json(file_path)
        name = str(payload.get("name") or file_path.stem)
        instruction_template = str(payload.get("instruction_template", "")).strip()
        if not instruction_template:
            raise PromptRewriteConfigError(
                f"Prompt rewrite template {name} is missing instruction_template."
            )
        templates[name] = PromptRewriteTemplateConfig(
            name=name,
            version=str(payload.get("version", "v1")),
            instruction_template=instruction_template,
            default_style_family=(
                str(payload.get("default_style_family"))
                if payload.get("default_style_family") not in (None, "")
                else None
            ),
        )
    if not templates:
        raise PromptRewriteConfigError("No prompt rewrite templates were loaded.")
    return templates


def _load_style_families(path: Path) -> dict[str, PromptRewriteStyleFamilyConfig]:
    if not path.is_dir():
        raise PromptRewriteConfigError(
            f"Prompt rewrite style-family directory does not exist: {path}"
        )
    style_families: dict[str, PromptRewriteStyleFamilyConfig] = {}
    for file_path in _iter_config_files(path):
        payload = _load_yaml_or_json(file_path)
        name = str(payload.get("name") or file_path.stem)
        style = PromptRewriteStyleFamilyConfig(
            name=name,
            version=str(payload.get("version", "v1")),
            description=str(payload.get("description", "")),
            style_instruction=str(payload.get("style_instruction", "")).strip(),
            output_contract=dict(payload.get("output_contract", {})),
            few_shot_examples=list(payload.get("few_shot_examples", [])),
            max_examples_per_request=max(int(payload.get("max_examples_per_request", 0)), 0),
            supported_modalities=[
                str(item) for item in payload.get("supported_modalities", []) or []
            ],
        )
        if not style.style_instruction:
            raise PromptRewriteConfigError(
                f"Prompt rewrite style family {name} is missing style_instruction."
            )
        style_families[name] = style
    if not style_families:
        raise PromptRewriteConfigError("No prompt rewrite style families were loaded.")
    return style_families


def _iter_config_files(path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml", ".json"}
        ],
        key=lambda candidate: candidate.name.lower(),
    )


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml(raw, source_path=path)
    if suffix == ".json":
        return _parse_json(raw, source_path=path)
    try:
        return _parse_yaml(raw, source_path=path)
    except PromptRewriteConfigError:
        return _parse_json(raw, source_path=path)


def _parse_yaml(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise PromptRewriteConfigError(
            f"PyYAML is required to parse prompt-rewrite config: {source_path}"
        ) from exc
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise PromptRewriteConfigError(
            f"Prompt-rewrite config must be an object: {source_path}"
        )
    return payload


def _parse_json(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptRewriteConfigError(
            f"Prompt-rewrite config is not valid JSON: {source_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PromptRewriteConfigError(
            f"Prompt-rewrite config must be an object: {source_path}"
        )
    return payload
