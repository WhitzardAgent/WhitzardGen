from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from whitzard.prompt_generation.models import PromptStyleFamilyConfig, PromptTemplateConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_GENERATION_CONFIG_PATH = REPO_ROOT / "configs" / "prompt_generation"
DEFAULT_TEMPLATE_NAME = "photorealistic_base"
DEFAULT_STYLE_FAMILY_NAME = "detailed_sentence"


class PromptGenerationConfigError(ValueError):
    """Raised when prompt-generation config is invalid."""


def load_prompt_generation_catalog(
    path: str | Path = DEFAULT_PROMPT_GENERATION_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.is_dir():
        root = config_path
        profiles = _load_profiles_file(root / "profiles.yaml")
        templates = _load_templates(root / "templates")
        style_families = _load_style_families(root / "style_families")
        target_style_mappings = _load_target_style_mappings(root / "target_style_mappings.yaml")
        return {
            "root": root,
            "profiles": profiles,
            "templates": templates,
            "style_families": style_families,
            "target_style_mappings": target_style_mappings,
        }

    if config_path.suffix.lower() in {".yaml", ".yml", ".json"}:
        profiles = _load_profiles_file(config_path)
        root = DEFAULT_PROMPT_GENERATION_CONFIG_PATH
        templates = _load_templates(root / "templates")
        style_families = _load_style_families(root / "style_families")
        target_style_mappings = _load_target_style_mappings(root / "target_style_mappings.yaml")
        return {
            "root": config_path.parent,
            "profiles": profiles,
            "templates": templates,
            "style_families": style_families,
            "target_style_mappings": target_style_mappings,
        }

    raise PromptGenerationConfigError(
        f"Prompt-generation config path is invalid: {config_path}"
    )


def render_instruction_template(
    template: str,
    *,
    values: dict[str, Any],
) -> str:
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return re.sub(r"\{\{[a-zA-Z0-9_]+\}\}", "", rendered).strip() + "\n"


def _load_profiles_file(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_or_json(path)
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict) or not profiles:
        raise PromptGenerationConfigError(
            f"Prompt-generation profiles must define a non-empty mapping: {path}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, config in profiles.items():
        if not isinstance(config, dict):
            raise PromptGenerationConfigError(f"Profile {name} must be an object.")
        normalized[str(name)] = dict(config)
    return normalized


def _load_templates(path: Path) -> dict[str, PromptTemplateConfig]:
    if not path.is_dir():
        raise PromptGenerationConfigError(f"Prompt template directory does not exist: {path}")
    templates: dict[str, PromptTemplateConfig] = {}
    for file_path in _iter_config_files(path):
        payload = _load_yaml_or_json(file_path)
        name = str(payload.get("name") or file_path.stem)
        instruction_template = str(payload.get("instruction_template", "")).strip()
        if not instruction_template:
            raise PromptGenerationConfigError(
                f"Prompt template {name} is missing instruction_template."
            )
        templates[name] = PromptTemplateConfig(
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
        raise PromptGenerationConfigError("No prompt templates were loaded.")
    return templates


def _load_style_families(path: Path) -> dict[str, PromptStyleFamilyConfig]:
    if not path.is_dir():
        raise PromptGenerationConfigError(f"Prompt style-family directory does not exist: {path}")
    style_families: dict[str, PromptStyleFamilyConfig] = {}
    for file_path in _iter_config_files(path):
        payload = _load_yaml_or_json(file_path)
        name = str(payload.get("name") or file_path.stem)
        style_families[name] = PromptStyleFamilyConfig(
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
        if not style_families[name].style_instruction:
            raise PromptGenerationConfigError(f"Style family {name} is missing style_instruction.")
    if not style_families:
        raise PromptGenerationConfigError("No prompt style families were loaded.")
    return style_families


def _load_target_style_mappings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = _load_yaml_or_json(path)
    mappings = payload.get("mappings", payload)
    if not isinstance(mappings, dict):
        raise PromptGenerationConfigError("Target style mappings must be an object.")
    return {str(model_name): str(style_name) for model_name, style_name in mappings.items()}


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
    except PromptGenerationConfigError:
        return _parse_json(raw, source_path=path)


def _parse_yaml(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise PromptGenerationConfigError(
            f"PyYAML is required to parse prompt-generation config: {source_path}"
        ) from exc
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise PromptGenerationConfigError(
            f"Prompt-generation config must be an object: {source_path}"
        )
    return payload


def _parse_json(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptGenerationConfigError(
            f"Prompt-generation config is not valid JSON: {source_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PromptGenerationConfigError(
            f"Prompt-generation config must be an object: {source_path}"
        )
    return payload
