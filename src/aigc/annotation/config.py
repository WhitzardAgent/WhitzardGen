from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aigc.annotation.models import AnnotationProfileConfig, AnnotationTemplateConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATION_CONFIG_PATH = REPO_ROOT / "configs" / "annotation"
DEFAULT_ANNOTATION_PROFILE = "default_review"


class AnnotationConfigError(ValueError):
    """Raised when annotation config cannot be loaded or resolved."""


def load_annotation_catalog(
    path: str | Path = DEFAULT_ANNOTATION_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_dir():
        raise AnnotationConfigError(f"Annotation config directory does not exist: {config_path}")
    profiles = _load_profiles_file(config_path / "profiles.yaml")
    templates = _load_templates(config_path / "templates")
    return {
        "root": config_path,
        "profiles": profiles,
        "templates": templates,
    }


def resolve_annotation_profile(
    catalog: dict[str, Any],
    *,
    profile_name: str | None,
    template_name: str | None,
) -> tuple[AnnotationProfileConfig, AnnotationTemplateConfig]:
    profiles = catalog.get("profiles", {})
    templates = catalog.get("templates", {})
    resolved_profile_name = profile_name or DEFAULT_ANNOTATION_PROFILE
    try:
        profile = profiles[resolved_profile_name]
    except KeyError as exc:
        raise AnnotationConfigError(f"Unknown annotation profile: {resolved_profile_name}") from exc
    resolved_template_name = template_name or profile.default_template
    if not resolved_template_name:
        raise AnnotationConfigError(
            f"Annotation profile {profile.name} does not declare a default template."
        )
    try:
        template = templates[resolved_template_name]
    except KeyError as exc:
        raise AnnotationConfigError(f"Unknown annotation template: {resolved_template_name}") from exc
    return profile, template


def render_annotation_template(template: str, *, values: dict[str, Any]) -> str:
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return re.sub(r"\{\{[a-zA-Z0-9_]+\}\}", "", rendered).strip() + "\n"


def render_output_contract(contract: dict[str, Any]) -> str:
    required_keys = [
        str(item).strip()
        for item in contract.get("required_keys", []) or []
        if str(item).strip()
    ]
    if not required_keys:
        return "Return JSON only."
    return "Return JSON only with keys: " + ", ".join(required_keys) + "."


def parse_annotation_response(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    if not text:
        raise AnnotationConfigError("Annotation response is empty.")

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
    raise AnnotationConfigError("Annotation response did not contain a valid JSON object.")


def validate_annotation_payload(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    required_keys = [
        str(item).strip()
        for item in contract.get("required_keys", []) or []
        if str(item).strip()
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise AnnotationConfigError(
            f"Annotation response is missing required keys: {', '.join(missing)}"
        )


def _load_profiles_file(path: Path) -> dict[str, AnnotationProfileConfig]:
    payload = _load_yaml_or_json(path)
    profiles_payload = payload.get("profiles", payload)
    if not isinstance(profiles_payload, dict) or not profiles_payload:
        raise AnnotationConfigError(
            f"Annotation profiles must define a non-empty mapping: {path}"
        )
    profiles: dict[str, AnnotationProfileConfig] = {}
    for name, config in profiles_payload.items():
        if not isinstance(config, dict):
            raise AnnotationConfigError(f"Annotation profile {name} must be an object.")
        profiles[str(name)] = AnnotationProfileConfig(
            name=str(name),
            version=str(config.get("version", "v1")),
            default_model=_normalize_optional_text(config.get("default_model")),
            default_template=_normalize_optional_text(config.get("default_template")),
            generation_defaults=dict(config.get("generation_defaults", {})),
            output_contract=dict(config.get("output_contract", {})),
            accepted_source_artifact_types=[
                str(item) for item in config.get("accepted_source_artifact_types", []) or []
            ],
        )
    return profiles


def _load_templates(path: Path) -> dict[str, AnnotationTemplateConfig]:
    if not path.is_dir():
        raise AnnotationConfigError(f"Annotation template directory does not exist: {path}")
    templates: dict[str, AnnotationTemplateConfig] = {}
    for file_path in _iter_config_files(path):
        payload = _load_yaml_or_json(file_path)
        name = str(payload.get("name") or file_path.stem)
        instruction_template = str(payload.get("instruction_template", "")).strip()
        if not instruction_template:
            raise AnnotationConfigError(
                f"Annotation template {name} is missing instruction_template."
            )
        templates[name] = AnnotationTemplateConfig(
            name=name,
            version=str(payload.get("version", "v1")),
            instruction_template=instruction_template,
        )
    if not templates:
        raise AnnotationConfigError("No annotation templates were loaded.")
    return templates


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
    except AnnotationConfigError:
        return _parse_json(raw, source_path=path)


def _parse_yaml(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise AnnotationConfigError(
            f"PyYAML is required to parse annotation config: {source_path}"
        ) from exc
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise AnnotationConfigError(f"Annotation config must be an object: {source_path}")
    return payload


def _parse_json(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnnotationConfigError(
            f"Annotation config is not valid JSON: {source_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AnnotationConfigError(f"Annotation config must be an object: {source_path}")
    return payload


def _normalize_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None
