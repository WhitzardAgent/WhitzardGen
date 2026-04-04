from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whitzard.benchmarking.discovery import discover_example_evaluator_specs
from whitzard.benchmarking.prompt_templates import resolve_prompt_template_config
from whitzard.evaluators.models import EvaluatorSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVALUATORS_CONFIG_PATH = REPO_ROOT / "configs" / "evaluators"


class EvaluatorConfigError(ValueError):
    """Raised when evaluator config cannot be loaded or resolved."""


def load_evaluator_catalog(path: str | Path = DEFAULT_EVALUATORS_CONFIG_PATH) -> dict[str, EvaluatorSpec]:
    config_root = Path(path)
    profiles_path = config_root / "profiles.yaml"
    payload = _load_yaml_or_json(profiles_path)
    profiles_payload = payload.get("evaluators", payload)
    if not isinstance(profiles_payload, dict) or not profiles_payload:
        raise EvaluatorConfigError(
            f"Evaluator profiles must define a non-empty mapping: {profiles_path}"
        )
    evaluators: dict[str, EvaluatorSpec] = {}
    for evaluator_id, config in profiles_payload.items():
        if not isinstance(config, dict):
            raise EvaluatorConfigError(f"Evaluator {evaluator_id} must be an object.")
        prompt_template = resolve_prompt_template_config(
            dict(config.get("prompt_template", {}) or {}),
            base_dir=profiles_path.parent,
        )
        evaluators[str(evaluator_id)] = EvaluatorSpec(
            evaluator_id=str(evaluator_id),
            evaluator_type=str(config.get("type", "judge")),
            description=str(config.get("description", "")),
            accepted_input_types=[str(item) for item in config.get("accepted_input_types", []) or []],
            rule_type=_normalize_optional_text(config.get("rule_type")),
            rule_config=dict(config.get("rule_config", {}) or {}),
            judge_model=_normalize_optional_text(config.get("judge_model")),
            annotation_profile=_normalize_optional_text(config.get("annotation_profile")),
            annotation_template=_normalize_optional_text(config.get("annotation_template")),
            prompt_template=prompt_template,
            generation_defaults=dict(config.get("generation_defaults", {}) or {}),
        )
    for evaluator_id, config in discover_example_evaluator_specs().items():
        manifest_path = Path(str(config.get("manifest_path") or ""))
        prompt_template = resolve_prompt_template_config(
            dict(config.get("prompt_template", {}) or {}),
            base_dir=manifest_path.parent if manifest_path else None,
        )
        evaluators[evaluator_id] = EvaluatorSpec(
            evaluator_id=str(evaluator_id),
            evaluator_type=str(config.get("type", "judge")),
            description=str(config.get("description", "")),
            accepted_input_types=[str(item) for item in config.get("accepted_input_types", []) or []],
            rule_type=_normalize_optional_text(config.get("rule_type")),
            rule_config=dict(config.get("rule_config", {}) or {}),
            judge_model=_normalize_optional_text(config.get("judge_model")),
            annotation_profile=_normalize_optional_text(config.get("annotation_profile")),
            annotation_template=_normalize_optional_text(config.get("annotation_template")),
            prompt_template=prompt_template,
            generation_defaults=dict(config.get("generation_defaults", {}) or {}),
        )
    return evaluators


def resolve_evaluators(
    evaluator_ids: list[str] | None,
    *,
    path: str | Path | None = DEFAULT_EVALUATORS_CONFIG_PATH,
) -> list[EvaluatorSpec]:
    catalog = load_evaluator_catalog(path or DEFAULT_EVALUATORS_CONFIG_PATH)
    if not evaluator_ids:
        return []
    resolved: list[EvaluatorSpec] = []
    for evaluator_id in evaluator_ids:
        try:
            resolved.append(catalog[evaluator_id])
        except KeyError as exc:
            raise EvaluatorConfigError(f"Unknown evaluator: {evaluator_id}") from exc
    return resolved


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EvaluatorConfigError(f"Evaluator config file does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _parse_yaml(raw, source_path=path)
    return _parse_json(raw, source_path=path)


def _parse_yaml(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise EvaluatorConfigError(
            f"PyYAML is required to parse evaluator config: {source_path}"
        ) from exc
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise EvaluatorConfigError(f"Evaluator config must be an object: {source_path}")
    return payload


def _parse_json(raw: str, *, source_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluatorConfigError(
            f"Evaluator config is not valid JSON: {source_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvaluatorConfigError(f"Evaluator config must be an object: {source_path}")
    return payload


def _normalize_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None
