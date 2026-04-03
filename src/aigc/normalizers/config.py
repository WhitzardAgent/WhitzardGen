from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc.benchmarking.discovery import discover_example_normalizer_specs
from aigc.benchmarking.models import NormalizerSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NORMALIZERS_CONFIG_PATH = REPO_ROOT / "configs" / "normalizers"


class NormalizerConfigError(ValueError):
    """Raised when normalizer config cannot be loaded or resolved."""


def load_normalizer_catalog(path: str | Path = DEFAULT_NORMALIZERS_CONFIG_PATH) -> dict[str, NormalizerSpec]:
    config_root = Path(path)
    profiles_path = config_root / "profiles.yaml"
    payload = _load_yaml_or_json(profiles_path)
    profiles_payload = payload.get("normalizers", payload)
    if not isinstance(profiles_payload, dict) or not profiles_payload:
        raise NormalizerConfigError(
            f"Normalizer profiles must define a non-empty mapping: {profiles_path}"
        )
    normalizers: dict[str, NormalizerSpec] = {}
    for normalizer_id, config in profiles_payload.items():
        if not isinstance(config, dict):
            raise NormalizerConfigError(f"Normalizer {normalizer_id} must be an object.")
        normalizers[str(normalizer_id)] = NormalizerSpec(
            normalizer_id=str(normalizer_id),
            normalizer_type=str(config.get("type", "text_extraction")),
            description=str(config.get("description", "")),
            accepted_input_types=[str(item) for item in config.get("accepted_input_types", []) or []],
            config=dict(config.get("config", {}) or {}),
            version=str(config.get("version", "v1")),
        )
    for normalizer_id, spec in discover_example_normalizer_specs().items():
        normalizers[normalizer_id] = spec
    return normalizers


def resolve_normalizers(
    normalizer_ids: list[str] | None,
    *,
    path: str | Path | None = DEFAULT_NORMALIZERS_CONFIG_PATH,
) -> list[NormalizerSpec]:
    catalog = load_normalizer_catalog(path or DEFAULT_NORMALIZERS_CONFIG_PATH)
    if not normalizer_ids:
        return []
    resolved: list[NormalizerSpec] = []
    for normalizer_id in normalizer_ids:
        try:
            resolved.append(catalog[normalizer_id])
        except KeyError as exc:
            raise NormalizerConfigError(f"Unknown normalizer: {normalizer_id}") from exc
    return resolved


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NormalizerConfigError(f"Normalizer config file does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise NormalizerConfigError(f"PyYAML is required to parse normalizer config: {path}") from exc
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise NormalizerConfigError(f"Normalizer config must be an object: {path}")
    return payload
