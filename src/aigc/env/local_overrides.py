from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalEnvOverrideError(ValueError):
    """Raised when local env override payloads cannot be parsed."""


LOCAL_ENVS_ENV_VAR = "AIGC_LOCAL_ENVS_FILE"


def load_local_env_overrides(path: str | Path | None) -> tuple[Path | None, dict[str, Any]]:
    if path is None:
        return None, {}

    override_path = Path(path)
    if not override_path.exists():
        return override_path, {}

    payload = _parse_override_payload(override_path.read_text(encoding="utf-8"))
    return override_path, payload


def resolve_local_env_override(
    payload: dict[str, Any],
    *,
    model_name: str,
    env_spec: str,
) -> dict[str, Any]:
    env_overrides = payload.get("envs", payload)
    model_overrides = payload.get("models", {})

    merged: dict[str, Any] = {}
    if isinstance(env_overrides, dict):
        raw_env_override = env_overrides.get(env_spec)
        if isinstance(raw_env_override, dict):
            merged.update(raw_env_override)
    if isinstance(model_overrides, dict):
        raw_model_override = model_overrides.get(model_name)
        if isinstance(raw_model_override, dict):
            merged.update(raw_model_override)
    return merged


def _parse_override_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise LocalEnvOverrideError(
                "Local env override file is not valid JSON and PyYAML is not installed for YAML parsing."
            ) from exc
        payload = yaml.safe_load(raw)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise LocalEnvOverrideError("Local env override payload must be an object.")
    return payload
