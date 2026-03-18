from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalOverrideError(ValueError):
    """Raised when local override payloads cannot be parsed."""


LOCAL_MODELS_ENV_VAR = "AIGC_LOCAL_MODELS_FILE"
LOCAL_OVERRIDE_FIELDS = (
    "local_path",
    "repo_path",
    "weights_path",
    "script_root",
    "hf_cache_dir",
)
LOCAL_RUNTIME_OVERRIDE_FIELDS = (
    "max_gpus",
    "conda_env_name",
)


def load_local_model_overrides(path: str | Path | None) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    if path is None:
        return None, {}

    override_path = Path(path)
    if not override_path.exists():
        return override_path, {}

    payload = _parse_override_payload(override_path.read_text(encoding="utf-8"))
    if "models" in payload:
        payload = payload["models"]
    if not isinstance(payload, dict):
        raise LocalOverrideError(
            "Local model override file must be a mapping of model names to overrides."
        )

    overrides: dict[str, dict[str, Any]] = {}
    for model_name, config in payload.items():
        if config is None:
            continue
        if not isinstance(config, dict):
            raise LocalOverrideError(
                f"Local model override for {model_name} must be an object of path fields."
            )
        normalized = {
            key: value
            for key, value in config.items()
            if value not in (None, "")
        }
        overrides[str(model_name)] = normalized
    return override_path, overrides


def _parse_override_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise LocalOverrideError(
                "Local model override file is not valid JSON and PyYAML is not installed for YAML parsing."
            ) from exc
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise LocalOverrideError("Local model override payload must be an object.")
    return payload


def summarize_local_path_overrides(overrides: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for field in LOCAL_OVERRIDE_FIELDS:
        value = overrides.get(field)
        if value:
            lines.append(f"{field}: {value}")
    for field, value in sorted(overrides.items()):
        if field in LOCAL_OVERRIDE_FIELDS or value in (None, ""):
            continue
        lines.append(f"{field}: {value}")
    return lines
