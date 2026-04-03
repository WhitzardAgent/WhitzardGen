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
LOCAL_NESTED_OVERRIDE_FIELDS = (
    "generation_defaults",
    "provider",
)


def load_local_model_overrides(path: str | Path | None) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    if path is None:
        return None, {}

    override_path = Path(path)
    if not override_path.exists():
        return override_path, {}

    if override_path.is_dir():
        return override_path, _load_override_directory(override_path)

    payload = _parse_override_payload(
        override_path.read_text(encoding="utf-8"),
        source_path=override_path,
    )
    return override_path, _normalize_override_mapping(
        _extract_override_mapping(payload, source_label=str(override_path))
    )


def _load_override_directory(path: Path) -> dict[str, dict[str, Any]]:
    files = _iter_override_files(path)
    if not files:
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for file_path in files:
        payload = _parse_override_payload(
            file_path.read_text(encoding="utf-8"),
            source_path=file_path,
        )
        mapping = _normalize_override_mapping(
            _extract_override_mapping(payload, source_label=str(file_path))
        )
        for model_name, config in mapping.items():
            if model_name in overrides:
                raise LocalOverrideError(
                    f"Duplicate local model override for {model_name} in {file_path}."
                )
            overrides[model_name] = config
    return overrides


def _iter_override_files(path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml", ".json"}
        ],
        key=lambda candidate: candidate.name.lower(),
    )


def _extract_override_mapping(payload: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    if "models" in payload:
        payload = payload["models"]
    if not isinstance(payload, dict):
        raise LocalOverrideError(
            f"Local model override payload in {source_label} must be a mapping of model names to overrides."
        )
    return payload


def _normalize_override_mapping(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
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
    return overrides


def _parse_override_payload(raw: str, *, source_path: Path | None = None) -> dict[str, Any]:
    suffix = source_path.suffix.lower() if source_path is not None else ""
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml_override_payload(raw)
    if suffix == ".json":
        return _parse_json_override_payload(raw)
    try:
        return _parse_yaml_override_payload(raw)
    except LocalOverrideError:
        return _parse_json_override_payload(raw)


def _parse_yaml_override_payload(raw: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise LocalOverrideError(
            "PyYAML is required to parse YAML local model override files."
        ) from exc
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise LocalOverrideError("Local model override payload must be an object.")
    return payload


def _parse_json_override_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalOverrideError("Local model override file is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise LocalOverrideError("Local model override payload must be an object.")
    return payload


def summarize_local_path_overrides(overrides: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for field in LOCAL_OVERRIDE_FIELDS:
        value = overrides.get(field)
        if value:
            lines.append(f"{field}: {value}")
    provider = overrides.get("provider")
    if isinstance(provider, dict):
        for key, value in sorted(provider.items()):
            if value in (None, ""):
                continue
            if key == "default_headers" and isinstance(value, dict):
                lines.append(
                    f"provider.{key}: "
                    + str({str(header): "<redacted>" for header in value})
                )
                continue
            lines.append(f"provider.{key}: {value}")
    for field, value in sorted(overrides.items()):
        if field in LOCAL_OVERRIDE_FIELDS or field == "provider" or value in (None, ""):
            continue
        lines.append(f"{field}: {value}")
    return lines
