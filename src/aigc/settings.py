from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_RUNTIME_PATH = REPO_ROOT / "configs" / "local_runtime.yaml"
LOCAL_RUNTIME_ENV_VAR = "AIGC_LOCAL_RUNTIME_FILE"


class RuntimeSettingsError(ValueError):
    """Raised when runtime settings cannot be loaded."""


@dataclass(slots=True)
class RuntimeSettings:
    runs_root: Path
    prompt_runs_root: Path
    default_seed: int | None = None
    config_path: Path | None = None


def load_runtime_settings(path: str | Path | None = None) -> RuntimeSettings:
    config_path = _resolve_runtime_settings_path(path)
    default_runs_root = REPO_ROOT / "runs"
    if config_path is None or not config_path.exists():
        return RuntimeSettings(
            runs_root=default_runs_root,
            prompt_runs_root=default_runs_root / "prompt_runs",
            default_seed=None,
            config_path=config_path,
        )

    payload = _parse_settings_payload(config_path.read_text(encoding="utf-8"))
    paths_payload = payload.get("paths", payload)
    if not isinstance(paths_payload, dict):
        raise RuntimeSettingsError("Runtime settings 'paths' payload must be an object.")

    generation_payload = payload.get("generation", {})
    if generation_payload in (None, ""):
        generation_payload = {}
    if not isinstance(generation_payload, dict):
        raise RuntimeSettingsError("Runtime settings 'generation' payload must be an object.")
    default_seed = generation_payload.get("default_seed")
    if default_seed in (None, ""):
        parsed_seed = None
    else:
        parsed_seed = int(default_seed)

    configured_root = paths_payload.get("runs_root") or paths_payload.get("output_root")
    configured_prompt_root = paths_payload.get("prompt_runs_root")
    if configured_root in (None, ""):
        return RuntimeSettings(
            runs_root=default_runs_root,
            prompt_runs_root=Path(str(configured_prompt_root)).expanduser()
            if configured_prompt_root not in (None, "")
            else default_runs_root / "prompt_runs",
            default_seed=parsed_seed,
            config_path=config_path,
        )

    resolved_runs_root = Path(str(configured_root)).expanduser()
    return RuntimeSettings(
        runs_root=resolved_runs_root,
        prompt_runs_root=Path(str(configured_prompt_root)).expanduser()
        if configured_prompt_root not in (None, "")
        else resolved_runs_root / "prompt_runs",
        default_seed=parsed_seed,
        config_path=config_path,
    )


def get_runs_root(path: str | Path | None = None) -> Path:
    return load_runtime_settings(path).runs_root


def get_default_seed(path: str | Path | None = None) -> int | None:
    return load_runtime_settings(path).default_seed


def get_prompt_runs_root(path: str | Path | None = None) -> Path:
    return load_runtime_settings(path).prompt_runs_root


def _resolve_runtime_settings_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    env_override = os.environ.get(LOCAL_RUNTIME_ENV_VAR)
    if env_override:
        return Path(env_override)
    return DEFAULT_LOCAL_RUNTIME_PATH


def _parse_settings_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeSettingsError(
                "Runtime settings file is not valid JSON and PyYAML is not installed for YAML parsing."
            ) from exc
        payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeSettingsError("Runtime settings payload must be an object.")
    return payload
