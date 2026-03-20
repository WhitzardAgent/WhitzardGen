from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from aigc.prompts import validate_generation_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_PROFILES_ROOT = REPO_ROOT / "configs" / "run_profiles"


class RunProfileError(ValueError):
    """Raised when a run profile cannot be parsed or validated."""


@dataclass(slots=True)
class RunProfile:
    path: Path
    name: str | None
    model_names: list[str]
    prompt_file: Path
    execution_mode: str | None
    output_dir: Path | None
    generation_defaults: dict[str, Any]
    runtime: dict[str, Any]
    raw: dict[str, Any]

    @property
    def available_gpus(self) -> list[int] | None:
        runtime_available = self.runtime.get("available_gpus")
        if runtime_available in (None, ""):
            return None
        if not isinstance(runtime_available, list) or not all(
            isinstance(item, int) for item in runtime_available
        ):
            raise RunProfileError(
                f"Run profile {self.path} must define runtime.available_gpus as a list of integers."
            )
        return list(runtime_available)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload["path"] = str(self.path)
        payload["models"] = list(self.model_names)
        payload["prompts"] = str(self.prompt_file)
        if self.output_dir is not None:
            payload["out"] = str(self.output_dir)
        if self.generation_defaults:
            payload["generation_defaults"] = dict(self.generation_defaults)
        return payload


def load_run_profile(path: str | Path) -> RunProfile:
    profile_path = Path(path)
    if not profile_path.exists():
        raise RunProfileError(f"Run profile not found: {profile_path}")

    payload = _parse_payload(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunProfileError(f"Run profile must be an object: {profile_path}")

    model_names = _normalize_model_names(payload.get("models"))
    if not model_names:
        raise RunProfileError(f"Run profile {profile_path} must define a non-empty models list.")

    prompts_value = payload.get("prompts")
    if not prompts_value:
        raise RunProfileError(f"Run profile {profile_path} must define prompts.")
    prompt_file = _resolve_profile_path(Path(str(prompts_value)), profile_path)
    if not prompt_file.exists():
        raise RunProfileError(
            f"Run profile {profile_path} references a prompt file that does not exist: {prompt_file}"
        )

    execution_mode = payload.get("execution_mode")
    if execution_mode is not None:
        execution_mode = str(execution_mode)
        if execution_mode not in {"mock", "real"}:
            raise RunProfileError(
                f"Run profile {profile_path} has invalid execution_mode: {execution_mode}"
            )

    output_value = payload.get("out") or payload.get("output_dir") or payload.get("output_root")
    output_dir = (
        _resolve_profile_path(Path(str(output_value)), profile_path)
        if output_value not in (None, "")
        else None
    )

    generation_defaults = payload.get("generation_defaults") or {}
    if not isinstance(generation_defaults, dict):
        raise RunProfileError(
            f"Run profile {profile_path} has invalid generation_defaults section."
        )
    try:
        generation_defaults = validate_generation_parameters(
            generation_defaults,
            owner_label=f"profile={profile_path}",
            prompt_source=profile_path,
            warn=None,
        )
    except ValueError as exc:
        raise RunProfileError(str(exc)) from exc

    runtime = payload.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise RunProfileError(f"Run profile {profile_path} has invalid runtime section.")

    return RunProfile(
        path=profile_path,
        name=str(payload["name"]) if payload.get("name") not in (None, "") else None,
        model_names=model_names,
        prompt_file=prompt_file,
        execution_mode=execution_mode,
        output_dir=output_dir,
        generation_defaults=generation_defaults,
        runtime=runtime,
        raw=payload,
    )


def resolve_profile_run_request(
    *,
    profile: RunProfile | None,
    models_arg: str | None,
    prompts_arg: str | None,
    execution_mode_arg: str | None,
    mock_flag: bool,
    out_arg: str | None,
    run_name_arg: str | None,
) -> dict[str, Any]:
    model_names = _normalize_model_names(models_arg) if models_arg else []
    if not model_names and profile is not None:
        model_names = list(profile.model_names)
    if not model_names:
        raise RunProfileError("aigc run requires at least one model or a profile that defines models.")

    prompt_file: Path | None = None
    if prompts_arg:
        prompt_file = Path(prompts_arg)
    elif profile is not None:
        prompt_file = profile.prompt_file
    if prompt_file is None:
        raise RunProfileError("aigc run requires --prompts or a profile that defines prompts.")
    if not prompt_file.exists():
        raise RunProfileError(f"Prompt file does not exist: {prompt_file}")

    resolved_execution_mode = "mock" if mock_flag else execution_mode_arg
    if resolved_execution_mode is None and profile is not None:
        resolved_execution_mode = profile.execution_mode
    if resolved_execution_mode is None:
        resolved_execution_mode = "real"

    out_dir: Path | None = Path(out_arg) if out_arg else None
    if out_dir is None and profile is not None:
        out_dir = profile.output_dir

    run_name = run_name_arg or (profile.name if profile is not None else None)

    return {
        "model_names": model_names,
        "prompt_file": prompt_file,
        "execution_mode": resolved_execution_mode,
        "mock_mode": mock_flag,
        "out_dir": out_dir,
        "run_name": run_name,
        "profile_name": profile.name if profile is not None else None,
        "profile_path": str(profile.path) if profile is not None else None,
        "generation_defaults": dict(profile.generation_defaults) if profile is not None else {},
        "runtime": dict(profile.runtime) if profile is not None else {},
    }


@contextmanager
def apply_profile_runtime_environment(profile: RunProfile | None) -> Iterator[None]:
    if profile is None or profile.available_gpus is None:
        yield
        return

    import os

    previous = os.environ.get("AIGC_AVAILABLE_GPUS")
    if previous not in (None, ""):
        yield
        return

    os.environ["AIGC_AVAILABLE_GPUS"] = ",".join(str(item) for item in profile.available_gpus)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AIGC_AVAILABLE_GPUS", None)
        else:
            os.environ["AIGC_AVAILABLE_GPUS"] = previous


def _normalize_model_names(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if item in (None, ""):
                continue
            names.append(str(item).strip())
        return [name for name in names if name]
    raise RunProfileError("Run profile models must be a list of model names or a comma-separated string.")


def _resolve_profile_path(path_value: Path, profile_path: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    relative_to_profile = (profile_path.parent / path_value).resolve()
    if relative_to_profile.exists():
        return relative_to_profile
    return (REPO_ROOT / path_value).resolve()


def _parse_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RunProfileError(
                "Run profile is not valid JSON and PyYAML is not installed for YAML parsing."
            ) from exc
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise RunProfileError("Run profile payload must be an object.")
    return payload
