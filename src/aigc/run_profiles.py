from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    global_negative_prompt: str | None
    raw: dict[str, Any]
    conditionings: list["ConditioningSpec"] = field(default_factory=list)
    prompt_rewrites: list["PromptRewriteSpec"] = field(default_factory=list)

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
        if self.global_negative_prompt:
            payload["global_negative_prompt"] = self.global_negative_prompt
        if self.conditionings:
            payload["conditionings"] = [conditioning.to_dict() for conditioning in self.conditionings]
        if self.prompt_rewrites:
            payload["prompt_rewrites"] = [rewrite.to_dict() for rewrite in self.prompt_rewrites]
        return payload


@dataclass(slots=True)
class ConditioningSpec:
    target_models: list[str]
    conditioning_type: str
    source_mode: str
    source_model: str | None = None
    generation_defaults: dict[str, Any] = field(default_factory=dict)
    artifact_retention: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload["target_models"] = list(self.target_models)
        payload["conditioning_type"] = self.conditioning_type
        payload["source_mode"] = self.source_mode
        if self.source_model:
            payload["source_model"] = self.source_model
        if self.generation_defaults:
            payload["generation_defaults"] = dict(self.generation_defaults)
        if self.artifact_retention:
            payload["artifact_retention"] = self.artifact_retention
        return payload


@dataclass(slots=True)
class PromptRewriteSpec:
    target_models: list[str]
    source_model: str
    template: str
    style_family: str
    generation_defaults: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    failure_policy: str = "fallback_original"
    stage_order: str = "before_conditioning"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def available_gpus(self) -> list[int] | None:
        runtime_available = self.runtime.get("available_gpus")
        if runtime_available in (None, ""):
            return None
        if not isinstance(runtime_available, list) or not all(
            isinstance(item, int) for item in runtime_available
        ):
            raise RunProfileError(
                "prompt_rewrites runtime.available_gpus must be a list of integers."
            )
        return list(runtime_available)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload["target_models"] = list(self.target_models)
        payload["source_model"] = self.source_model
        payload["template"] = self.template
        payload["style_family"] = self.style_family
        payload["generation_defaults"] = dict(self.generation_defaults)
        payload["runtime"] = dict(self.runtime)
        payload["failure_policy"] = self.failure_policy
        payload["stage_order"] = self.stage_order
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
    global_negative_prompt = _normalize_optional_text(payload.get("global_negative_prompt"))

    conditionings = _normalize_conditionings(
        payload.get("conditionings"),
        profile_path=profile_path,
    )
    prompt_rewrites = _normalize_prompt_rewrites(
        payload.get("prompt_rewrites"),
        profile_path=profile_path,
    )

    return RunProfile(
        path=profile_path,
        name=str(payload["name"]) if payload.get("name") not in (None, "") else None,
        model_names=model_names,
        prompt_file=prompt_file,
        execution_mode=execution_mode,
        output_dir=output_dir,
        generation_defaults=generation_defaults,
        runtime=runtime,
        global_negative_prompt=global_negative_prompt,
        conditionings=conditionings,
        prompt_rewrites=prompt_rewrites,
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
        "global_negative_prompt": profile.global_negative_prompt if profile is not None else None,
        "conditionings": [conditioning.to_dict() for conditioning in profile.conditionings]
        if profile is not None
        else [],
        "prompt_rewrites": [rewrite.to_dict() for rewrite in profile.prompt_rewrites]
        if profile is not None
        else [],
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


def _normalize_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None


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


def _normalize_conditionings(
    value: Any,
    *,
    profile_path: Path,
) -> list[ConditioningSpec]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise RunProfileError(f"Run profile {profile_path} has invalid conditionings section.")

    specs: list[ConditioningSpec] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise RunProfileError(
                f"Run profile {profile_path} conditioning #{index} must be an object."
            )
        targets_value = item.get("target_models", item.get("target_model"))
        target_models = _normalize_model_names(targets_value)
        if not target_models:
            raise RunProfileError(
                f"Run profile {profile_path} conditioning #{index} must define target_model(s)."
            )
        conditioning_type = str(item.get("conditioning_type", item.get("type", ""))).strip().lower()
        if conditioning_type not in {"image", "audio", "text", "video"}:
            raise RunProfileError(
                f"Run profile {profile_path} conditioning #{index} has invalid conditioning_type: "
                f"{conditioning_type or '<empty>'}"
            )
        source_mode = str(item.get("source_mode", "")).strip().lower()
        if source_mode not in {"provided", "generated"}:
            raise RunProfileError(
                f"Run profile {profile_path} conditioning #{index} has invalid source_mode: "
                f"{source_mode or '<empty>'}"
            )
        source_model = item.get("source_model")
        if source_mode == "generated":
            source_model = str(source_model or "").strip()
            if not source_model:
                raise RunProfileError(
                    f"Run profile {profile_path} conditioning #{index} must define source_model "
                    "when source_mode=generated."
                )
        else:
            source_model = str(source_model).strip() if source_model not in (None, "") else None

        generation_defaults = item.get("generation_defaults") or {}
        if not isinstance(generation_defaults, dict):
            raise RunProfileError(
                f"Run profile {profile_path} conditioning #{index} has invalid generation_defaults."
            )
        try:
            generation_defaults = validate_generation_parameters(
                generation_defaults,
                owner_label=f"profile={profile_path} conditioning={index}",
                prompt_source=profile_path,
                warn=None,
            )
        except ValueError as exc:
            raise RunProfileError(str(exc)) from exc

        artifact_retention = item.get("artifact_retention")
        if artifact_retention not in (None, ""):
            artifact_retention = str(artifact_retention).strip()

        specs.append(
            ConditioningSpec(
                target_models=target_models,
                conditioning_type=conditioning_type,
                source_mode=source_mode,
                source_model=source_model,
                generation_defaults=generation_defaults,
                artifact_retention=artifact_retention,
                raw=dict(item),
            )
        )
    return specs


def _normalize_prompt_rewrites(
    value: Any,
    *,
    profile_path: Path,
) -> list[PromptRewriteSpec]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise RunProfileError(f"Run profile {profile_path} has invalid prompt_rewrites section.")

    specs: list[PromptRewriteSpec] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} must be an object."
            )
        targets_value = item.get("target_models", item.get("target_model"))
        target_models = _normalize_model_names(targets_value)
        if not target_models:
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} must define target_model(s)."
            )

        source_model = str(item.get("source_model", "")).strip()
        if not source_model:
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} must define source_model."
            )
        template = str(item.get("template", "")).strip()
        if not template:
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} must define template."
            )
        style_family = str(item.get("style_family", "")).strip()
        if not style_family:
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} must define style_family."
            )

        generation_defaults = item.get("generation_defaults") or {}
        if not isinstance(generation_defaults, dict):
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} has invalid generation_defaults."
            )
        try:
            generation_defaults = validate_generation_parameters(
                generation_defaults,
                owner_label=f"profile={profile_path} prompt_rewrites={index}",
                prompt_source=profile_path,
                warn=None,
            )
        except ValueError as exc:
            raise RunProfileError(str(exc)) from exc

        runtime = item.get("runtime") or {}
        if not isinstance(runtime, dict):
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} has invalid runtime section."
            )
        runtime_available = runtime.get("available_gpus")
        if runtime_available not in (None, ""):
            if not isinstance(runtime_available, list) or not all(
                isinstance(gpu_id, int) for gpu_id in runtime_available
            ):
                raise RunProfileError(
                    f"Run profile {profile_path} prompt_rewrites #{index} must define "
                    "runtime.available_gpus as a list of integers."
                )
            runtime = dict(runtime)
            runtime["available_gpus"] = list(runtime_available)

        failure_policy = str(item.get("failure_policy", "fallback_original")).strip().lower()
        if failure_policy not in {"fallback_original"}:
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} has invalid failure_policy: "
                f"{failure_policy or '<empty>'}"
            )

        stage_order = str(item.get("stage_order", "before_conditioning")).strip().lower()
        if stage_order not in {"before_conditioning", "after_conditioning"}:
            raise RunProfileError(
                f"Run profile {profile_path} prompt_rewrites #{index} has invalid stage_order: "
                f"{stage_order or '<empty>'}"
            )

        specs.append(
            PromptRewriteSpec(
                target_models=target_models,
                source_model=source_model,
                template=template,
                style_family=style_family,
                generation_defaults=generation_defaults,
                runtime=runtime,
                failure_policy=failure_policy,
                stage_order=stage_order,
                raw=dict(item),
            )
        )
    return specs
