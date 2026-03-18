from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import inspect
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from aigc.env.local_overrides import (
    LOCAL_ENVS_ENV_VAR,
    load_local_env_overrides,
    resolve_local_env_override,
)
from aigc.registry import ModelRegistry, load_registry
from aigc.utils.runtime_logging import print_log_line

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVS_ROOT = REPO_ROOT / "envs"
RUNTIME_ROOT = REPO_ROOT / "runtime"
LOCKS_ROOT = RUNTIME_ROOT / "locks"
DEFAULT_ENV_METADATA_PATH = RUNTIME_ROOT / "env_metadata.json"
DEFAULT_LOCAL_ENVS_PATH = REPO_ROOT / "configs" / "local_envs.yaml"
DEFAULT_ENV_VALIDATION_TTL_SEC = 6 * 60 * 60
ENV_VALIDATION_TTL_ENV_VAR = "AIGC_ENV_VALIDATION_TTL_SEC"


class EnvManagerError(RuntimeError):
    """Raised for environment manager failures."""


class MissingEnvironmentError(EnvManagerError):
    """Raised when a required conda environment does not exist."""

    def __init__(self, model_name: str, conda_env_name: str) -> None:
        self.model_name = model_name
        self.conda_env_name = conda_env_name
        super().__init__(
            f"Environment for {model_name} is not available.\n"
            f"Required conda env: {conda_env_name}\n"
            f"Please create this environment manually before running."
        )


@dataclass(slots=True)
class EnvSpec:
    spec_name: str
    directory: Path
    python_version_file: Path
    python_version: str
    requirements_file: Path | None
    pip_install_args: list[str]
    pip_requirement_overrides: dict[str, str]
    post_install_file: Path | None
    validation_file: Path | None
    validation_imports: list[str]
    local_override_source: str | None = None


@dataclass(slots=True)
class EnvironmentRecord:
    model_name: str
    env_spec: str
    conda_env_name: str
    state: str
    conda_available: bool
    execution_mode: str
    exists: bool
    validation_passed: bool
    path: str | None
    associated_models: list[str]
    local_paths: dict[str, Any]
    path_checks: dict[str, dict[str, Any]]
    last_validation: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EnvManager:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        metadata_path: Path = DEFAULT_ENV_METADATA_PATH,
        local_envs_path: str | Path | None = None,
        validation_ttl_sec: int | None = None,
    ) -> None:
        self.registry = registry or load_registry()
        self.metadata_path = metadata_path
        resolved_local_envs_path = self._resolve_local_envs_path(local_envs_path)
        self.local_envs_path, self.local_env_overrides = load_local_env_overrides(
            resolved_local_envs_path
        )
        self.validation_ttl_sec = self._resolve_validation_ttl_sec(validation_ttl_sec)
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        LOCKS_ROOT.mkdir(parents=True, exist_ok=True)

    def resolve_conda_env_name(self, model_name: str) -> str:
        model = self.registry.get_model(model_name)
        return model.conda_env_name

    def resolve_spec_for_model(self, model_name: str) -> EnvSpec:
        model = self.registry.get_model(model_name)
        spec_name = model.env_spec
        spec_dir = ENVS_ROOT / spec_name
        python_version_file = spec_dir / "python_version.txt"
        if not python_version_file.exists():
            raise EnvManagerError(
                f"Environment spec {spec_name!r} for model {model_name} is missing {python_version_file}"
            )
        python_version = python_version_file.read_text(encoding="utf-8").strip()
        if not python_version:
            raise EnvManagerError(
                f"Environment spec {spec_name!r} for model {model_name} has an empty {python_version_file}"
            )
        requirements_file = spec_dir / "requirements.txt"
        post_install_file = spec_dir / "post_install.sh"
        validation_file = spec_dir / "validation.json"
        validation_imports = []
        if validation_file.exists():
            payload = json.loads(validation_file.read_text(encoding="utf-8"))
            validation_imports = list(payload.get("imports", []))
        local_override = resolve_local_env_override(
            self.local_env_overrides,
            model_name=model_name,
            env_spec=spec_name,
        )
        effective_python_version = str(local_override.get("python_version", python_version)).strip()
        requirements_override = local_override.get("requirements_file")
        effective_requirements_file = (
            Path(str(requirements_override)) if requirements_override else requirements_file
        )
        if effective_requirements_file is not None and not effective_requirements_file.exists():
            raise EnvManagerError(
                f"Effective requirements file for {model_name} does not exist: {effective_requirements_file}"
            )
        raw_pip_install_args = local_override.get("pip_install_args", [])
        if raw_pip_install_args in (None, ""):
            raw_pip_install_args = []
        if not isinstance(raw_pip_install_args, list):
            raise EnvManagerError(
                f"Local env override for {model_name} has invalid pip_install_args; expected a list."
            )
        pip_install_args = [str(item) for item in raw_pip_install_args if str(item).strip()]
        raw_requirement_overrides = local_override.get("pip_requirement_overrides", {})
        if raw_requirement_overrides in (None, ""):
            raw_requirement_overrides = {}
        if not isinstance(raw_requirement_overrides, dict):
            raise EnvManagerError(
                f"Local env override for {model_name} has invalid pip_requirement_overrides; expected a mapping."
            )
        pip_requirement_overrides = {
            str(key): str(value)
            for key, value in raw_requirement_overrides.items()
            if str(value).strip()
        }
        return EnvSpec(
            spec_name=spec_name,
            directory=spec_dir,
            python_version_file=python_version_file,
            python_version=effective_python_version,
            requirements_file=effective_requirements_file if effective_requirements_file.exists() else None,
            pip_install_args=pip_install_args,
            pip_requirement_overrides=pip_requirement_overrides,
            post_install_file=post_install_file if post_install_file.exists() else None,
            validation_file=validation_file if validation_file.exists() else None,
            validation_imports=validation_imports,
            local_override_source=str(self.local_envs_path) if local_override else None,
        )

    def conda_available(self) -> bool:
        return shutil.which("conda") is not None

    def conda_env_exists(self, env_name: str) -> tuple[bool, str | None]:
        if not self.conda_available():
            return False, None
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False, None
        try:
            payload = json.loads(result.stdout)
            envs = payload.get("envs", [])
            for env_path in envs:
                if Path(env_path).name == env_name:
                    return True, env_path
        except json.JSONDecodeError:
            pass
        return False, None

    def validate_environment(self, conda_env_name: str, spec: EnvSpec) -> tuple[bool, str | None]:
        if not self.conda_available():
            return False, "Conda is not available."
        if not spec.validation_imports:
            return True, None
        import_statements = "; ".join(f"import {module}" for module in spec.validation_imports)
        command = self.wrap_command(
            conda_env_name,
            ["python", "-c", import_statements],
            foreground=False,
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip() or "Validation import failed."
            return False, error
        return True, None

    def wrap_command(
        self,
        conda_env_name: str,
        command: list[str],
        *,
        foreground: bool = False,
    ) -> list[str]:
        wrapped = ["conda", "run"]
        if foreground:
            wrapped.append("--no-capture-output")
        wrapped.extend(["-n", conda_env_name, *command])
        return wrapped

    def ensure_ready(
        self,
        model_name: str,
        *,
        foreground: bool = True,
        progress: Callable[[str], None] | None = None,
        wait_timeout_sec: float = 1800.0,
        poll_interval_sec: float = 2.0,
    ) -> EnvironmentRecord:
        progress_cb = self._resolve_progress_callback(foreground=foreground, progress=progress)
        record = self.inspect_model_environment(model_name)
        if record.state == "ready":
            return record
        if not record.conda_available:
            raise EnvManagerError(
                f"Failed to prepare environment for {model_name}: Conda is not available in PATH."
            )
        if record.state == "missing":
            raise MissingEnvironmentError(model_name, record.conda_env_name)
        raise EnvManagerError(
            f"Environment for {model_name} is not ready: {record.state}.\n"
            f"Required conda env: {record.conda_env_name}"
        )

    def build_model_process_env(self, model_name: str) -> dict[str, str]:
        env = os.environ.copy()
        model = self.registry.get_model(model_name)
        hf_cache_dir = model.weights.get("hf_cache_dir")
        if hf_cache_dir:
            env["HF_HOME"] = str(hf_cache_dir)
            env["HF_HUB_CACHE"] = str(hf_cache_dir)
            env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir)
            env["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
        for field in ("local_path", "repo_path", "weights_path", "script_root"):
            value = model.weights.get(field)
            if value:
                env[f"AIGC_MODEL_{field.upper()}"] = str(value)
        env["AIGC_MODEL_NAME"] = model.name
        return env

    def ensure_environment(self, model_name: str) -> EnvironmentRecord:
        return self.ensure_ready(model_name, foreground=False)

    def inspect_model_environment(
        self,
        model_name: str,
        *,
        force_revalidate: bool = False,
    ) -> EnvironmentRecord:
        model = self.registry.get_model(model_name)
        conda_env_name = model.conda_env_name
        spec = self.resolve_spec_for_model(model_name)
        metadata_key = f"env_{conda_env_name}"
        metadata = self._load_metadata().get(metadata_key, {})
        metadata_models = list(metadata.get("models", []))

        conda_available = self.conda_available()
        exists = False
        env_path = None
        validation_passed = False
        error = metadata.get("error")
        last_validation = dict(metadata.get("last_validation", {}))

        if conda_available:
            exists, env_path = self.conda_env_exists(conda_env_name)
            if exists:
                if self._can_reuse_cached_validation(
                    metadata=metadata,
                    force_revalidate=force_revalidate,
                ):
                    validation_passed = bool(last_validation.get("passed", False))
                    validation_error = last_validation.get("error")
                else:
                    validation_passed, validation_error = self.validate_environment(conda_env_name, spec)
                    last_validation = {
                        "passed": validation_passed,
                        "error": validation_error,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
                    self._update_metadata(
                        metadata_key=metadata_key,
                        conda_env_name=conda_env_name,
                        env_spec=spec.spec_name,
                        models=[model.name],
                        state="ready" if validation_passed else "invalid",
                        path=env_path,
                        validation=last_validation,
                        error=validation_error,
                    )
                if validation_error:
                    error = validation_error

        if not conda_available:
            state = "missing"
            error = error or "Conda is not available."
        elif exists and validation_passed:
            state = "ready"
        elif exists and not validation_passed:
            state = "invalid"
        else:
            state = "missing"

        return EnvironmentRecord(
            model_name=model.name,
            env_spec=spec.spec_name,
            conda_env_name=conda_env_name,
            state=state,
            conda_available=conda_available,
            execution_mode=model.execution_mode,
            exists=exists,
            validation_passed=validation_passed,
            path=env_path,
            associated_models=sorted(set(metadata_models + [model.name])),
            local_paths=dict(model.local_paths),
            path_checks=self._build_path_checks(model),
            last_validation=last_validation,
            error=error,
        )

    def doctor(self, model_name: str | None = None) -> list[EnvironmentRecord]:
        model_names = [model_name] if model_name else [model.name for model in self.registry.list_models()]
        return [self.inspect_model_environment(name, force_revalidate=True) for name in model_names]

    def _resolve_validation_ttl_sec(self, configured: int | None) -> int:
        if configured is not None:
            return max(int(configured), 0)
        raw = os.environ.get(ENV_VALIDATION_TTL_ENV_VAR)
        if raw not in (None, ""):
            try:
                return max(int(raw), 0)
            except ValueError:
                pass
        return DEFAULT_ENV_VALIDATION_TTL_SEC

    def _can_reuse_cached_validation(
        self,
        *,
        metadata: dict[str, Any],
        force_revalidate: bool,
    ) -> bool:
        if force_revalidate:
            return False
        if str(metadata.get("state")) != "ready":
            return False
        last_validation = dict(metadata.get("last_validation", {}))
        if not bool(last_validation.get("passed", False)):
            return False
        checked_at = last_validation.get("checked_at")
        if not checked_at:
            return False
        if self.validation_ttl_sec <= 0:
            return False
        try:
            checked_at_dt = datetime.fromisoformat(str(checked_at))
        except ValueError:
            return False
        age_seconds = (datetime.now(UTC) - checked_at_dt).total_seconds()
        return age_seconds <= self.validation_ttl_sec

    def _resolve_progress_callback(
        self,
        *,
        foreground: bool,
        progress: Callable[[str], None] | None,
    ) -> Callable[[str], None] | None:
        if progress is not None:
            return progress
        if not foreground:
            return None
        return self._default_progress

    def _emit_progress(
        self,
        progress: Callable[[str], None] | None,
        message: str,
    ) -> None:
        if progress is not None:
            progress(message)

    def _default_progress(self, message: str) -> None:
        print_log_line(message, stream=sys.stderr)

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _update_metadata(
        self,
        *,
        metadata_key: str,
        conda_env_name: str,
        env_spec: str,
        models: list[str],
        state: str,
        path: str | None,
        validation: dict[str, Any],
        error: str | None,
    ) -> None:
        metadata = self._load_metadata()
        now = datetime.now(UTC).isoformat()
        record = metadata.get(metadata_key, {})
        record.update(
            {
                "conda_env_name": conda_env_name,
                "env_spec": env_spec,
                "models": sorted(set(models + list(record.get("models", [])))),
                "state": state,
                "path": path,
                "last_validation": validation,
                "error": error,
                "updated_at": now,
            }
        )
        if "created_at" not in record:
            record["created_at"] = now
        metadata[metadata_key] = record
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _build_path_checks(self, model: Any) -> dict[str, dict[str, Any]]:
        checks: dict[str, dict[str, Any]] = {}
        for field, value in sorted(model.local_paths.items()):
            target = Path(str(value))
            checks[field] = {
                "value": str(target),
                "exists": target.exists(),
                "kind": "directory" if target.is_dir() else "file" if target.is_file() else "missing",
            }
        return checks

    def _resolve_local_envs_path(self, path: str | Path | None) -> Path | None:
        if path is not None:
            return Path(path)
        env_value = os.environ.get(LOCAL_ENVS_ENV_VAR)
        if env_value:
            return Path(env_value)
        return DEFAULT_LOCAL_ENVS_PATH
