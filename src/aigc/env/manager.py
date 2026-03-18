from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from aigc.registry import ModelRegistry, load_registry
from aigc.utils.runtime_logging import print_log_line

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVS_ROOT = REPO_ROOT / "envs"
RUNTIME_ROOT = REPO_ROOT / "runtime"
CONDA_ENVS_ROOT = RUNTIME_ROOT / "conda_envs"
CONDA_PKGS_ROOT = RUNTIME_ROOT / "conda_pkgs"
CONDA_CACHE_ROOT = RUNTIME_ROOT / "cache"
DEFAULT_ENV_METADATA_PATH = RUNTIME_ROOT / "env_metadata.json"
DEFAULT_ENV_VALIDATION_TTL_SEC = 6 * 60 * 60
ENV_VALIDATION_TTL_ENV_VAR = "AIGC_ENV_VALIDATION_TTL_SEC"


class EnvManagerError(RuntimeError):
    """Raised for environment manager failures."""


@dataclass(slots=True)
class EnvSpec:
    spec_name: str
    directory: Path
    conda_env_name: str
    validation_file: Path | None
    validation_imports: list[str]


@dataclass(slots=True)
class EnvironmentRecord:
    model_name: str
    env_spec: str
    env_id: str
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
        self.validation_ttl_sec = self._resolve_validation_ttl_sec(validation_ttl_sec)
        self.local_envs_path = Path(local_envs_path) if local_envs_path is not None else None
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        CONDA_ENVS_ROOT.mkdir(parents=True, exist_ok=True)
        CONDA_PKGS_ROOT.mkdir(parents=True, exist_ok=True)
        CONDA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    def resolve_spec_for_model(self, model_name: str) -> EnvSpec:
        model = self.registry.get_model(model_name)
        spec_name = model.env_spec
        spec_dir = ENVS_ROOT / spec_name
        validation_file = spec_dir / "validation.json"
        validation_imports: list[str] = []
        if validation_file.exists():
            payload = json.loads(validation_file.read_text(encoding="utf-8"))
            validation_imports = [str(item) for item in payload.get("imports", [])]
        return EnvSpec(
            spec_name=spec_name,
            directory=spec_dir,
            conda_env_name=model.conda_env_name,
            validation_file=validation_file if validation_file.exists() else None,
            validation_imports=validation_imports,
        )

    def compute_env_id(self, spec: EnvSpec) -> str:
        return spec.conda_env_name

    def conda_available(self) -> bool:
        return shutil.which("conda") is not None

    def environment_exists(
        self,
        env_id: str,
        *,
        prefix_override: str | Path | None = None,
    ) -> tuple[bool, str | None]:
        del prefix_override
        if not self.conda_available():
            return False, None
        prefixes = self._list_conda_env_prefixes()
        base_prefix = self._conda_base_prefix()
        for prefix in prefixes:
            if self._env_name_matches(prefix, env_id, base_prefix=base_prefix):
                return True, prefix
        return False, None

    def validate_environment(self, env_id: str, spec: EnvSpec) -> tuple[bool, str | None]:
        if not self.conda_available():
            return False, "Conda is not available."
        if not spec.validation_imports:
            return True, None
        import_statements = "; ".join(f"import {module}" for module in spec.validation_imports)
        command = self.wrap_command(
            env_id,
            ["python", "-c", import_statements],
            foreground=False,
        )
        result = subprocess.run(
            command,
            env=self.conda_process_env(),
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
        env_id: str,
        command: list[str],
        *,
        foreground: bool = False,
        env_path: str | Path | None = None,
    ) -> list[str]:
        del env_path
        wrapped = ["conda", "run"]
        if foreground:
            wrapped.append("--no-capture-output")
        wrapped.extend(["-n", str(env_id), *command])
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
        del wait_timeout_sec, poll_interval_sec
        progress_cb = self._resolve_progress_callback(foreground=foreground, progress=progress)
        record = self.inspect_model_environment(model_name)
        if record.state == "ready":
            return record
        if not record.conda_available:
            raise EnvManagerError(
                f"Environment for {model_name} is not available.\n"
                f"Required conda env: {record.conda_env_name}\n"
                "Conda is not available in PATH."
            )
        self._emit_progress(progress_cb, f"Environment for {model_name} is not available.")
        self._emit_progress(progress_cb, f"Required conda env: {record.conda_env_name}")
        if record.state == "missing":
            raise EnvManagerError(
                f"Environment for {model_name} is not available.\n"
                f"Required conda env: {record.conda_env_name}\n"
                "Please create this environment manually before running."
            )
        if record.state == "invalid":
            raise EnvManagerError(
                f"Environment for {model_name} exists but failed validation.\n"
                f"Required conda env: {record.conda_env_name}\n"
                f"Reason: {record.error or 'validation failed'}"
            )
        raise EnvManagerError(
            f"Environment for {model_name} is not ready.\n"
            f"Required conda env: {record.conda_env_name}\n"
            f"State: {record.state}"
        )

    def build_model_process_env(self, model_name: str) -> dict[str, str]:
        env = self.conda_process_env()
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

    def create_environment(
        self,
        model_name: str,
        *,
        foreground: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> EnvironmentRecord:
        del foreground, progress
        record = self.inspect_model_environment(model_name)
        raise EnvManagerError(
            f"Automatic Conda environment creation has been disabled.\n"
            f"Model: {model_name}\n"
            f"Required conda env: {record.conda_env_name}\n"
            "Please create this environment manually before running."
        )

    def environment_prefix(
        self,
        env_id: str,
        *,
        prefix_override: str | Path | None = None,
    ) -> Path:
        if prefix_override not in (None, ""):
            return Path(str(prefix_override))
        exists, path = self.environment_exists(env_id)
        if exists and path:
            return Path(path)
        return CONDA_ENVS_ROOT / env_id

    def conda_process_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CONDA_PKGS_DIRS"] = str(CONDA_PKGS_ROOT)
        env["XDG_CACHE_HOME"] = str(CONDA_CACHE_ROOT)
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
        spec = self.resolve_spec_for_model(model_name)
        env_id = self.compute_env_id(spec)
        metadata = self._load_metadata().get(env_id, {})
        metadata_models = list(metadata.get("models", []))

        conda_available = self.conda_available()
        exists = False
        env_path = metadata.get("path")
        validation_passed = False
        error = metadata.get("error")
        last_validation = dict(metadata.get("last_validation", {}))
        validation_error = last_validation.get("error")

        if conda_available:
            exists, env_path = self.environment_exists(env_id)
            if exists:
                if self._can_reuse_cached_validation(metadata=metadata, force_revalidate=force_revalidate):
                    validation_passed = bool(last_validation.get("passed", False))
                    validation_error = last_validation.get("error")
                else:
                    validation_passed, validation_error = self.validate_environment(env_id, spec)
                    last_validation = {
                        "passed": validation_passed,
                        "error": validation_error,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
                    self._update_metadata(
                        env_id=env_id,
                        env_spec=spec.spec_name,
                        conda_env_name=spec.conda_env_name,
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
            error = error or f"Required conda env {spec.conda_env_name} was not found."

        return EnvironmentRecord(
            model_name=model.name,
            env_spec=spec.spec_name,
            env_id=env_id,
            conda_env_name=spec.conda_env_name,
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
        env_id: str,
        env_spec: str,
        conda_env_name: str,
        models: list[str],
        state: str,
        path: str | None,
        validation: dict[str, Any],
        error: str | None,
    ) -> None:
        metadata = self._load_metadata()
        now = datetime.now(UTC).isoformat()
        record = metadata.get(env_id, {})
        record.update(
            {
                "env_id": env_id,
                "env_spec": env_spec,
                "conda_env_name": conda_env_name,
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
        metadata[env_id] = record
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _build_path_checks(self, model: Any) -> dict[str, dict[str, Any]]:
        checks: dict[str, dict[str, Any]] = {}
        for field, value in sorted(model.local_paths.items()):
            if value in (None, ""):
                continue
            target = Path(str(value))
            checks[field] = {
                "value": str(target),
                "exists": target.exists(),
                "kind": "directory" if target.is_dir() else "file" if target.is_file() else "missing",
            }
        return checks

    def _list_conda_env_prefixes(self) -> list[str]:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            env=self.conda_process_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return []
        raw_envs = payload.get("envs", [])
        if not isinstance(raw_envs, list):
            return []
        return [str(item) for item in raw_envs]

    def _conda_base_prefix(self) -> str | None:
        result = subprocess.run(
            ["conda", "info", "--base"],
            env=self.conda_process_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        value = (result.stdout or "").strip()
        return value or None

    def _env_name_matches(
        self,
        prefix: str,
        env_name: str,
        *,
        base_prefix: str | None,
    ) -> bool:
        prefix_path = Path(prefix)
        if env_name == "base" and base_prefix and Path(base_prefix) == prefix_path:
            return True
        return prefix_path.name == env_name
