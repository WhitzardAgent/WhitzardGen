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
from hashlib import sha256
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
CONDA_ENVS_ROOT = RUNTIME_ROOT / "conda_envs"
CONDA_PKGS_ROOT = RUNTIME_ROOT / "conda_pkgs"
CONDA_CACHE_ROOT = RUNTIME_ROOT / "cache"
EFFECTIVE_REQUIREMENTS_ROOT = RUNTIME_ROOT / "effective_requirements"
DEFAULT_ENV_METADATA_PATH = RUNTIME_ROOT / "env_metadata.json"
DEFAULT_LOCAL_ENVS_PATH = REPO_ROOT / "configs" / "local_envs.yaml"
DEFAULT_ENV_VALIDATION_TTL_SEC = 6 * 60 * 60
ENV_VALIDATION_TTL_ENV_VAR = "AIGC_ENV_VALIDATION_TTL_SEC"


class EnvManagerError(RuntimeError):
    """Raised for environment manager failures."""


@dataclass(slots=True)
class EnvSpec:
    spec_name: str
    directory: Path
    python_version_file: Path
    python_version: str
    requirements_file: Path | None
    pip_install_args: list[str]
    pip_requirement_overrides: dict[str, str]
    reuse_prefix: Path | None
    post_install_file: Path | None
    validation_file: Path | None
    validation_imports: list[str]
    local_override_source: str | None = None


@dataclass(slots=True)
class EnvironmentRecord:
    model_name: str
    env_spec: str
    env_id: str
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
        CONDA_ENVS_ROOT.mkdir(parents=True, exist_ok=True)
        CONDA_PKGS_ROOT.mkdir(parents=True, exist_ok=True)
        CONDA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        EFFECTIVE_REQUIREMENTS_ROOT.mkdir(parents=True, exist_ok=True)

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
        raw_reuse_prefix = local_override.get("reuse_prefix")
        reuse_prefix = Path(str(raw_reuse_prefix)) if raw_reuse_prefix not in (None, "") else None
        return EnvSpec(
            spec_name=spec_name,
            directory=spec_dir,
            python_version_file=python_version_file,
            python_version=effective_python_version,
            requirements_file=effective_requirements_file if effective_requirements_file.exists() else None,
            pip_install_args=pip_install_args,
            pip_requirement_overrides=pip_requirement_overrides,
            reuse_prefix=reuse_prefix,
            post_install_file=post_install_file if post_install_file.exists() else None,
            validation_file=validation_file if validation_file.exists() else None,
            validation_imports=validation_imports,
            local_override_source=str(self.local_envs_path) if local_override else None,
        )

    def compute_env_id(self, spec: EnvSpec) -> str:
        digest = sha256()
        digest.update(spec.python_version.encode("utf-8"))
        digest.update(json.dumps(spec.pip_install_args, sort_keys=True).encode("utf-8"))
        digest.update(
            json.dumps(spec.pip_requirement_overrides, sort_keys=True).encode("utf-8")
        )
        if spec.reuse_prefix is not None:
            digest.update(str(spec.reuse_prefix).encode("utf-8"))
        for file_path in (spec.requirements_file, spec.post_install_file, spec.validation_file):
            if file_path is None or not file_path.exists():
                continue
            digest.update(file_path.read_bytes())
        short_hash = digest.hexdigest()[:8]
        sanitized_name = spec.spec_name.replace("-", "_")
        return f"env_{sanitized_name}_{short_hash}"

    def conda_available(self) -> bool:
        return shutil.which("conda") is not None

    def environment_exists(
        self,
        env_id: str,
        *,
        prefix_override: str | Path | None = None,
    ) -> tuple[bool, str | None]:
        env_path = self.environment_prefix(env_id, prefix_override=prefix_override)
        if env_path.exists():
            return True, str(env_path)
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
            env_path=spec.reuse_prefix,
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
        wrapped = ["conda", "run"]
        if foreground:
            wrapped.append("--no-capture-output")
        wrapped.extend(["--prefix", str(self.environment_prefix(env_id, prefix_override=env_path)), *command])
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

        if record.state == "creating":
            if self._creation_lock_path(record.env_id).exists():
                self._emit_progress(
                    progress_cb,
                    f"Environment for {model_name} is currently marked as creating. Waiting for env {record.env_id}...",
                )
                deadline = time.monotonic() + max(wait_timeout_sec, poll_interval_sec)
                while time.monotonic() < deadline:
                    time.sleep(poll_interval_sec)
                    record = self.inspect_model_environment(model_name)
                    if record.state == "ready":
                        self._emit_progress(progress_cb, f"Environment ready: {record.env_id}")
                        return record
                    if record.state in {"failed", "invalid", "missing"}:
                        break
                    if not self._creation_lock_path(record.env_id).exists():
                        break
                record = self.inspect_model_environment(model_name)

            if record.state == "creating" and not self._creation_lock_path(record.env_id).exists():
                self._emit_progress(
                    progress_cb,
                    f"Recovering stale creating state for {model_name}: {record.env_id}",
                )
                spec = self.resolve_spec_for_model(model_name)
                self._update_metadata(
                    env_id=record.env_id,
                    env_spec=spec.spec_name,
                    models=[model_name],
                    state="failed",
                    path=record.path,
                    validation={
                        "passed": False,
                        "error": "Recovered stale creating state before foreground recreation.",
                        "checked_at": datetime.now(UTC).isoformat(),
                    },
                    error="Recovered stale creating state before foreground recreation.",
                )
                record = self.inspect_model_environment(model_name)

        if record.state == "ready":
            return record

        if record.state in {"missing", "failed", "invalid"}:
            record = self.create_environment(
                model_name,
                foreground=foreground,
                progress=progress_cb,
            )
            if record.state == "ready":
                return record
            raise EnvManagerError(
                f"Failed to create environment for {model_name}. "
                f"Env spec: {record.env_spec}. "
                f"Reason: {record.error or f'environment ended in state {record.state}'}"
            )

        raise EnvManagerError(f"Environment for {model_name} is not ready: {record.state}")

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
        model = self.registry.get_model(model_name)
        spec = self.resolve_spec_for_model(model_name)
        env_id = self.compute_env_id(spec)
        if not self.conda_available():
            raise EnvManagerError("Conda is not available in PATH.")
        progress_cb = self._resolve_progress_callback(foreground=foreground, progress=progress)

        with self._env_lock(env_id):
            self._emit_progress(
                progress_cb,
                f"Environment for {model_name} is not ready.",
            )
            self._emit_progress(
                progress_cb,
                f"Creating Conda environment: {env_id}",
            )
            self._emit_progress(
                progress_cb,
                f"Using Python version: {spec.python_version} ({spec.python_version_file})",
            )
            if spec.local_override_source:
                self._emit_progress(
                    progress_cb,
                    f"Using local env overrides: {spec.local_override_source}",
                )
            if spec.reuse_prefix is not None:
                self._emit_progress(
                    progress_cb,
                    f"Reusing existing environment prefix: {spec.reuse_prefix}",
                )
            if spec.requirements_file is not None:
                self._emit_progress(
                    progress_cb,
                    f"Using requirements: {spec.requirements_file}",
                )
            self._remove_stale_environment_prefix(env_id=env_id, progress=progress_cb)
            if spec.reuse_prefix is not None:
                raise EnvManagerError(
                    f"Configured reuse_prefix for {model_name} does not exist or is not ready: {spec.reuse_prefix}"
                )
            self._update_metadata(
                env_id=env_id,
                env_spec=spec.spec_name,
                models=[model_name],
                state="creating",
                path=None,
                validation={"passed": False, "error": None, "checked_at": datetime.now(UTC).isoformat()},
                error=None,
            )
            create_command = [
                "conda",
                "create",
                "--prefix",
                str(self.environment_prefix(env_id, prefix_override=spec.reuse_prefix)),
                "-y",
                f"python={spec.python_version}",
                "pip",
            ]
            result = self._invoke_run_command(
                create_command,
                env=self.conda_process_env(),
                foreground=foreground,
                progress=progress_cb,
            )
            if result.returncode != 0:
                error = self._command_logs(result) or "conda create failed"
                self._update_metadata(
                    env_id=env_id,
                    env_spec=spec.spec_name,
                    models=[model_name],
                    state="failed",
                    path=None,
                    validation={
                        "passed": False,
                        "error": error,
                        "checked_at": datetime.now(UTC).isoformat(),
                    },
                    error=error,
                )
                raise EnvManagerError(error)

            if spec.requirements_file is not None:
                effective_requirements_file = self._build_effective_requirements_file(
                    env_id=env_id,
                    spec=spec,
                )
                self._emit_progress(progress_cb, "Installing pip requirements...")
                if effective_requirements_file != spec.requirements_file:
                    self._emit_progress(
                        progress_cb,
                        f"Using effective requirements: {effective_requirements_file}",
                    )
                pip_command = self.wrap_command(
                    env_id,
                    [
                        "python",
                        "-m",
                        "pip",
                        "install",
                        *spec.pip_install_args,
                        "-r",
                        str(effective_requirements_file),
                    ],
                    foreground=foreground,
                    env_path=spec.reuse_prefix,
                )
                pip_result = self._invoke_run_command(
                    pip_command,
                    env=self.conda_process_env(),
                    foreground=foreground,
                    progress=progress_cb,
                )
                if pip_result.returncode != 0:
                    error = self._command_logs(pip_result) or "pip install failed"
                    self._update_metadata(
                        env_id=env_id,
                        env_spec=spec.spec_name,
                        models=[model_name],
                        state="failed",
                        path=None,
                        validation={
                            "passed": False,
                            "error": error,
                            "checked_at": datetime.now(UTC).isoformat(),
                        },
                        error=error,
                    )
                    raise EnvManagerError(error)

            if spec.post_install_file is not None:
                self._emit_progress(progress_cb, f"Running post-install hook: {spec.post_install_file}")
                post_install_command = self.wrap_command(
                    env_id,
                    ["bash", str(spec.post_install_file)],
                    foreground=foreground,
                    env_path=spec.reuse_prefix,
                )
                post_install_result = self._invoke_run_command(
                    post_install_command,
                    env=self.conda_process_env(),
                    foreground=foreground,
                    progress=progress_cb,
                )
                if post_install_result.returncode != 0:
                    error = self._command_logs(post_install_result) or "post-install hook failed"
                    self._update_metadata(
                        env_id=env_id,
                        env_spec=spec.spec_name,
                        models=[model_name],
                        state="failed",
                        path=None,
                        validation={
                            "passed": False,
                            "error": error,
                            "checked_at": datetime.now(UTC).isoformat(),
                        },
                        error=error,
                    )
                    raise EnvManagerError(error)

            self._emit_progress(progress_cb, "Validating environment...")
            self._update_metadata(
                env_id=env_id,
                env_spec=spec.spec_name,
                models=[model_name],
                state="validating",
                path=str(self.environment_prefix(env_id, prefix_override=spec.reuse_prefix)),
                validation={
                    "passed": False,
                    "error": None,
                    "checked_at": datetime.now(UTC).isoformat(),
                },
                error=None,
            )
            exists, env_path = self.environment_exists(env_id, prefix_override=spec.reuse_prefix)
            validation_passed, validation_error = self.validate_environment(env_id, spec)
            if exists and validation_passed:
                state = "ready"
            else:
                state = "failed"
            self._update_metadata(
                env_id=env_id,
                env_spec=spec.spec_name,
                models=[model_name],
                state=state,
                path=env_path,
                validation={
                    "passed": validation_passed,
                    "error": validation_error,
                    "checked_at": datetime.now(UTC).isoformat(),
                },
                error=validation_error,
            )
            if state == "ready":
                self._emit_progress(progress_cb, f"Environment ready: {env_id}")
            else:
                self._emit_progress(
                    progress_cb,
                    f"Failed to validate environment for {model_name}: {validation_error or 'validation failed'}",
                )

        return self.inspect_model_environment(model_name, force_revalidate=True)

    def environment_prefix(
        self,
        env_id: str,
        *,
        prefix_override: str | Path | None = None,
    ) -> Path:
        if prefix_override not in (None, ""):
            return Path(str(prefix_override))
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

        if conda_available:
            exists, env_path = self.environment_exists(env_id, prefix_override=spec.reuse_prefix)
            if exists:
                if self._can_reuse_cached_validation(
                    metadata=metadata,
                    force_revalidate=force_revalidate,
                ):
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
            state = str(metadata.get("state", "missing"))
            if state not in {"creating", "failed", "validating"}:
                state = "missing"

        return EnvironmentRecord(
            model_name=model.name,
            env_spec=spec.spec_name,
            env_id=env_id,
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

    def _run_command(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        foreground: bool,
        cwd: str | Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if not foreground:
            return subprocess.run(
                command,
                env=env,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                text=True,
                check=False,
            )

        process = subprocess.Popen(
            command,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            text = line.rstrip("\n")
            if text:
                self._emit_progress(progress, text)
        returncode = process.wait()
        logs = "".join(output_lines).strip()
        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=logs,
            stderr="",
        )

    def _invoke_run_command(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        foreground: bool,
        cwd: str | Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            signature = inspect.signature(self._run_command)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "progress" in signature.parameters:
            return self._run_command(
                command,
                env=env,
                foreground=foreground,
                cwd=cwd,
                progress=progress,
            )
        return self._run_command(
            command,
            env=env,
            foreground=foreground,
            cwd=cwd,
        )

    def _command_logs(self, result: subprocess.CompletedProcess[str]) -> str:
        return ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()

    def _build_effective_requirements_file(self, *, env_id: str, spec: EnvSpec) -> Path:
        if spec.requirements_file is None:
            raise EnvManagerError(f"No requirements file is configured for env spec {spec.spec_name}.")
        if not spec.pip_requirement_overrides:
            return spec.requirements_file

        effective_lines: list[str] = []
        raw_lines = spec.requirements_file.read_text(encoding="utf-8").splitlines()
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                effective_lines.append(line)
                continue
            replacement = self._resolve_requirement_override(stripped, spec.pip_requirement_overrides)
            effective_lines.append(replacement or line)

        output_path = EFFECTIVE_REQUIREMENTS_ROOT / f"{env_id}.requirements.txt"
        output_path.write_text("\n".join(effective_lines) + "\n", encoding="utf-8")
        return output_path

    def _resolve_requirement_override(
        self,
        requirement: str,
        overrides: dict[str, str],
    ) -> str | None:
        candidate_keys = [requirement, requirement.lower()]
        package_name = self._extract_requirement_name(requirement)
        if package_name:
            candidate_keys.extend([package_name, package_name.lower()])
        for key in candidate_keys:
            if key in overrides:
                return overrides[key]
        return None

    def _extract_requirement_name(self, requirement: str) -> str | None:
        if "#egg=" in requirement:
            return requirement.split("#egg=", maxsplit=1)[1].strip()
        if requirement.startswith(("git+", "http://", "https://")):
            tail = requirement.rstrip("/").split("/")[-1]
            if tail.endswith(".git"):
                tail = tail[:-4]
            if tail:
                return tail
        separators = ["==", ">=", "<=", "!=", "~=", ">", "<", "[", " "]
        token = requirement
        for separator in separators:
            if separator in token:
                token = token.split(separator, maxsplit=1)[0]
        token = token.strip()
        return token or None

    def _remove_stale_environment_prefix(
        self,
        *,
        env_id: str,
        progress: Callable[[str], None] | None,
    ) -> None:
        prefix = self.environment_prefix(env_id)
        if not prefix.exists():
            return
        self._emit_progress(progress, f"Removing stale environment prefix: {prefix}")
        shutil.rmtree(prefix)

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _update_metadata(
        self,
        *,
        env_id: str,
        env_spec: str,
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
            target = Path(str(value))
            checks[field] = {
                "value": str(target),
                "exists": target.exists(),
                "kind": "directory" if target.is_dir() else "file" if target.is_file() else "missing",
            }
        return checks

    def _creation_lock_path(self, env_id: str) -> Path:
        return LOCKS_ROOT / f"{env_id}.lock"

    def _resolve_local_envs_path(self, path: str | Path | None) -> Path | None:
        if path is not None:
            return Path(path)
        env_value = os.environ.get(LOCAL_ENVS_ENV_VAR)
        if env_value:
            return Path(env_value)
        return DEFAULT_LOCAL_ENVS_PATH

    @contextmanager
    def _env_lock(self, env_id: str) -> Iterator[None]:
        lock_path = self._creation_lock_path(env_id)
        fd = None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            yield
        finally:
            if fd is not None:
                os.close(fd)
            if lock_path.exists():
                lock_path.unlink()
