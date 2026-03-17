from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from aigc.settings import get_runs_root

RUNS_ROOT = get_runs_root()
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_FAILURES_NAME = "failures.json"


class RunStoreError(RuntimeError):
    """Raised when run artifacts cannot be located or parsed."""


def write_run_manifest(run_root: str | Path, manifest: dict[str, Any]) -> Path:
    target = Path(run_root) / RUN_MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def write_failures_summary(run_root: str | Path, failures: list[dict[str, Any]]) -> Path:
    target = Path(run_root) / RUN_FAILURES_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_run_manifest(run_id: str, runs_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(runs_root) if runs_root is not None else get_runs_root()
    run_root = root / run_id
    manifest_path = run_root / RUN_MANIFEST_NAME
    legacy_manifest_path = run_root / "run.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if legacy_manifest_path.exists():
        payload = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("status", "completed")
        payload.setdefault("manifest_path", str(legacy_manifest_path))
        return payload
    raise RunStoreError(f"Run manifest not found for run_id={run_id}")


def load_failures_summary(run_id: str, runs_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(runs_root) if runs_root is not None else get_runs_root()
    failures_path = root / run_id / RUN_FAILURES_NAME
    if not failures_path.exists():
        return []
    payload = json.loads(failures_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RunStoreError(f"Invalid failures summary for run_id={run_id}")
    return payload


def list_runs(runs_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(runs_root) if runs_root is not None else get_runs_root()
    if not root.exists():
        return []

    manifests: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        try:
            manifest = load_run_manifest(run_dir.name, runs_root=root)
        except RunStoreError:
            continue
        manifest.setdefault("run_id", run_dir.name)
        manifests.append(manifest)
    return manifests


def export_dataset_for_run(
    run_id: str,
    *,
    runs_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    manifest = load_run_manifest(run_id, runs_root=runs_root)
    export_path = Path(manifest["export_path"])
    if not export_path.exists():
        raise RunStoreError(f"Dataset export missing for run_id={run_id}: {export_path}")
    if output_path is None:
        return export_path

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(export_path, target)
    return target
