from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc.exporters import ExportBundleResult, ExportBundleSource, export_dataset_bundle, export_dataset_bundle_for_runs
from aigc.settings import get_runs_root
from aigc.runtime.payloads import TaskPayload

RUNS_ROOT = get_runs_root()
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_FAILURES_NAME = "failures.json"
RUN_SAMPLES_NAME = "samples.jsonl"


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


def load_samples_ledger(run_id: str, runs_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(runs_root) if runs_root is not None else get_runs_root()
    ledger_path = root / run_id / RUN_SAMPLES_NAME
    if not ledger_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RunStoreError(f"Invalid samples ledger record for run_id={run_id}")
        records.append(payload)
    return records


def load_task_payloads(run_id: str, runs_root: str | Path | None = None) -> list[TaskPayload]:
    root = Path(runs_root) if runs_root is not None else get_runs_root()
    tasks_root = root / run_id / "tasks"
    if not tasks_root.exists():
        return []
    payloads: list[TaskPayload] = []
    for task_file in sorted(tasks_root.rglob("*.json")):
        if ".result." in task_file.name:
            continue
        payload = json.loads(task_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RunStoreError(f"Invalid task payload file for run_id={run_id}: {task_file}")
        payloads.append(TaskPayload.from_dict(payload))
    return payloads


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
    mode: str = "link",
    selected_models: list[str] | None = None,
) -> ExportBundleResult:
    return export_dataset_for_runs(
        [run_id],
        runs_root=runs_root,
        output_path=output_path,
        mode=mode,
        selected_models=selected_models,
    )


def export_dataset_for_runs(
    run_ids: list[str],
    *,
    runs_root: str | Path | None = None,
    output_path: str | Path | None = None,
    mode: str = "link",
    selected_models: list[str] | None = None,
) -> ExportBundleResult:
    if not run_ids:
        raise RunStoreError("At least one run_id is required for dataset export.")

    root = Path(runs_root) if runs_root is not None else get_runs_root()
    sources: list[ExportBundleSource] = []
    for run_id in run_ids:
        manifest = load_run_manifest(run_id, runs_root=root)
        export_path = Path(manifest["export_path"])
        if not export_path.exists():
            raise RunStoreError(f"Dataset export missing for run_id={run_id}: {export_path}")
        manifest.setdefault("manifest_path", str(root / run_id / RUN_MANIFEST_NAME))
        sources.append(
            ExportBundleSource(
                run_id=run_id,
                source_manifest=manifest,
                source_dataset_path=str(export_path),
            )
        )

    bundle_root = (
        Path(output_path)
        if output_path is not None
        else _default_bundle_root(root=root, run_ids=run_ids)
    )
    try:
        if len(sources) == 1:
            source = sources[0]
            return export_dataset_bundle(
                run_id=source.run_id,
                source_manifest=source.source_manifest,
                source_dataset_path=source.source_dataset_path,
                bundle_root=bundle_root,
                mode=mode,
                selected_models=selected_models,
            )
        return export_dataset_bundle_for_runs(
            sources=sources,
            bundle_root=bundle_root,
            mode=mode,
            selected_models=selected_models,
        )
    except Exception as exc:  # pragma: no cover - normalized through tests at higher level
        raise RunStoreError(str(exc)) from exc


def _default_bundle_root(*, root: Path, run_ids: list[str]) -> Path:
    if len(run_ids) == 1:
        return root / run_ids[0] / "exports" / "dataset_bundle"
    lead = run_ids[0]
    suffix = f"{lead}__plus_{len(run_ids) - 1}"
    return root / "exports" / suffix
