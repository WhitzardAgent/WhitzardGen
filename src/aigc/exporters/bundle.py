from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExportBundleError(RuntimeError):
    """Raised when a dataset export bundle cannot be created."""


@dataclass(slots=True)
class ExportBundleSource:
    run_id: str
    source_manifest: dict[str, Any]
    source_dataset_path: str


@dataclass(slots=True)
class ExportBundleResult:
    bundle_path: str
    dataset_path: str
    manifest_path: str
    readme_path: str
    export_mode: str
    source_run_ids: list[str]
    selected_models: list[str]
    exported_models: list[str]
    record_count: int
    skipped_count: int
    filtered_out_count: int
    counts_by_model: dict[str, int]
    counts_by_split: dict[str, int]
    counts_by_artifact_type: dict[str, int]
    counts_by_run_id: dict[str, int]

    @property
    def run_id(self) -> str | None:
        if len(self.source_run_ids) == 1:
            return self.source_run_ids[0]
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ExportBundleError(f"Invalid JSONL record in {path}")
        records.append(payload)
    return records


def export_dataset_bundle(
    *,
    run_id: str,
    source_manifest: dict[str, Any],
    source_dataset_path: str | Path,
    bundle_root: str | Path,
    mode: str = "link",
    selected_models: list[str] | None = None,
) -> ExportBundleResult:
    return export_dataset_bundle_for_runs(
        sources=[
            ExportBundleSource(
                run_id=run_id,
                source_manifest=source_manifest,
                source_dataset_path=str(source_dataset_path),
            )
        ],
        bundle_root=bundle_root,
        mode=mode,
        selected_models=selected_models,
    )


def export_dataset_bundle_for_runs(
    *,
    sources: list[ExportBundleSource],
    bundle_root: str | Path,
    mode: str = "link",
    selected_models: list[str] | None = None,
) -> ExportBundleResult:
    if not sources:
        raise ExportBundleError("At least one source run is required for dataset export.")
    if mode not in {"link", "copy"}:
        raise ExportBundleError(f"Unsupported export mode: {mode}")

    bundle_dir = Path(bundle_root)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    media_root = bundle_dir / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    readme_path = bundle_dir / "README.md"
    manifest_path = bundle_dir / "export_manifest.json"
    dataset_path = bundle_dir / "dataset.jsonl"

    selected_model_set = {model_name for model_name in selected_models or [] if model_name}
    exported_records: list[dict[str, Any]] = []
    skipped_count = 0
    filtered_out_count = 0
    per_source_stats: list[dict[str, Any]] = []
    next_index = 1

    for source in sources:
        source_path = Path(source.source_dataset_path)
        if not source_path.exists():
            raise ExportBundleError(f"Source dataset JSONL missing: {source_path}")

        source_records = load_jsonl_records(source_path)
        source_exported_count = 0
        source_skipped_count = 0
        source_filtered_count = 0
        for source_record in source_records:
            model_name = str(source_record.get("model_name") or "unknown-model")
            if selected_model_set and model_name not in selected_model_set:
                filtered_out_count += 1
                source_filtered_count += 1
                continue

            status = str(source_record.get("execution_metadata", {}).get("status", ""))
            source_artifact_value = str(source_record.get("artifact_path") or "").strip()
            if status != "success" or not source_artifact_value:
                skipped_count += 1
                source_skipped_count += 1
                continue

            source_artifact_path = Path(source_artifact_value)
            if not source_artifact_path.exists() or not source_artifact_path.is_file():
                skipped_count += 1
                source_skipped_count += 1
                continue

            artifact_type = str(source_record.get("artifact_type") or "artifact")
            split_value = _extract_split(source_record)
            exported_relpath = (
                Path("media")
                / _sanitize_path_component(split_value)
                / _sanitize_path_component(model_name)
                / _sanitize_path_component(artifact_type)
                / _exported_filename(
                    run_id=source.run_id,
                    record=source_record,
                    source_path=source_artifact_path,
                )
            )
            destination = bundle_dir / exported_relpath
            _materialize_artifact(
                source_path=source_artifact_path,
                destination=destination,
                mode=mode,
            )
            exported_records.append(
                _build_export_record(
                    record=source_record,
                    record_index=next_index,
                    exported_relpath=exported_relpath,
                    source_artifact_path=source_artifact_path,
                    source_manifest=source.source_manifest,
                    bundle_name=bundle_dir.name,
                    export_mode=mode,
                    split_value=split_value,
                )
            )
            next_index += 1
            source_exported_count += 1

        per_source_stats.append(
            {
                "run_id": source.run_id,
                "status": source.source_manifest.get("status"),
                "manifest_path": source.source_manifest.get("manifest_path"),
                "dataset_path": str(source_path),
                "parent_run_id": source.source_manifest.get("parent_run_id"),
                "source_run_id": source.source_manifest.get("source_run_id"),
                "recovery_mode": source.source_manifest.get("recovery_mode"),
                "exported_record_count": source_exported_count,
                "skipped_record_count": source_skipped_count,
                "filtered_out_count": source_filtered_count,
            }
        )

    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in exported_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts_by_model = _count_records_by_key(exported_records, "model_name")
    counts_by_split = _count_records_by_key(exported_records, "split")
    counts_by_artifact_type = _count_records_by_key(exported_records, "artifact_type")
    counts_by_run_id = _count_records_by_key(exported_records, "run_id")
    exported_models = sorted(counts_by_model)
    manifest = {
        "export_name": bundle_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "source_runs": per_source_stats,
        "selected_models": sorted(selected_model_set),
        "exported_models": exported_models,
        "record_count": len(exported_records),
        "skipped_record_count": skipped_count,
        "filtered_out_count": filtered_out_count,
        "counts_by_model": counts_by_model,
        "counts_by_split": counts_by_split,
        "counts_by_artifact_type": counts_by_artifact_type,
        "counts_by_run_id": counts_by_run_id,
        "export_mode": mode,
        "bundle_path": str(bundle_dir),
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "media_root": str(media_root),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme_path.write_text(
        _build_readme(manifest),
        encoding="utf-8",
    )

    return ExportBundleResult(
        bundle_path=str(bundle_dir),
        dataset_path=str(dataset_path),
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
        export_mode=mode,
        source_run_ids=[source.run_id for source in sources],
        selected_models=sorted(selected_model_set),
        exported_models=exported_models,
        record_count=len(exported_records),
        skipped_count=skipped_count,
        filtered_out_count=filtered_out_count,
        counts_by_model=counts_by_model,
        counts_by_split=counts_by_split,
        counts_by_artifact_type=counts_by_artifact_type,
        counts_by_run_id=counts_by_run_id,
    )


def _build_export_record(
    *,
    record: dict[str, Any],
    record_index: int,
    exported_relpath: Path,
    source_artifact_path: Path,
    source_manifest: dict[str, Any],
    bundle_name: str,
    export_mode: str,
    split_value: str,
) -> dict[str, Any]:
    payload = dict(record)
    payload["source_record_id"] = str(record.get("record_id") or "")
    payload["record_id"] = f"rec_{record_index:08d}"
    payload["artifact_path"] = exported_relpath.as_posix()
    payload["source_artifact_path"] = str(source_artifact_path)
    payload["split"] = split_value
    payload["source_run_lineage"] = {
        "parent_run_id": source_manifest.get("parent_run_id"),
        "source_run_id": source_manifest.get("source_run_id"),
        "recovery_mode": source_manifest.get("recovery_mode"),
    }
    payload["export_metadata"] = {
        "bundle_name": bundle_name,
        "export_mode": export_mode,
    }
    return payload


def _materialize_artifact(*, source_path: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir():
            raise ExportBundleError(f"Export destination is a directory: {destination}")
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source_path, destination)
        return
    try:
        rel_target = os.path.relpath(source_path, start=destination.parent)
        destination.symlink_to(rel_target)
    except OSError as exc:
        raise ExportBundleError(
            f"Failed to create symlink export for {source_path} -> {destination}: {exc}"
        ) from exc


def _extract_split(record: dict[str, Any]) -> str:
    prompt_metadata = record.get("prompt_metadata", {})
    if isinstance(prompt_metadata, dict):
        split = str(prompt_metadata.get("split") or "").strip()
        if split:
            return split
    split = str(record.get("split") or "").strip()
    return split or "unspecified"


def _exported_filename(*, run_id: str, record: dict[str, Any], source_path: Path) -> str:
    source_record_id = str(record.get("record_id") or "").strip()
    prompt_id = str(record.get("prompt_id") or "sample").strip()
    suffix = source_path.suffix or ""
    stem = source_record_id or prompt_id or source_path.stem or "artifact"
    return f"{_sanitize_path_component(run_id)}__{_sanitize_path_component(stem)}{suffix}"


def _sanitize_path_component(value: str) -> str:
    sanitized = (
        value.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("\n", "_")
        .strip()
    )
    return sanitized or "unknown"


def _count_records_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_readme(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Export Bundle: {manifest['export_name']}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Created: {manifest['created_at']}")
    lines.append(f"- Export mode: {manifest['export_mode']}")
    lines.append(f"- Total records: {manifest['record_count']}")
    lines.append(f"- Skipped records: {manifest['skipped_record_count']}")
    lines.append(f"- Filtered out: {manifest['filtered_out_count']}")
    source_runs = [str(item.get("run_id")) for item in manifest.get("source_runs", [])]
    if source_runs:
        lines.append(f"- Source runs: {', '.join(source_runs)}")
    selected_models = list(manifest.get("selected_models") or [])
    lines.append(
        f"- Selected models: {', '.join(selected_models) if selected_models else 'all exported models'}"
    )
    exported_models = list(manifest.get("exported_models") or [])
    lines.append(f"- Exported models: {', '.join(exported_models) if exported_models else '-'}")
    lines.append("")
    lines.append("## Dataset Card Summary")
    lines.append("")
    lines.append("This bundle contains successful artifact-bearing records exported from one or more WhitzardGen runs.")
    lines.append("")
    lines.append("Intended uses:")
    lines.append("")
    lines.append("- dataset inspection")
    lines.append("- downstream curation")
    lines.append("- model-by-model comparison")
    lines.append("- train/val style organization")
    lines.append("")
    lines.append("Important notes:")
    lines.append("")
    lines.append("- `dataset.jsonl` is the artifact-level export index")
    lines.append("- `export_manifest.json` is the structured source of truth for aggregate stats")
    lines.append("- `artifact_path` in exported records is bundle-relative")
    lines.append("- `source_artifact_path` preserves the original run artifact location")
    lines.append("- `source_run_lineage` preserves retry/resume lineage when available")
    lines.append("")
    lines.append("## Source Runs")
    lines.append("")
    if manifest.get("source_runs"):
        for source in manifest["source_runs"]:
            lines.append(
                f"- {source.get('run_id')}: "
                f"exported={source.get('exported_record_count', 0)}, "
                f"skipped={source.get('skipped_record_count', 0)}, "
                f"filtered={source.get('filtered_out_count', 0)}, "
                f"status={source.get('status', '-')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Counts By Model")
    lines.extend(_render_count_block(manifest.get("counts_by_model", {})))
    lines.append("")
    lines.append("## Counts By Split")
    lines.extend(_render_count_block(manifest.get("counts_by_split", {})))
    lines.append("")
    lines.append("## Counts By Artifact Type")
    lines.extend(_render_count_block(manifest.get("counts_by_artifact_type", {})))
    lines.append("")
    lines.append("## Counts By Run")
    lines.extend(_render_count_block(manifest.get("counts_by_run_id", {})))
    lines.append("")
    lines.append("## Bundle Layout")
    lines.append("")
    lines.append("```text")
    lines.append(f"{Path(manifest['bundle_path']).name}/")
    lines.append("  dataset.jsonl")
    lines.append("  export_manifest.json")
    lines.append("  README.md")
    lines.append("  media/")
    lines.append("    <split>/")
    lines.append("      <model_name>/")
    lines.append("        <artifact_type>/")
    lines.append("```")
    lines.append("")
    lines.append("## Recommended Follow-Up")
    lines.append("")
    lines.append("- inspect `dataset.jsonl` for record-level metadata")
    lines.append("- inspect `export_manifest.json` for aggregate stats and source-run context")
    lines.append("- use split/model directories under `media/` for downstream organization workflows")
    lines.append("")
    lines.append("This README is a human-oriented summary. `export_manifest.json` is the structured source of truth.")
    return "\n".join(lines) + "\n"


def _render_count_block(counts: dict[str, Any]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]
