from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_annotation_bundle(
    *,
    bundle_dir: str | Path,
    annotations: list[dict[str, Any]],
    manifest: dict[str, Any],
    annotation_log: list[dict[str, Any]],
    stats: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, str]:
    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)
    annotations_path = target / "annotations.jsonl"
    manifest_path = target / "annotation_manifest.json"
    log_path = target / "annotation_log.jsonl"
    stats_path = target / "stats.json"
    failures_path = target / "failures.json"

    with annotations_path.open("w", encoding="utf-8") as handle:
        for record in annotations:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as handle:
        for event in annotation_log:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "annotations_path": str(annotations_path),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "stats_path": str(stats_path),
        "failures_path": str(failures_path),
    }


def load_annotation_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def build_annotation_stats(
    annotations: list[dict[str, Any]],
    *,
    failures: list[dict[str, Any]],
    skipped_count: int,
) -> dict[str, Any]:
    counts_by_artifact_type = Counter()
    counts_by_source_model = Counter()
    counts_by_annotator_model = Counter()
    for record in annotations:
        counts_by_artifact_type[str(record.get("source_artifact_type", ""))] += 1
        counts_by_source_model[str(record.get("source_model_name", ""))] += 1
        counts_by_annotator_model[str(record.get("annotator_model", ""))] += 1
    return {
        "annotation_count": len(annotations),
        "failed_count": len(failures),
        "skipped_count": skipped_count,
        "counts_by_artifact_type": dict(sorted(counts_by_artifact_type.items())),
        "counts_by_source_model": dict(sorted(counts_by_source_model.items())),
        "counts_by_annotator_model": dict(sorted(counts_by_annotator_model.items())),
    }
