from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from whitzard.benchmarking.models import (
    PreviewSummary,
    RequestPreviewBundle,
    RequestPreviewRecord,
)


class PreviewCollector:
    def __init__(
        self,
        *,
        enabled_stages: set[str] | None = None,
        source_context: dict[str, Any] | None = None,
    ) -> None:
        self.enabled_stages = set(enabled_stages or {"all"})
        self.source_context = dict(source_context or {})
        self.records: list[RequestPreviewRecord] = []

    def supports(self, stage: str) -> bool:
        return "all" in self.enabled_stages or stage in self.enabled_stages

    def collect(self, record: RequestPreviewRecord) -> None:
        if not self.supports(record.stage):
            return
        self.records.append(record)

    def to_bundle(self) -> RequestPreviewBundle:
        counts = Counter(record.stage for record in self.records)
        return RequestPreviewBundle(
            records=list(self.records),
            counts_by_stage=dict(sorted(counts.items())),
            source_context=dict(self.source_context),
        )

    def sample_records(self, count: int) -> list[RequestPreviewRecord]:
        limit = max(int(count), 0)
        if limit == 0:
            return []
        sampled: list[RequestPreviewRecord] = []
        counts: dict[str, int] = Counter()
        for record in self.records:
            if counts.get(record.stage, 0) >= limit:
                continue
            sampled.append(record)
            counts[record.stage] = counts.get(record.stage, 0) + 1
        return sampled


def parse_preview_stage(value: str | None, *, allowed_stages: set[str]) -> set[str]:
    normalized = str(value or "all").strip().lower()
    if normalized in {"", "all"}:
        return {"all"}
    parts = {
        item.strip().lower()
        for item in normalized.replace(",", "+").split("+")
        if item.strip()
    }
    return {item for item in parts if item in allowed_stages} or {"all"}


def write_request_preview_bundle(
    *,
    preview_dir: str | Path,
    bundle: RequestPreviewBundle,
    preview_only: bool,
    preview_stage: str,
    preview_count: int,
    render_markdown: bool = True,
) -> PreviewSummary:
    target = Path(preview_dir)
    target.mkdir(parents=True, exist_ok=True)
    request_previews_path = target / "request_previews.jsonl"
    request_preview_summary_path = target / "request_preview_summary.json"
    request_previews_markdown_path = target / "request_previews.md"

    with request_previews_path.open("w", encoding="utf-8") as handle:
        for record in bundle.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    sample_records = [item.to_dict() for item in _sample_records(bundle.records, preview_count)]
    summary_payload = {
        "preview_only": bool(preview_only),
        "preview_stage": str(preview_stage),
        "preview_count": int(preview_count),
        "counts_by_stage": dict(bundle.counts_by_stage),
        "source_context": dict(bundle.source_context),
        "sample_records": sample_records,
    }
    request_preview_summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown_path: str | None = None
    if render_markdown:
        request_previews_markdown_path.write_text(
            _render_request_previews_markdown(bundle=bundle, sample_records=sample_records),
            encoding="utf-8",
        )
        markdown_path = str(request_previews_markdown_path)

    return PreviewSummary(
        preview_dir=str(target),
        request_previews_path=str(request_previews_path),
        request_preview_summary_path=str(request_preview_summary_path),
        request_previews_markdown_path=markdown_path,
        preview_only=bool(preview_only),
        preview_stage=str(preview_stage),
        preview_count=int(preview_count),
        counts_by_stage=dict(bundle.counts_by_stage),
        sample_records=sample_records,
        source_context=dict(bundle.source_context),
    )


def _sample_records(
    records: list[RequestPreviewRecord],
    preview_count: int,
) -> list[RequestPreviewRecord]:
    limit = max(int(preview_count), 0)
    if limit == 0:
        return []
    sampled: list[RequestPreviewRecord] = []
    counts_by_stage: dict[str, int] = {}
    for record in records:
        if counts_by_stage.get(record.stage, 0) >= limit:
            continue
        sampled.append(record)
        counts_by_stage[record.stage] = counts_by_stage.get(record.stage, 0) + 1
    return sampled


def _render_request_previews_markdown(
    *,
    bundle: RequestPreviewBundle,
    sample_records: list[dict[str, Any]],
) -> str:
    lines = ["# Request Previews", ""]
    if bundle.source_context:
        lines.append("## Source Context")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(bundle.source_context, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    counts = dict(bundle.counts_by_stage)
    if counts:
        lines.append("## Counts By Stage")
        lines.append("")
        for stage, count in sorted(counts.items()):
            lines.append(f"- `{stage}`: {count}")
        lines.append("")
    lines.append("## Sample Records")
    lines.append("")
    if not sample_records:
        lines.append("_No preview records available._")
        lines.append("")
        return "\n".join(lines)
    for index, record in enumerate(sample_records, start=1):
        lines.append(f"### {index}. {record.get('stage', 'unknown')}")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key != "rendered_prompt"
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("```text")
        lines.append(str(record.get("rendered_prompt", "")))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
