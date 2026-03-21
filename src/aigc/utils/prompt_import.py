from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_IMPORT_CATEGORY = "aigc_safety"


def convert_legacy_prompt_csv_to_jsonl(
    *,
    csv_path: str | Path,
    jsonl_path: str | Path,
    category: str = DEFAULT_IMPORT_CATEGORY,
    forced_language: str | None = None,
) -> dict[str, Any]:
    source_path = Path(csv_path)
    output_path = Path(jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_forced_language = (
        _normalize_language(forced_language) if forced_language not in (None, "") else None
    )

    records: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue

            source_idx = str(row.get("idx", "")).strip()
            source_uuid = str(row.get("uuid", "")).strip()
            small_class = str(row.get("class_level_0", "")).strip()
            subcategory = str(row.get("class_level_1", "")).strip()
            theme = str(row.get("keyword", "")).strip() or prompt[:80]
            language = normalized_forced_language or _normalize_language(str(row.get("lang", "")).strip())
            model_type = str(row.get("model_type", "")).strip()

            prompt_id = source_uuid or (
                f"legacy_csv_{source_idx}" if source_idx else f"legacy_prompt_{row_index:06d}"
            )

            theme_path = [category]
            if subcategory:
                theme_path.append(subcategory)
            if small_class:
                theme_path.append(small_class)
            if theme:
                theme_path.append(theme)

            records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "language": language,
                    "metadata": {
                        "category": category,
                        "subcategory": subcategory or category,
                        "subtopic": small_class or subcategory or category,
                        "theme": theme,
                        "theme_path": theme_path,
                        "source_format": "legacy_prompt_csv",
                        "source_idx": source_idx or None,
                        "source_uuid": source_uuid or None,
                        "model_type": model_type or None,
                    },
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "csv_path": str(source_path),
        "jsonl_path": str(output_path),
        "record_count": len(records),
        "category": category,
        "forced_language": normalized_forced_language,
    }


def _normalize_language(raw: str) -> str:
    normalized = raw.lower()
    if normalized in {"zh", "zh-cn", "zh_cn", "cn", "chinese"}:
        return "zh"
    if normalized in {"en", "en-us", "en_us", "english"}:
        return "en"
    return normalized or "en"
