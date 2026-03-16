from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

from aigc.prompts.models import PromptRecord, PromptValidationError, SUPPORTED_LANGUAGES

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def load_prompts(path: str | Path) -> list[PromptRecord]:
    prompt_path = Path(path)
    suffix = prompt_path.suffix.lower()
    if suffix == ".txt":
        records = _load_txt(prompt_path)
    elif suffix == ".csv":
        records = _load_csv(prompt_path)
    elif suffix == ".jsonl":
        records = _load_jsonl(prompt_path)
    else:
        raise PromptValidationError(
            f"Unsupported prompt file type: {prompt_path.suffix}. "
            "Supported types are .txt, .csv, and .jsonl."
        )
    validate_prompts(records)
    return records


def validate_prompts(records: list[PromptRecord]) -> None:
    seen_prompt_ids: set[str] = set()
    for record in records:
        if not record.prompt_id:
            raise PromptValidationError("Prompt record is missing prompt_id.")
        if record.prompt_id in seen_prompt_ids:
            raise PromptValidationError(f"Duplicate prompt_id detected: {record.prompt_id}")
        seen_prompt_ids.add(record.prompt_id)

        if not record.prompt:
            raise PromptValidationError(f"Prompt {record.prompt_id} has empty prompt text.")

        if record.language not in SUPPORTED_LANGUAGES:
            raise PromptValidationError(
                f"Prompt {record.prompt_id} uses unsupported language: {record.language}"
            )

        if record.negative_prompt is not None and not isinstance(record.negative_prompt, str):
            raise PromptValidationError(
                f"Prompt {record.prompt_id} has invalid negative_prompt type."
            )

        if not isinstance(record.parameters, dict):
            raise PromptValidationError(
                f"Prompt {record.prompt_id} parameters must be an object/dict."
            )

        if not isinstance(record.metadata, dict):
            raise PromptValidationError(
                f"Prompt {record.prompt_id} metadata must be an object/dict."
            )


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.strip()
    return _WHITESPACE_RE.sub(" ", normalized)


def infer_language(value: str) -> str:
    return "zh" if _CJK_RE.search(value) else "en"


def generate_prompt_id(index: int) -> str:
    return f"prompt_{index:06d}"


def _load_txt(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            prompt = normalize_text(line)
            if not prompt:
                continue
            records.append(
                PromptRecord(
                    prompt_id=generate_prompt_id(len(records) + 1),
                    prompt=prompt,
                    language=infer_language(prompt),
                )
            )
    return records


def _load_csv(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PromptValidationError(f"CSV file {path} is missing a header row.")
        if "prompt" not in reader.fieldnames:
            raise PromptValidationError(f"CSV file {path} must include a 'prompt' column.")

        for row in reader:
            raw_prompt = row.get("prompt", "")
            prompt = normalize_text(raw_prompt)
            if not prompt:
                continue
            prompt_id = normalize_text(row.get("prompt_id", "")) or generate_prompt_id(
                len(records) + 1
            )
            language = normalize_text(row.get("language", "")).lower() or infer_language(prompt)
            negative_prompt = normalize_text(row.get("negative_prompt", ""))
            parameters = _parse_json_object_cell(row.get("parameters", ""))
            metadata = _parse_json_object_cell(row.get("metadata", ""))
            version = normalize_text(row.get("version", "")) or None
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    language=language,
                    negative_prompt=negative_prompt or None,
                    parameters=parameters,
                    metadata=metadata,
                    version=version,
                )
            )
    return records


def _load_jsonl(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptValidationError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise PromptValidationError(
                    f"Line {line_number} of {path} must be a JSON object."
                )

            prompt_id = normalize_text(str(payload.get("prompt_id", "")))
            if not prompt_id:
                raise PromptValidationError(
                    f"Line {line_number} of {path} is missing required field 'prompt_id'."
                )

            prompt = normalize_text(str(payload.get("prompt", "")))
            language = normalize_text(str(payload.get("language", ""))).lower() or infer_language(
                prompt
            )
            negative_prompt_value = payload.get("negative_prompt")
            negative_prompt = None
            if negative_prompt_value is not None:
                negative_prompt = normalize_text(str(negative_prompt_value)) or None

            parameters = payload.get("parameters") or {}
            metadata = payload.get("metadata") or {}
            version_value = payload.get("version")
            version = normalize_text(str(version_value)) if version_value is not None else None

            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    language=language,
                    negative_prompt=negative_prompt,
                    parameters=parameters,
                    metadata=metadata,
                    version=version or None,
                )
            )
    return records


def _parse_json_object_cell(value: str) -> dict[str, object]:
    raw = normalize_text(value)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptValidationError(f"Expected JSON object cell, got invalid JSON: {raw}") from exc
    if not isinstance(parsed, dict):
        raise PromptValidationError(f"Expected JSON object cell, got: {raw}")
    return parsed
