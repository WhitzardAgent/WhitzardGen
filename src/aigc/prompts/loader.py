from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from aigc.prompts.models import PromptRecord, PromptValidationError, SUPPORTED_LANGUAGES

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_RESOLUTION_RE = re.compile(r"^\d+\s*[xX*]\s*\d+$")

_INT_PARAMETER_KEYS = {
    "width",
    "height",
    "seed",
    "num_inference_steps",
    "fps",
    "num_frames",
    "max_sequence_length",
    "cp_size",
    "context_parallel_size",
    "timeout_sec",
    "max_new_tokens",
}
_FLOAT_PARAMETER_KEYS = {
    "guidance_scale",
    "guidance_scale_2",
    "temperature",
    "top_p",
    "repetition_penalty",
}
_BOOL_PARAMETER_KEYS = {"stream", "do_sample", "enable_thinking"}
_STRING_PARAMETER_KEYS = {
    "attn_implementation",
    "moe_impl",
    "local_model_path",
    "checkpoint_dir",
    "repo_dir",
    "offload",
    "image_path",
    "ref_path",
    "resolution",
}
SUPPORTED_GENERATION_PARAMETER_KEYS = (
    _INT_PARAMETER_KEYS
    | _FLOAT_PARAMETER_KEYS
    | _BOOL_PARAMETER_KEYS
    | _STRING_PARAMETER_KEYS
)


def load_prompts(
    path: str | Path,
    *,
    warn: Callable[[str], None] | None = None,
) -> list[PromptRecord]:
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
    validate_prompts(records, prompt_source=prompt_path, warn=warn)
    return records


def validate_prompts(
    records: list[PromptRecord],
    *,
    prompt_source: str | Path | None = None,
    warn: Callable[[str], None] | None = None,
) -> None:
    seen_prompt_ids: set[str] = set()
    source_label = str(prompt_source) if prompt_source is not None else None
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
        record.parameters = validate_generation_parameters(
            record.parameters,
            owner_label=f"prompt_id={record.prompt_id}",
            prompt_source=source_label,
            warn=warn,
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


def validate_generation_parameters(
    params: dict[str, Any],
    *,
    owner_label: str,
    prompt_source: str | Path | None = None,
    warn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise PromptValidationError(f"{owner_label} parameters must be an object/dict.")

    normalized: dict[str, Any] = {}
    source_suffix = f" source={prompt_source}" if prompt_source is not None else ""
    for key, value in params.items():
        key_str = str(key)
        if key_str not in SUPPORTED_GENERATION_PARAMETER_KEYS:
            if warn is not None:
                warn(
                    f"[prompt] Unknown generation parameter key owner={owner_label} "
                    f"key={key_str}{source_suffix}"
                )
            normalized[key_str] = value
            continue

        try:
            normalized[key_str] = _normalize_generation_parameter_value(key_str, value)
        except PromptValidationError as exc:
            raise PromptValidationError(
                f"{owner_label} has invalid parameter {key_str}: {exc}"
            ) from exc
    return normalized


def _normalize_generation_parameter_value(key: str, value: Any) -> Any:
    if key in _INT_PARAMETER_KEYS:
        if isinstance(value, bool):
            raise PromptValidationError("expected integer, got boolean")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise PromptValidationError(f"expected integer, got {value!r}") from exc

    if key in _FLOAT_PARAMETER_KEYS:
        if isinstance(value, bool):
            raise PromptValidationError("expected number, got boolean")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise PromptValidationError(f"expected number, got {value!r}") from exc

    if key in _BOOL_PARAMETER_KEYS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        raise PromptValidationError(f"expected boolean, got {value!r}")

    if key == "resolution":
        if not isinstance(value, str):
            raise PromptValidationError(f"expected string resolution, got {value!r}")
        normalized = normalize_text(value)
        if not _RESOLUTION_RE.match(normalized):
            raise PromptValidationError(f"expected resolution like 1024x1024, got {value!r}")
        return normalized

    if key in _STRING_PARAMETER_KEYS:
        if value is None:
            raise PromptValidationError("expected non-null string")
        return str(value)

    return value


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
