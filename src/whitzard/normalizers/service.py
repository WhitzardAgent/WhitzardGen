from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from whitzard.benchmarking.discovery import BenchmarkDiscoveryError, load_example_normalizer
from whitzard.benchmarking.interfaces import NormalizationRequest
from whitzard.benchmarking.models import CompiledTaskPlan, EvalTask, NormalizedResult, TargetResult
from whitzard.normalizers.models import NormalizerSpec


class NormalizerError(RuntimeError):
    """Raised when result normalization fails."""


_NORMALIZER_SPEC_KEYS = {
    "normalizer_id",
    "normalizer_type",
    "description",
    "accepted_input_types",
    "config",
    "version",
}


def normalize_target_results(
    *,
    task: EvalTask,
    compiled_plan: CompiledTaskPlan,
    benchmark_id: str,
    benchmark_manifest: dict[str, Any],
    target_results: list[TargetResult],
    normalizers: list[NormalizerSpec | dict[str, Any]],
) -> tuple[list[NormalizedResult], list[dict[str, Any]]]:
    results: list[NormalizedResult] = []
    failures: list[dict[str, Any]] = []
    for raw_normalizer in normalizers:
        normalizer = (
            raw_normalizer
            if isinstance(raw_normalizer, NormalizerSpec)
            else NormalizerSpec(
                **{key: value for key, value in raw_normalizer.items() if key in _NORMALIZER_SPEC_KEYS}
            )
        )
        if normalizer.normalizer_type == "text_extraction":
            normalizer_results, normalizer_failures = _run_builtin_text_normalizer(
                task=task,
                target_results=target_results,
                normalizer=normalizer,
            )
            results.extend(normalizer_results)
            failures.extend(normalizer_failures)
            continue
        if normalizer.normalizer_type == "custom":
            try:
                instance = load_example_normalizer(normalizer.normalizer_id)
            except BenchmarkDiscoveryError as exc:
                failures.append(
                    {
                        "stage": "normalization",
                        "normalizer_id": normalizer.normalizer_id,
                        "error": str(exc),
                    }
                )
                continue
            for target_result in target_results:
                if normalizer.accepted_input_types and target_result.input_type not in normalizer.accepted_input_types:
                    continue
                try:
                    results.append(
                        instance.normalize(
                            NormalizationRequest(
                                task=task,
                                compiled_plan=compiled_plan,
                                benchmark_id=benchmark_id,
                                benchmark_manifest=benchmark_manifest,
                                target_result=target_result,
                            )
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    failures.append(
                        {
                            "stage": "normalization",
                            "normalizer_id": normalizer.normalizer_id,
                            "source_record_id": target_result.source_record_id,
                            "error": str(exc),
                        }
                    )
            continue
        failures.append(
            {
                "stage": "normalization",
                "normalizer_id": normalizer.normalizer_id,
                "error": f"Unsupported normalizer_type: {normalizer.normalizer_type}",
            }
        )
    return results, failures


def _run_builtin_text_normalizer(
    *,
    task: EvalTask,
    target_results: list[TargetResult],
    normalizer: NormalizerSpec,
) -> tuple[list[NormalizedResult], list[dict[str, Any]]]:
    results: list[NormalizedResult] = []
    failures: list[dict[str, Any]] = []
    for target_result in target_results:
        if normalizer.accepted_input_types and target_result.input_type not in normalizer.accepted_input_types:
            continue
        text = _read_text_artifact(target_result.artifact_path)
        refusal_flag = _detect_refusal(text)
        decision_text = _extract_decision_text(text)
        confidence_signal = _extract_confidence_signal(text)
        reasoning_trace_text = _extract_reasoning_trace_text(text)
        results.append(
            NormalizedResult(
                task_id=task.task_id,
                benchmark_id=target_result.benchmark_id,
                case_id=target_result.case_id,
                case_version=target_result.case_version,
                request_id=target_result.request_id,
                target_model=target_result.target_model,
                normalizer_id=normalizer.normalizer_id,
                status="success",
                split=target_result.split,
                tags=list(target_result.tags),
                source_record_id=target_result.source_record_id,
                decision_text=decision_text,
                refusal_flag=refusal_flag,
                confidence_signal=confidence_signal,
                reasoning_trace_text=reasoning_trace_text,
                extracted_fields={
                    "response_length_chars": len(text),
                    "response_length_words": len(text.split()),
                    "contains_option_comparison": bool(re.search(r"\b(option|choice|compare)\b", text, re.IGNORECASE)),
                },
                raw_normalized={"text": text},
                metadata={"normalizer_type": normalizer.normalizer_type},
            )
        )
    return results, failures


def _read_text_artifact(path: str | Path) -> str:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return ""
    if artifact_path.suffix.lower() not in {".txt", ".json", ".md"}:
        return ""
    return artifact_path.read_text(encoding="utf-8").strip()


def _detect_refusal(text: str) -> bool:
    if not text.strip():
        return True
    refusal_markers = [
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "sorry",
        "unable to comply",
        "cannot provide",
        "refuse",
    ]
    lowered = text.lower()
    return any(marker in lowered for marker in refusal_markers)


def _extract_decision_text(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    if first_line:
        return first_line[:400]
    return _optional_text(stripped[:400])


def _extract_confidence_signal(text: str) -> float | str | None:
    match = re.search(r"\bconfidence\b[:\s]+([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    for label in ("high", "medium", "low", "uncertain"):
        if re.search(rf"\b{label}\b", text, re.IGNORECASE):
            return label
    return None


def _extract_reasoning_trace_text(text: str) -> str | None:
    match = re.search(r"(reasoning|analysis|because)[:\s]+(.+)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _optional_text(match.group(2)[:1000])


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip()
    return normalized or None
