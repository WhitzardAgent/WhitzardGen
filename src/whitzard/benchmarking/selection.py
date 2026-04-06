from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from whitzard.benchmarking.models import BenchmarkCase, CaseSelectionResult, CaseSelectionSpec, CaseSet


class CaseSelectionError(RuntimeError):
    """Raised when case selection configuration is invalid."""


def normalize_case_selection_spec(payload: dict[str, Any] | None) -> CaseSelectionSpec | None:
    if not payload:
        return None
    if "case_selection" in payload and isinstance(payload.get("case_selection"), dict):
        payload = dict(payload.get("case_selection") or {})
    spec = CaseSelectionSpec(
        seed=int(payload.get("seed", 42)),
        group_selector=_optional_text(payload.get("group_selector")),
        sample_size_per_group=_optional_int(payload.get("sample_size_per_group")),
        undersized_group_policy=str(payload.get("undersized_group_policy", "keep_all_warn") or "keep_all_warn"),
        include_groups=_normalize_text_list(payload.get("include_groups")),
        exclude_groups=_normalize_text_list(payload.get("exclude_groups")),
        include_case_ids=_normalize_text_list(payload.get("include_case_ids")),
        exclude_case_ids=_normalize_text_list(payload.get("exclude_case_ids")),
        split_filter=_normalize_text_list(payload.get("split_filter")),
        tag_filter=_normalize_text_list(payload.get("tag_filter")),
        max_cases=_optional_int(payload.get("max_cases")),
    )
    validate_case_selection_spec(spec)
    return spec


def validate_case_selection_spec(spec: CaseSelectionSpec) -> None:
    if spec.sample_size_per_group is not None and spec.sample_size_per_group <= 0:
        raise CaseSelectionError("case_selection.sample_size_per_group must be greater than 0.")
    if spec.max_cases is not None and spec.max_cases <= 0:
        raise CaseSelectionError("case_selection.max_cases must be greater than 0.")
    if spec.undersized_group_policy not in {"keep_all_warn", "error", "drop_group"}:
        raise CaseSelectionError(
            "case_selection.undersized_group_policy must be one of: keep_all_warn, error, drop_group."
        )


def apply_case_selection(*, case_set: CaseSet, spec: CaseSelectionSpec) -> CaseSelectionResult:
    validate_case_selection_spec(spec)
    original_cases = list(case_set.cases)
    grouped_before = _count_by_group(original_cases, spec.group_selector)
    warnings: list[str] = []

    filtered_cases = _apply_pre_group_filters(original_cases, spec)
    groups = _group_cases(filtered_cases, spec.group_selector)
    selected_ids: set[str] = set()
    undersized_groups: list[dict[str, Any]] = []
    rng = random.Random(spec.seed)

    for group_key in sorted(groups):
        cases = list(groups[group_key])
        if spec.sample_size_per_group is None:
            sampled = cases
        else:
            if len(cases) < spec.sample_size_per_group:
                record = {
                    "group_key": group_key,
                    "requested_count": spec.sample_size_per_group,
                    "actual_count": len(cases),
                }
                undersized_groups.append(record)
                if spec.undersized_group_policy == "error":
                    raise CaseSelectionError(
                        f"Group {group_key} only has {len(cases)} cases, requested {spec.sample_size_per_group}."
                    )
                if spec.undersized_group_policy == "drop_group":
                    warnings.append(
                        f"Dropped undersized group {group_key}: requested {spec.sample_size_per_group}, found {len(cases)}."
                    )
                    continue
                warnings.append(
                    f"Keeping all cases for undersized group {group_key}: requested {spec.sample_size_per_group}, found {len(cases)}."
                )
                sampled = cases
            else:
                shuffled = list(cases)
                rng.shuffle(shuffled)
                sampled = shuffled[: spec.sample_size_per_group]
        for case in sampled:
            selected_ids.add(case.case_id)

    selected_cases = [case for case in filtered_cases if case.case_id in selected_ids]
    if spec.max_cases is not None and len(selected_cases) > spec.max_cases:
        shuffled_ids = [case.case_id for case in selected_cases]
        rng.shuffle(shuffled_ids)
        keep_ids = set(shuffled_ids[: spec.max_cases])
        warnings.append(
            f"Applied max_cases={spec.max_cases} after grouping; retained {len(keep_ids)} selected cases."
        )
        selected_cases = [case for case in selected_cases if case.case_id in keep_ids]
        selected_ids = {case.case_id for case in selected_cases}

    excluded_cases = [case for case in original_cases if case.case_id not in selected_ids]
    grouped_after = _count_by_group(selected_cases, spec.group_selector)
    selection_manifest = {
        "selection_applied": True,
        "seed": spec.seed,
        "group_selector": spec.group_selector or "all",
        "sample_size_per_group": spec.sample_size_per_group,
        "undersized_group_policy": spec.undersized_group_policy,
        "include_groups": list(spec.include_groups),
        "exclude_groups": list(spec.exclude_groups),
        "include_case_ids": list(spec.include_case_ids),
        "exclude_case_ids": list(spec.exclude_case_ids),
        "split_filter": list(spec.split_filter),
        "tag_filter": list(spec.tag_filter),
        "max_cases": spec.max_cases,
        "counts_before": len(original_cases),
        "counts_after": len(selected_cases),
        "counts_by_group_before": grouped_before,
        "counts_by_group_after": grouped_after,
        "undersized_groups": list(undersized_groups),
        "selected_case_ids": [case.case_id for case in selected_cases],
        "excluded_case_ids": [case.case_id for case in excluded_cases],
        "warnings": list(warnings),
    }
    return CaseSelectionResult(
        spec=spec,
        selected_cases=selected_cases,
        excluded_cases=excluded_cases,
        counts_before=len(original_cases),
        counts_after=len(selected_cases),
        counts_by_group_before=grouped_before,
        counts_by_group_after=grouped_after,
        undersized_groups=undersized_groups,
        warnings=warnings,
        selection_manifest=selection_manifest,
    )


def clone_case_set_with_selection(*, case_set: CaseSet, selection_result: CaseSelectionResult) -> CaseSet:
    from whitzard.benchmarking.bundle import build_benchmark_stats

    manifest = dict(case_set.manifest)
    manifest["selection_applied"] = True
    manifest["selection_spec"] = selection_result.spec.to_dict()
    manifest["source_case_count"] = selection_result.counts_before
    manifest["selected_case_count"] = selection_result.counts_after
    manifest["excluded_case_count"] = len(selection_result.excluded_cases)
    stats = build_benchmark_stats(selection_result.selected_cases)
    stats["counts_by_group_before"] = dict(selection_result.counts_by_group_before)
    stats["counts_by_group_after"] = dict(selection_result.counts_by_group_after)
    return CaseSet(
        benchmark_id=case_set.benchmark_id,
        cases=list(selection_result.selected_cases),
        source=case_set.source,
        manifest=manifest,
        stats=stats,
        case_set_path=case_set.case_set_path,
    )


def infer_case_id_prefix(case_id: str) -> str:
    match = re.match(r"^(?P<prefix>.+)_\d+$", str(case_id))
    if match:
        return str(match.group("prefix"))
    return str(case_id)


def load_case_selection_config(path: str | Path) -> CaseSelectionSpec | None:
    from whitzard.benchmarking.service import load_yaml_file

    return normalize_case_selection_spec(load_yaml_file(Path(path)))


def _apply_pre_group_filters(cases: list[BenchmarkCase], spec: CaseSelectionSpec) -> list[BenchmarkCase]:
    include_case_ids = set(spec.include_case_ids)
    exclude_case_ids = set(spec.exclude_case_ids)
    split_filter = set(spec.split_filter)
    tag_filter = set(spec.tag_filter)
    include_groups = set(spec.include_groups)
    exclude_groups = set(spec.exclude_groups)
    filtered: list[BenchmarkCase] = []
    for case in cases:
        if include_case_ids and case.case_id not in include_case_ids:
            continue
        if exclude_case_ids and case.case_id in exclude_case_ids:
            continue
        if split_filter and case.split not in split_filter:
            continue
        if tag_filter and not tag_filter.intersection(case.tags):
            continue
        group_value = _resolve_group_value(case, spec.group_selector)
        if include_groups and group_value not in include_groups:
            continue
        if exclude_groups and group_value in exclude_groups:
            continue
        filtered.append(case)
    return filtered


def _group_cases(cases: list[BenchmarkCase], selector: str | None) -> dict[str, list[BenchmarkCase]]:
    grouped: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        group_key = _resolve_group_value(case, selector)
        grouped.setdefault(group_key, []).append(case)
    return grouped


def _count_by_group(cases: list[BenchmarkCase], selector: str | None) -> dict[str, int]:
    counter = Counter(_resolve_group_value(case, selector) for case in cases)
    return dict(sorted(counter.items()))


def _resolve_group_value(case: BenchmarkCase, selector: str | None) -> str:
    if selector in (None, "", "all"):
        return "all"
    if selector == "case_id":
        return str(case.case_id)
    if selector == "case_id_prefix":
        return infer_case_id_prefix(case.case_id)
    if selector == "split":
        return str(case.split)
    if selector.startswith("metadata."):
        value = case.metadata.get(selector.split(".", 1)[1])
        return _normalize_group_value(value)
    if selector.startswith("grouping."):
        value = case.grouping.get(selector.split(".", 1)[1])
        return _normalize_group_value(value)
    if selector.startswith("execution_hints."):
        value = case.execution_hints.get(selector.split(".", 1)[1])
        return _normalize_group_value(value)
    if selector.startswith("evaluation_hints."):
        value = case.evaluation_hints.get(selector.split(".", 1)[1])
        return _normalize_group_value(value)
    if selector == "metadata":
        return _normalize_group_value(case.metadata)
    raise CaseSelectionError(f"Unsupported case_selection.group_selector: {selector}")


def _normalize_group_value(value: Any) -> str:
    if value in (None, ""):
        return "ungrouped"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return str(value)


def _normalize_text_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    raise CaseSelectionError("Expected a string or list of strings in case selection config.")


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
