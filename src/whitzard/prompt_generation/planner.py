from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from whitzard.prompt_generation.models import SampledTheme, ThemeNode, ThemeTree


class ThemePlanningError(ValueError):
    """Raised when theme-tree planning cannot satisfy quotas."""


def build_sampling_plan(
    *,
    tree: ThemeTree,
    seed: int,
    count_config_path: str | Path | None = None,
) -> list[SampledTheme]:
    rng = random.Random(seed)
    count_overrides, level_defaults = _load_count_overrides(count_config_path)
    allocations: list[tuple[tuple[str, ...], int, list[_ThemeLeaf], bool]] = []

    for category in tree.categories:
        allocations.extend(
            _plan_node(
                node=category,
                path=(category.name,),
                count_overrides=count_overrides,
                level_defaults=level_defaults,
            )
        )

    sampled: list[SampledTheme] = []
    if not any(count > 0 for _quota_path, count, _leaves, _forced_resample in allocations):
        raise ThemePlanningError(
            "No prompt quotas were resolved. Add inline `count` fields or pass a count-config file."
        )
    next_index = 1
    for quota_path, count, leaves, forced_resample in allocations:
        if count <= 0:
            continue
        for draw_index in range(count):
            resampled = forced_resample or draw_index >= len(leaves)
            chosen_leaf = rng.choice(leaves) if resampled else leaves[draw_index % len(leaves)]
            category, subcategory, subtopic, theme = _expand_theme_path(chosen_leaf.path)
            sampled.append(
                SampledTheme(
                    sample_id=f"sample_{next_index:06d}",
                    quota_source_path=quota_path,
                    theme_path=chosen_leaf.path,
                    category=category,
                    subcategory=subcategory,
                    subtopic=subtopic,
                    theme=theme,
                    resampled=resampled,
                    metadata=dict(chosen_leaf.metadata),
                    constraints=dict(chosen_leaf.constraints),
                    tags=list(chosen_leaf.tags),
                )
            )
            next_index += 1
        if not forced_resample:
            rng.shuffle(leaves)
    return sampled


def summarize_sampling_plan(samples: list[SampledTheme]) -> dict[str, Any]:
    counts_by_category: dict[str, int] = defaultdict(int)
    counts_by_subcategory: dict[str, int] = defaultdict(int)
    counts_by_theme: dict[str, int] = defaultdict(int)
    resampled_count = 0
    for sample in samples:
        counts_by_category[sample.category] += 1
        counts_by_subcategory[" / ".join(sample.theme_path[:2])] += 1
        counts_by_theme[" / ".join(sample.theme_path)] += 1
        if sample.resampled:
            resampled_count += 1
    return {
        "sample_count": len(samples),
        "resampled_count": resampled_count,
        "counts_by_category": dict(sorted(counts_by_category.items())),
        "counts_by_subcategory": dict(sorted(counts_by_subcategory.items())),
        "counts_by_theme": dict(sorted(counts_by_theme.items())),
        "items": [sample.to_dict() for sample in samples],
    }


class _ThemeLeaf:
    def __init__(
        self,
        *,
        path: tuple[str, ...],
        metadata: dict[str, Any],
        constraints: dict[str, Any],
        tags: list[str],
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.constraints = constraints
        self.tags = tags


def _plan_node(
    *,
    node: ThemeNode,
    path: tuple[str, ...],
    count_overrides: dict[str, int],
    level_defaults: dict[str, int],
) -> list[tuple[tuple[str, ...], int, list[_ThemeLeaf], bool]]:
    child_allocations: list[tuple[tuple[str, ...], int, list[_ThemeLeaf], bool]] = []
    explicit_descendant_paths: set[tuple[str, ...]] = set()
    for child in node.children:
        child_path = (*path, child.name)
        child_allocations.extend(
            _plan_node(
                node=child,
                path=child_path,
                count_overrides=count_overrides,
                level_defaults=level_defaults,
            )
        )
        if _resolve_node_count(child_path, child.count, count_overrides, level_defaults) is not None:
            explicit_descendant_paths.add(child_path)
        explicit_descendant_paths.update(
            allocation_path for allocation_path, _count, _leaves, _forced in child_allocations
        )

    node_count = _resolve_node_count(path, node.count, count_overrides, level_defaults)
    if node_count is None:
        return child_allocations

    descendant_allocated = sum(
        count
        for allocation_path, count, _leaves, _forced in child_allocations
        if allocation_path[: len(path)] == path and allocation_path != path
    )
    residual = max(node_count - descendant_allocated, 0)

    residual_leaves = _collect_leaves(
        node=node,
        path=path,
        excluded_prefixes=explicit_descendant_paths,
        inherited_metadata={},
        inherited_constraints={},
        inherited_tags=[],
    )
    all_draws_resampled = False
    if residual > 0 and not residual_leaves:
        residual_leaves = _collect_leaves(
            node=node,
            path=path,
            excluded_prefixes=set(),
            inherited_metadata={},
            inherited_constraints={},
            inherited_tags=[],
        )
        all_draws_resampled = True

    allocations = list(child_allocations)
    if residual > 0:
        if not residual_leaves:
            raise ThemePlanningError(
                f"Unable to allocate residual quota for {'/'.join(path)} because no leaves exist."
            )
        allocations.append((path, residual, residual_leaves, all_draws_resampled))
    return allocations


def _collect_leaves(
    *,
    node: ThemeNode,
    path: tuple[str, ...],
    excluded_prefixes: set[tuple[str, ...]],
    inherited_metadata: dict[str, Any],
    inherited_constraints: dict[str, Any],
    inherited_tags: list[str],
) -> list[_ThemeLeaf]:
    merged_metadata = dict(inherited_metadata)
    merged_metadata.update(node.metadata)
    merged_constraints = dict(inherited_constraints)
    merged_constraints.update(node.constraints)
    merged_tags = list(dict.fromkeys([*inherited_tags, *node.tags]))
    if not node.children:
        return [
            _ThemeLeaf(
                path=path,
                metadata=merged_metadata,
                constraints=merged_constraints,
                tags=merged_tags,
            )
        ]
    leaves: list[_ThemeLeaf] = []
    for child in node.children:
        child_path = (*path, child.name)
        if any(child_path[: len(prefix)] == prefix for prefix in excluded_prefixes):
            continue
        leaves.extend(
            _collect_leaves(
                node=child,
                path=child_path,
                excluded_prefixes=excluded_prefixes,
                inherited_metadata=merged_metadata,
                inherited_constraints=merged_constraints,
                inherited_tags=merged_tags,
            )
        )
    return leaves


def _resolve_node_count(
    path: tuple[str, ...],
    inline_count: int | None,
    count_overrides: dict[str, int],
    level_defaults: dict[str, int],
) -> int | None:
    path_key = "/".join(path)
    if path_key in count_overrides:
        return count_overrides[path_key]
    if inline_count is not None:
        return inline_count
    level_name = _level_name_for_depth(len(path))
    if level_name and level_name in level_defaults:
        return level_defaults[level_name]
    return None


def _load_count_overrides(path: str | Path | None) -> tuple[dict[str, int], dict[str, int]]:
    if path is None:
        return {}, {}
    count_path = Path(path)
    if not count_path.exists():
        raise ThemePlanningError(f"Count-config file does not exist: {count_path}")
    raw = count_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise ThemePlanningError("PyYAML is required to load count-config files.") from exc
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}, {}
    if not isinstance(payload, dict):
        raise ThemePlanningError("Count-config file must contain an object.")
    counts_payload = (
        payload["counts"]
        if "counts" in payload
        else {key: value for key, value in payload.items() if key != "defaults"}
    )
    if not isinstance(counts_payload, dict):
        raise ThemePlanningError("Count-config counts payload must be an object.")
    normalized: dict[str, int] = {}
    for key, value in counts_payload.items():
        normalized[str(key)] = int(value)
    defaults_payload = payload.get("defaults", {})
    if defaults_payload in (None, ""):
        defaults_payload = {}
    if not isinstance(defaults_payload, dict):
        raise ThemePlanningError("Count-config defaults payload must be an object.")
    normalized_defaults: dict[str, int] = {}
    for key, value in defaults_payload.items():
        normalized_key = _normalize_level_default_key(str(key))
        if normalized_key is None:
            raise ThemePlanningError(
                f"Unsupported count-config default key: {key}. "
                "Use category, subcategory, subtopic, or theme."
            )
        normalized_defaults[normalized_key] = int(value)
    return normalized, normalized_defaults


def _expand_theme_path(path: tuple[str, ...]) -> tuple[str, str, str, str]:
    category = path[0]
    subcategory = path[1] if len(path) > 1 else category
    subtopic = path[-2] if len(path) > 2 else subcategory
    theme = path[-1]
    return category, subcategory, subtopic, theme


def _level_name_for_depth(depth: int) -> str | None:
    return {
        1: "category",
        2: "subcategory",
        3: "subtopic",
        4: "theme",
    }.get(depth)


def _normalize_level_default_key(raw_key: str) -> str | None:
    key = raw_key.strip().lower()
    aliases = {
        "category": "category",
        "categories": "category",
        "per_category": "category",
        "subcategory": "subcategory",
        "subcategories": "subcategory",
        "per_subcategory": "subcategory",
        "subtopic": "subtopic",
        "subtopics": "subtopic",
        "per_subtopic": "subtopic",
        "theme": "theme",
        "themes": "theme",
        "per_theme": "theme",
    }
    return aliases.get(key)
