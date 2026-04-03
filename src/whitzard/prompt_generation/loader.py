from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.prompt_generation.models import ThemeNode, ThemeTree


class ThemeTreeError(ValueError):
    """Raised when a theme tree is invalid."""


def load_theme_tree(path: str | Path) -> ThemeTree:
    tree_path = Path(path)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise ThemeTreeError("PyYAML is required to load theme-tree YAML files.") from exc
    payload = yaml.safe_load(tree_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ThemeTreeError(f"Theme tree {tree_path} must contain a top-level object.")
    categories_payload = payload.get("categories")
    if not isinstance(categories_payload, list) or not categories_payload:
        raise ThemeTreeError(f"Theme tree {tree_path} must define a non-empty categories list.")
    categories = [_parse_theme_node(item, label=f"categories[{index}]") for index, item in enumerate(categories_payload)]
    tree = ThemeTree(
        version=str(payload.get("version", "v1")),
        name=str(payload.get("name") or tree_path.stem),
        defaults=dict(payload.get("defaults", {})),
        categories=categories,
    )
    _validate_tree(tree)
    return tree


def _parse_theme_node(payload: Any, *, label: str) -> ThemeNode:
    if not isinstance(payload, dict):
        raise ThemeTreeError(f"{label} must be an object.")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ThemeTreeError(f"{label} is missing required field 'name'.")
    count = payload.get("count")
    if count in (None, ""):
        parsed_count = None
    else:
        parsed_count = int(count)
        if parsed_count <= 0:
            raise ThemeTreeError(f"{label}.count must be a positive integer.")
    metadata = payload.get("metadata") or {}
    constraints = payload.get("constraints") or {}
    tags = payload.get("tags") or []
    children_payload = payload.get("children") or []
    if not isinstance(metadata, dict):
        raise ThemeTreeError(f"{label}.metadata must be an object.")
    if not isinstance(constraints, dict):
        raise ThemeTreeError(f"{label}.constraints must be an object.")
    if not isinstance(tags, list):
        raise ThemeTreeError(f"{label}.tags must be a list.")
    if not isinstance(children_payload, list):
        raise ThemeTreeError(f"{label}.children must be a list.")
    children = [
        _parse_theme_node(item, label=f"{label}.children[{index}]")
        for index, item in enumerate(children_payload)
    ]
    return ThemeNode(
        name=name,
        count=parsed_count,
        metadata=dict(metadata),
        constraints=dict(constraints),
        tags=[str(tag) for tag in tags],
        children=children,
    )


def _validate_tree(tree: ThemeTree) -> None:
    def walk(node: ThemeNode, *, path: tuple[str, ...], seen: set[tuple[str, ...]]) -> None:
        node_path = (*path, node.name)
        if node_path in seen:
            raise ThemeTreeError(f"Duplicate node path detected: {'/'.join(node_path)}")
        seen.add(node_path)
        for child in node.children:
            walk(child, path=node_path, seen=seen)

    seen: set[tuple[str, ...]] = set()
    for category in tree.categories:
        walk(category, path=(), seen=seen)
