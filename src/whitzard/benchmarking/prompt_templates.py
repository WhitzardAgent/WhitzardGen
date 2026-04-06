from __future__ import annotations

from typing import Any

from whitzard.structured_io import (
    build_allowed_context,
    default_judge_template_context,
    default_target_template_context,
    format_structured_choices,
    render_output_contract_block,
    render_template_spec,
    resolve_template_spec,
)


def resolve_prompt_template_config(
    config: dict[str, Any] | None,
    *,
    base_dir: str | None = None,
) -> dict[str, Any]:
    if not config:
        return {}
    return resolve_template_spec(config, base_dir=base_dir).to_dict()


def render_scoped_prompt_template(
    *,
    template_config: dict[str, Any],
    root_context: dict[str, Any],
    warning_prefix: str,
) -> tuple[str, list[str]]:
    return render_template_spec(
        template_spec=resolve_template_spec(template_config),
        root_context=root_context,
        warning_prefix=warning_prefix,
    )


__all__ = [
    "build_allowed_context",
    "default_judge_template_context",
    "default_target_template_context",
    "format_structured_choices",
    "render_output_contract_block",
    "render_scoped_prompt_template",
    "resolve_prompt_template_config",
]
