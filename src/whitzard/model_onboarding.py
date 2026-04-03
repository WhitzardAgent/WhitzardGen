from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whitzard.registry import load_registry
from whitzard.registry.models import ModelInfo
from whitzard.run_flow import RunSummary, run_single_model
from whitzard.settings import get_runs_root
from whitzard.utils.progress import RunProgress

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPABILITY_MATRIX_MARKDOWN_PATH = REPO_ROOT / "docs" / "model_capability_matrix.md"
DEFAULT_CAPABILITY_MATRIX_JSON_PATH = REPO_ROOT / "docs" / "model_capability_matrix.json"

LOCAL_DEPLOYMENT_FIELDS = (
    "local_path",
    "weights_path",
    "repo_path",
    "script_root",
    "hf_cache_dir",
)


def default_canary_prompt_file(model: ModelInfo) -> Path:
    prompt_name = {
        "image": "canary_image.jsonl",
        "video": "canary_video.jsonl",
        "text": "canary_text.jsonl",
    }.get(model.modality)
    if prompt_name is None:
        raise ValueError(f"No default canary prompt file is defined for modality {model.modality}.")
    return REPO_ROOT / "prompts" / prompt_name


def run_model_canary(
    *,
    model_name: str,
    prompt_file: str | Path | None = None,
    out_dir: str | Path | None = None,
    execution_mode: str = "real",
    progress: RunProgress | None = None,
) -> RunSummary:
    registry = load_registry()
    model = registry.get_model(model_name)
    resolved_prompt_file = Path(prompt_file) if prompt_file is not None else default_canary_prompt_file(model)
    run_name = f"canary-{_slugify(model_name)}"
    resolved_out_dir = Path(out_dir) if out_dir is not None else get_runs_root() / run_name
    return run_single_model(
        model_name=model_name,
        prompt_file=resolved_prompt_file,
        out_dir=resolved_out_dir,
        run_name=run_name,
        execution_mode=execution_mode,
        progress=progress,
    )


def build_model_capability_rows() -> list[dict[str, Any]]:
    registry = load_registry()
    rows: list[dict[str, Any]] = []
    for model in registry.list_models():
        adapter_class = registry.resolve_adapter_class(model.name)
        adapter_capabilities = getattr(adapter_class, "capabilities", None)
        local_fields = [
            field
            for field in LOCAL_DEPLOYMENT_FIELDS
            if model.weights.get(field) not in (None, "") or model.local_paths.get(field) not in (None, "")
        ]
        rows.append(
            {
                "name": model.name,
                "version": model.version,
                "modality": model.modality,
                "task_type": model.task_type,
                "adapter": model.adapter,
                "execution_mode": model.execution_mode,
                "gpu_required": bool(model.gpu_required),
                "worker_strategy": model.worker_strategy,
                "provider_type": str(model.provider.get("type", "")) or None,
                "supports_persistent_worker": bool(
                    getattr(adapter_capabilities, "supports_persistent_worker", False)
                ),
                "supports_batch_prompts": bool(model.capabilities.get("supports_batch_prompts", False)),
                "max_batch_size": int(model.capabilities.get("max_batch_size", 1)),
                "supports_multi_replica": bool(model.supports_multi_replica),
                "supports_seed": bool(model.capabilities.get("supports_seed", True)),
                "supports_negative_prompt": bool(
                    model.capabilities.get("supports_negative_prompt", False)
                ),
                "conda_env_name": model.conda_env_name,
                "key_local_fields": local_fields,
                "generation_defaults": dict(model.generation_defaults),
            }
        )
    return rows


def render_model_capability_matrix_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Model Capability Matrix",
        "",
        "Generated from the current registry configuration.",
        "",
        "| Model | Modality | Task | Execution | GPU Required | Worker | Provider | Batch | Multi-Replica | Seed | Negative | Conda Env | Local Fields |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["modality"]),
                    str(row["task_type"]),
                    str(row["execution_mode"]),
                    "yes" if row["gpu_required"] else "no",
                    str(row["worker_strategy"]),
                    str(row["provider_type"] or "-"),
                    f"{'yes' if row['supports_batch_prompts'] else 'no'}"
                    + (f" ({row['max_batch_size']})" if row["supports_batch_prompts"] else ""),
                    "yes" if row["supports_multi_replica"] else "no",
                    "yes" if row["supports_seed"] else "no",
                    "yes" if row["supports_negative_prompt"] else "no",
                    str(row["conda_env_name"]),
                    ", ".join(row["key_local_fields"]) if row["key_local_fields"] else "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_model_capability_docs(
    *,
    markdown_path: str | Path = DEFAULT_CAPABILITY_MATRIX_MARKDOWN_PATH,
    json_path: str | Path = DEFAULT_CAPABILITY_MATRIX_JSON_PATH,
) -> dict[str, str]:
    rows = build_model_capability_rows()
    markdown_target = Path(markdown_path)
    json_target = Path(json_path)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.write_text(render_model_capability_matrix_markdown(rows), encoding="utf-8")
    json_target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "markdown_path": str(markdown_target),
        "json_path": str(json_target),
    }


def _slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-")
