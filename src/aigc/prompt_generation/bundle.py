from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from aigc.prompts import PromptRecord, load_prompts


def inspect_prompt_bundle(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_dir():
        manifest_path = target / "prompt_manifest.json"
        prompts_path = target / "prompts.jsonl"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        prompts = load_prompts(prompts_path) if prompts_path.exists() else []
        return _build_prompt_inspect_payload(
            prompts=prompts,
            bundle_dir=target,
            manifest=manifest,
        )
    prompts = load_prompts(target)
    return _build_prompt_inspect_payload(prompts=prompts, bundle_dir=None, manifest={})


def write_prompt_bundle(
    *,
    bundle_dir: str | Path,
    prompts: list[PromptRecord],
    manifest: dict[str, Any],
    sampling_plan: dict[str, Any],
    generation_log: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, str]:
    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)
    prompts_path = target / "prompts.jsonl"
    manifest_path = target / "prompt_manifest.json"
    sampling_plan_path = target / "sampling_plan.json"
    generation_log_path = target / "generation_log.jsonl"
    stats_path = target / "stats.json"

    with prompts_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "negative_prompt": prompt.negative_prompt,
                        "language": prompt.language,
                        "parameters": prompt.parameters,
                        "metadata": prompt.metadata,
                        "version": prompt.version,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    sampling_plan_path.write_text(
        json.dumps(sampling_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with generation_log_path.open("w", encoding="utf-8") as handle:
        for event in generation_log:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "prompts_path": str(prompts_path),
        "manifest_path": str(manifest_path),
        "sampling_plan_path": str(sampling_plan_path),
        "generation_log_path": str(generation_log_path),
        "stats_path": str(stats_path),
    }


def build_prompt_stats(prompts: list[PromptRecord]) -> dict[str, Any]:
    category_counts = Counter()
    subcategory_counts = Counter()
    theme_counts = Counter()
    for prompt in prompts:
        metadata = dict(prompt.metadata)
        category_counts[str(metadata.get("category", ""))] += 1
        subcategory_counts[str(metadata.get("subcategory", ""))] += 1
        theme_counts[str(metadata.get("theme", ""))] += 1
    return {
        "prompt_count": len(prompts),
        "counts_by_category": dict(sorted(category_counts.items())),
        "counts_by_subcategory": dict(sorted(subcategory_counts.items())),
        "counts_by_theme": dict(sorted(theme_counts.items())),
    }


def _build_prompt_inspect_payload(
    *,
    prompts: list[PromptRecord],
    bundle_dir: Path | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    stats = build_prompt_stats(prompts)
    return {
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
        "manifest": manifest,
        "prompt_count": len(prompts),
        "counts_by_category": stats["counts_by_category"],
        "counts_by_subcategory": stats["counts_by_subcategory"],
        "counts_by_theme": stats["counts_by_theme"],
        "sample_prompt_ids": [prompt.prompt_id for prompt in prompts[:5]],
        "sample_few_shot_example_ids": [
            dict(prompt.metadata).get("few_shot_example_ids", [])
            for prompt in prompts[:3]
        ],
    }
