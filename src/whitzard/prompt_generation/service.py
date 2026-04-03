from __future__ import annotations

import json
import random
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from whitzard.prompt_generation.bundle import (
    build_prompt_stats,
    inspect_prompt_bundle,
    write_prompt_bundle,
)
from whitzard.prompt_generation.config import (
    DEFAULT_STYLE_FAMILY_NAME,
    DEFAULT_TEMPLATE_NAME,
    load_prompt_generation_catalog,
    render_instruction_template,
)
from whitzard.prompt_generation.loader import ThemeTreeError, load_theme_tree
from whitzard.prompt_generation.models import (
    PromptGenerationSummary,
    PromptStyleFamilyConfig,
    PromptTemplateConfig,
    SampledTheme,
    ThemeTree,
)
from whitzard.prompt_generation.planner import build_sampling_plan, summarize_sampling_plan
from whitzard.prompts import PromptRecord, normalize_text
from whitzard.run_flow import run_single_model
from whitzard.settings import get_prompt_runs_root
from whitzard.utils.progress import NullRunProgress, RunProgress


class PromptGenerationError(RuntimeError):
    """Raised when prompt generation fails."""


def plan_theme_tree(
    *,
    tree_path: str | Path,
    seed: int,
    count_config_path: str | Path | None = None,
) -> dict[str, Any]:
    tree = load_theme_tree(tree_path)
    samples = build_sampling_plan(
        tree=tree,
        seed=seed,
        count_config_path=count_config_path,
    )
    summary = summarize_sampling_plan(samples)
    summary["tree"] = tree.to_dict()
    summary["seed"] = seed
    return summary


def generate_prompt_bundle(
    *,
    tree_path: str | Path,
    out_dir: str | Path | None = None,
    llm_model: str | None = None,
    execution_mode: str = "real",
    seed: int = 42,
    count_config_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    template_name: str | None = None,
    style_family_name: str | None = None,
    target_model_name: str | None = None,
    intended_modality: str | None = None,
    progress: RunProgress | None = None,
) -> PromptGenerationSummary:
    tree = load_theme_tree(tree_path)
    bundle_id = _generate_bundle_id(tree.name)
    bundle_dir = Path(out_dir) if out_dir is not None else get_prompt_runs_root() / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or NullRunProgress()
    progress.env_message(f"[prompt-gen] loading theme tree: {tree.name}")

    catalog = (
        load_prompt_generation_catalog(profile_path)
        if profile_path
        else load_prompt_generation_catalog()
    )
    profiles = dict(catalog["profiles"])
    profile_name = str(tree.defaults.get("generation_profile") or "photorealistic")
    try:
        generation_profile = dict(profiles[profile_name])
    except KeyError as exc:
        raise PromptGenerationError(
            f"Unknown prompt-generation profile: {profile_name}"
        ) from exc

    resolved_intended_modality = intended_modality or str(tree.defaults.get("intended_modality", "image"))
    resolved_prompt_config = _resolve_prompt_generation_config(
        tree=tree,
        catalog=catalog,
        template_name=template_name,
        style_family_name=style_family_name,
        target_model_name=target_model_name,
    )
    few_shot_examples = _select_few_shot_examples(
        style_family=resolved_prompt_config["style_family"],
        intended_modality=resolved_intended_modality,
    )

    samples = build_sampling_plan(
        tree=tree,
        seed=seed,
        count_config_path=count_config_path,
    )
    sampling_plan = summarize_sampling_plan(samples)
    sampling_plan["tree_name"] = tree.name
    sampling_plan["seed"] = seed

    progress.env_message(f"[prompt-gen] planned {len(samples)} prompt requests")
    generation_log: list[dict[str, Any]] = [
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "template_resolved",
            "prompt_template": resolved_prompt_config["template"].name,
            "prompt_template_version": resolved_prompt_config["template"].version,
        },
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "style_family_resolved",
            "prompt_style_family": resolved_prompt_config["style_family"].name,
            "prompt_style_family_version": resolved_prompt_config["style_family"].version,
            "resolved_style_source": resolved_prompt_config["resolved_style_source"],
            "target_model_name": resolved_prompt_config["target_model_name"],
        },
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "few_shot_examples_selected",
            "prompt_style_family": resolved_prompt_config["style_family"].name,
            "few_shot_example_ids": [str(example.get("id", "")) for example in few_shot_examples],
            "few_shot_example_count": len(few_shot_examples),
        },
    ]

    resolved_llm_model = (
        llm_model
        or str(generation_profile.get("default_llm_model") or "").strip()
        or "Qwen3-32B"
    )
    prompts = _synthesize_prompts(
        samples=samples,
        tree=tree,
        bundle_id=bundle_id,
        bundle_dir=bundle_dir,
        generation_profile_name=profile_name,
        generation_profile=generation_profile,
        template=resolved_prompt_config["template"],
        style_family=resolved_prompt_config["style_family"],
        target_model_name=resolved_prompt_config["target_model_name"],
        few_shot_examples=few_shot_examples,
        resolved_style_source=resolved_prompt_config["resolved_style_source"],
        llm_model=resolved_llm_model,
        execution_mode=execution_mode,
        intended_modality=resolved_intended_modality,
        seed=seed,
        generation_log=generation_log,
        progress=progress,
    )
    stats = build_prompt_stats(prompts)
    manifest = {
        "bundle_id": bundle_id,
        "bundle_dir": str(bundle_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "tree_path": str(Path(tree_path)),
        "tree_name": tree.name,
        "tree_version": tree.version,
        "generation_profile": profile_name,
        "realism_target": generation_profile.get("realism_target", "photorealistic"),
        "prompt_template": resolved_prompt_config["template"].name,
        "prompt_style_family": resolved_prompt_config["style_family"].name,
        "target_model_name": resolved_prompt_config["target_model_name"],
        "resolved_style_source": resolved_prompt_config["resolved_style_source"],
        "few_shot_example_count": len(few_shot_examples),
        "seed": seed,
        "prompt_count": len(prompts),
        "execution_mode": execution_mode,
        "llm_model": resolved_llm_model,
        "intended_modality": resolved_intended_modality,
    }
    paths = write_prompt_bundle(
        bundle_dir=bundle_dir,
        prompts=prompts,
        manifest=manifest,
        sampling_plan=sampling_plan,
        generation_log=generation_log,
        stats=stats,
    )
    return PromptGenerationSummary(
        bundle_id=bundle_id,
        bundle_dir=str(bundle_dir),
        prompts_path=paths["prompts_path"],
        manifest_path=paths["manifest_path"],
        sampling_plan_path=paths["sampling_plan_path"],
        generation_log_path=paths["generation_log_path"],
        stats_path=paths["stats_path"],
        prompt_count=len(prompts),
        llm_model=resolved_llm_model,
        execution_mode=execution_mode,
        prompt_template=resolved_prompt_config["template"].name,
        prompt_style_family=resolved_prompt_config["style_family"].name,
        target_model_name=resolved_prompt_config["target_model_name"],
        few_shot_example_count=len(few_shot_examples),
    )


def _synthesize_prompts(
    *,
    samples: list[SampledTheme],
    tree: ThemeTree,
    bundle_id: str,
    bundle_dir: Path,
    generation_profile_name: str,
    generation_profile: dict[str, Any],
    template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    llm_model: str,
    execution_mode: str,
    intended_modality: str,
    seed: int,
    generation_log: list[dict[str, Any]],
    progress: RunProgress,
) -> list[PromptRecord]:
    drafts = [
        _build_combination_draft(sample, generation_profile, seed + index)
        for index, sample in enumerate(samples, start=1)
    ]
    if execution_mode == "mock":
        prompts = [
            _build_prompt_record_from_internal_synthesis(
                sample=sample,
                draft=draft,
                tree=tree,
                bundle_id=bundle_id,
                generation_profile_name=generation_profile_name,
                prompt_template=template,
                style_family=style_family,
                target_model_name=target_model_name,
                few_shot_examples=few_shot_examples,
                resolved_style_source=resolved_style_source,
                intended_modality=intended_modality,
                llm_model=llm_model,
                seed=seed,
                prompt_index=index,
                dedupe_pass=0,
            )
            for index, (sample, draft) in enumerate(zip(samples, drafts, strict=True), start=1)
        ]
        generation_log.extend(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "deterministic_synthesis",
                "sample_id": sample.sample_id,
                "theme_path": list(sample.theme_path),
            }
            for sample in samples
        )
        return _dedupe_prompt_records(
            prompts=prompts,
            samples=samples,
            drafts=drafts,
            tree=tree,
            bundle_id=bundle_id,
            generation_profile_name=generation_profile_name,
            prompt_template=template,
            style_family=style_family,
            target_model_name=target_model_name,
            few_shot_examples=few_shot_examples,
            resolved_style_source=resolved_style_source,
            intended_modality=intended_modality,
            llm_model=llm_model,
            seed=seed,
            generation_log=generation_log,
            regenerate_with=lambda sample, draft, prompt_index, dedupe_pass, avoid_texts: _build_prompt_record_from_internal_synthesis(
                sample=sample,
                draft=_mutate_combination_draft(
                    draft,
                    generation_profile,
                    seed + prompt_index + dedupe_pass,
                ),
                tree=tree,
                bundle_id=bundle_id,
                generation_profile_name=generation_profile_name,
                prompt_template=template,
                style_family=style_family,
                target_model_name=target_model_name,
                few_shot_examples=few_shot_examples,
                resolved_style_source=resolved_style_source,
                intended_modality=intended_modality,
                llm_model=llm_model,
                seed=seed,
                prompt_index=prompt_index,
                dedupe_pass=dedupe_pass,
                avoid_texts=avoid_texts,
            ),
        )

    progress.env_message(f"[prompt-gen] synthesizing prompts with llm model: {llm_model}")
    prompts = _run_llm_synthesis(
        samples=samples,
        drafts=drafts,
        tree=tree,
        bundle_id=bundle_id,
        bundle_dir=bundle_dir,
        generation_profile_name=generation_profile_name,
        template=template,
        style_family=style_family,
        target_model_name=target_model_name,
        few_shot_examples=few_shot_examples,
        resolved_style_source=resolved_style_source,
        intended_modality=intended_modality,
        llm_model=llm_model,
        execution_mode=execution_mode,
        seed=seed,
        generation_log=generation_log,
    )
    return _dedupe_prompt_records(
        prompts=prompts,
        samples=samples,
        drafts=drafts,
        tree=tree,
        bundle_id=bundle_id,
        generation_profile_name=generation_profile_name,
        prompt_template=template,
        style_family=style_family,
        target_model_name=target_model_name,
        few_shot_examples=few_shot_examples,
        resolved_style_source=resolved_style_source,
        intended_modality=intended_modality,
        llm_model=llm_model,
        seed=seed,
        generation_log=generation_log,
        regenerate_with=lambda sample, draft, prompt_index, dedupe_pass, avoid_texts: _run_llm_single_rewrite(
            sample=sample,
            draft=draft,
            tree=tree,
            bundle_id=bundle_id,
            bundle_dir=bundle_dir,
            generation_profile_name=generation_profile_name,
            template=template,
            style_family=style_family,
            target_model_name=target_model_name,
            few_shot_examples=few_shot_examples,
            resolved_style_source=resolved_style_source,
            intended_modality=intended_modality,
            llm_model=llm_model,
            execution_mode=execution_mode,
            seed=seed,
            prompt_index=prompt_index,
            dedupe_pass=dedupe_pass,
            avoid_texts=avoid_texts,
            generation_log=generation_log,
        ),
    )


def _run_llm_synthesis(
    *,
    samples: list[SampledTheme],
    drafts: list[dict[str, str]],
    tree: ThemeTree,
    bundle_id: str,
    bundle_dir: Path,
    generation_profile_name: str,
    template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    intended_modality: str,
    llm_model: str,
    execution_mode: str,
    seed: int,
    generation_log: list[dict[str, Any]],
) -> list[PromptRecord]:
    request_prompts: list[PromptRecord] = []
    for index, (sample, draft) in enumerate(zip(samples, drafts, strict=True), start=1):
        request_prompts.append(
            PromptRecord(
                prompt_id=f"synth_{index:06d}",
                prompt=_render_llm_instruction(
                    sample=sample,
                    draft=draft,
                    template=template,
                    style_family=style_family,
                    few_shot_examples=few_shot_examples,
                    generation_profile_name=generation_profile_name,
                    intended_modality=intended_modality,
                    avoid_texts=[],
                ),
                language=str(tree.defaults.get("language", "en")),
                metadata={
                    "sample_id": sample.sample_id,
                    "theme_path": list(sample.theme_path),
                },
            )
        )
    return _execute_llm_requests(
        request_prompts=request_prompts,
        samples=samples,
        drafts=drafts,
        tree=tree,
        bundle_id=bundle_id,
        bundle_dir=bundle_dir,
        generation_profile_name=generation_profile_name,
        prompt_template=template,
        style_family=style_family,
        target_model_name=target_model_name,
        few_shot_examples=few_shot_examples,
        resolved_style_source=resolved_style_source,
        intended_modality=intended_modality,
        llm_model=llm_model,
        execution_mode=execution_mode,
        seed=seed,
        generation_log=generation_log,
        prompt_index_offset=0,
        dedupe_pass=0,
    )


def _run_llm_single_rewrite(
    *,
    sample: SampledTheme,
    draft: dict[str, str],
    tree: ThemeTree,
    bundle_id: str,
    bundle_dir: Path,
    generation_profile_name: str,
    template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    intended_modality: str,
    llm_model: str,
    execution_mode: str,
    seed: int,
    prompt_index: int,
    dedupe_pass: int,
    avoid_texts: list[str],
    generation_log: list[dict[str, Any]],
) -> PromptRecord:
    prompts = [
        PromptRecord(
            prompt_id=f"rewrite_{prompt_index:06d}_pass_{dedupe_pass}",
            prompt=_render_llm_instruction(
                sample=sample,
                draft=draft,
                template=template,
                style_family=style_family,
                few_shot_examples=few_shot_examples,
                generation_profile_name=generation_profile_name,
                intended_modality=intended_modality,
                avoid_texts=avoid_texts,
            ),
            language=str(tree.defaults.get("language", "en")),
            metadata={"sample_id": sample.sample_id, "theme_path": list(sample.theme_path)},
        )
    ]
    records = _execute_llm_requests(
        request_prompts=prompts,
        samples=[sample],
        drafts=[draft],
        tree=tree,
        bundle_id=bundle_id,
        bundle_dir=bundle_dir,
        generation_profile_name=generation_profile_name,
        prompt_template=template,
        style_family=style_family,
        target_model_name=target_model_name,
        few_shot_examples=few_shot_examples,
        resolved_style_source=resolved_style_source,
        intended_modality=intended_modality,
        llm_model=llm_model,
        execution_mode=execution_mode,
        seed=seed,
        generation_log=generation_log,
        prompt_index_offset=prompt_index - 1,
        dedupe_pass=dedupe_pass,
    )
    return records[0]


def _execute_llm_requests(
    *,
    request_prompts: list[PromptRecord],
    samples: list[SampledTheme],
    drafts: list[dict[str, str]],
    tree: ThemeTree,
    bundle_id: str,
    bundle_dir: Path,
    generation_profile_name: str,
    prompt_template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    intended_modality: str,
    llm_model: str,
    execution_mode: str,
    seed: int,
    generation_log: list[dict[str, Any]],
    prompt_index_offset: int,
    dedupe_pass: int,
) -> list[PromptRecord]:
    requests_path = bundle_dir / "_llm_requests" / f"pass_{dedupe_pass:02d}.jsonl"
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    with requests_path.open("w", encoding="utf-8") as handle:
        for prompt in request_prompts:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "language": prompt.language,
                        "metadata": prompt.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    llm_run_dir = bundle_dir / "_llm_runs" / f"pass_{dedupe_pass:02d}"
    summary = run_single_model(
        model_name=llm_model,
        prompt_file=requests_path,
        out_dir=llm_run_dir,
        run_name=f"{bundle_id}-llm-pass-{dedupe_pass:02d}",
        execution_mode=execution_mode,
    )
    dataset_records = [
        json.loads(line)
        for line in Path(summary.export_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    text_by_prompt_id: dict[str, str] = {}
    for record in dataset_records:
        artifact_path = Path(str(record["artifact_path"]))
        text_by_prompt_id[str(record["prompt_id"])] = artifact_path.read_text(encoding="utf-8")

    prompt_records: list[PromptRecord] = []
    for index, (request_prompt, sample, draft) in enumerate(
        zip(request_prompts, samples, drafts, strict=True),
        start=1,
    ):
        raw_response = text_by_prompt_id.get(request_prompt.prompt_id, "")
        parsed = _parse_llm_response(raw_response)
        generation_log.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "llm_synthesis",
                "sample_id": sample.sample_id,
                "request_prompt_id": request_prompt.prompt_id,
                "llm_run_id": summary.run_id,
                "dedupe_pass": dedupe_pass,
            }
        )
        prompt_records.append(
            _build_prompt_record(
                sample=sample,
                draft=draft,
                tree=tree,
                bundle_id=bundle_id,
                generation_profile_name=generation_profile_name,
                prompt_template=prompt_template,
                style_family=style_family,
                target_model_name=target_model_name,
                few_shot_examples=few_shot_examples,
                resolved_style_source=resolved_style_source,
                intended_modality=intended_modality,
                llm_model=llm_model,
                seed=seed,
                prompt_index=prompt_index_offset + index,
                dedupe_pass=dedupe_pass,
                prompt_text=parsed["prompt"],
                negative_prompt=parsed.get("negative_prompt"),
                annotation_hints=parsed.get("annotation_hints"),
                tags=parsed.get("tags"),
            )
        )
    return prompt_records


def _dedupe_prompt_records(
    *,
    prompts: list[PromptRecord],
    samples: list[SampledTheme],
    drafts: list[dict[str, str]],
    tree: ThemeTree,
    bundle_id: str,
    generation_profile_name: str,
    prompt_template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    intended_modality: str,
    llm_model: str | None,
    seed: int,
    generation_log: list[dict[str, Any]],
    regenerate_with: Callable[[SampledTheme, dict[str, str], int, int, list[str]], PromptRecord],
) -> list[PromptRecord]:
    del prompt_template, style_family, target_model_name, few_shot_examples, resolved_style_source
    del intended_modality, llm_model, seed
    dedupe_pass = 1
    while dedupe_pass <= 2:
        duplicate_groups = _find_duplicate_groups(prompts)
        if not duplicate_groups:
            break
        seen_rewrites: set[int] = set()
        for group in duplicate_groups:
            anchor_prompt = prompts[group[0]].prompt
            for index in group[1:]:
                if index in seen_rewrites:
                    continue
                seen_rewrites.add(index)
                regenerated = regenerate_with(
                    samples[index],
                    drafts[index],
                    index + 1,
                    dedupe_pass,
                    [anchor_prompt],
                )
                prompts[index] = regenerated
                generation_log.append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "event": "dedupe_regeneration",
                        "prompt_id": regenerated.prompt_id,
                        "sample_id": samples[index].sample_id,
                        "dedupe_pass": dedupe_pass,
                    }
                )
        dedupe_pass += 1
    return prompts


def _find_duplicate_groups(prompts: list[PromptRecord]) -> list[list[int]]:
    normalized_groups: dict[str, list[int]] = {}
    for index, prompt in enumerate(prompts):
        normalized_groups.setdefault(_normalize_prompt_for_similarity(prompt.prompt), []).append(index)

    groups: list[list[int]] = [group for group in normalized_groups.values() if len(group) > 1]
    token_sets = [_tokenize_for_similarity(prompt.prompt) for prompt in prompts]
    for left in range(len(prompts)):
        for right in range(left + 1, len(prompts)):
            if any(left in group and right in group for group in groups):
                continue
            if _jaccard_similarity(token_sets[left], token_sets[right]) >= 0.9:
                groups.append([left, right])
    return groups


def _build_prompt_record_from_internal_synthesis(
    *,
    sample: SampledTheme,
    draft: dict[str, str],
    tree: ThemeTree,
    bundle_id: str,
    generation_profile_name: str,
    prompt_template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    intended_modality: str,
    llm_model: str | None,
    seed: int,
    prompt_index: int,
    dedupe_pass: int,
    avoid_texts: list[str] | None = None,
) -> PromptRecord:
    prompt_text = _render_internal_prompt(
        sample=sample,
        draft=draft,
        style_family_name=style_family.name,
    )
    if avoid_texts:
        for avoid in avoid_texts:
            if prompt_text == avoid:
                prompt_text = f"{prompt_text}, captured from a new angle with distinct motion and environmental detail"
    return _build_prompt_record(
        sample=sample,
        draft=draft,
        tree=tree,
        bundle_id=bundle_id,
        generation_profile_name=generation_profile_name,
        prompt_template=prompt_template,
        style_family=style_family,
        target_model_name=target_model_name,
        few_shot_examples=few_shot_examples,
        resolved_style_source=resolved_style_source,
        intended_modality=intended_modality,
        llm_model=llm_model,
        seed=seed,
        prompt_index=prompt_index,
        dedupe_pass=dedupe_pass,
        prompt_text=prompt_text,
        negative_prompt=_default_negative_prompt(),
        annotation_hints=_default_annotation_hints(sample=sample, draft=draft),
        tags=sample.tags,
    )


def _build_prompt_record(
    *,
    sample: SampledTheme,
    draft: dict[str, str],
    tree: ThemeTree,
    bundle_id: str,
    generation_profile_name: str,
    prompt_template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    target_model_name: str | None,
    few_shot_examples: list[dict[str, Any]],
    resolved_style_source: str,
    intended_modality: str,
    llm_model: str | None,
    seed: int,
    prompt_index: int,
    dedupe_pass: int,
    prompt_text: str,
    negative_prompt: str | None,
    annotation_hints: Any,
    tags: Any,
) -> PromptRecord:
    created_at = datetime.now(UTC).isoformat()
    metadata = {
        "uuid": str(uuid.uuid4()),
        "created_at": created_at,
        "generation_profile": generation_profile_name,
        "realism_target": "photorealistic",
        "prompt_template": prompt_template.name,
        "prompt_template_version": prompt_template.version,
        "prompt_style_family": style_family.name,
        "prompt_style_family_version": style_family.version,
        "target_model_name": target_model_name,
        "few_shot_example_ids": [str(example.get("id", "")) for example in few_shot_examples],
        "instruction_render_version": "phase28_v1",
        "resolved_style_source": resolved_style_source,
        "category": sample.category,
        "subcategory": sample.subcategory,
        "subtopic": sample.subtopic,
        "theme": sample.theme,
        "theme_path": list(sample.theme_path),
        "theme_tree_name": tree.name,
        "sampling_seed": seed,
        "llm_model": llm_model,
        "llm_generation_version": "phase28_v1",
        "bundle_id": bundle_id,
        "bundle_prompt_index": prompt_index,
        "resampled": sample.resampled,
        "dedupe_pass": dedupe_pass,
        "language": str(tree.defaults.get("language", "en")),
        "intended_modality": intended_modality,
        "annotation_hints": annotation_hints or _default_annotation_hints(sample=sample, draft=draft),
        "tags": list(tags) if isinstance(tags, list) else list(sample.tags),
        "quota_source_path": list(sample.quota_source_path),
        "constraints": dict(sample.constraints),
    }
    metadata.update(sample.metadata)
    return PromptRecord(
        prompt_id=f"prompt_{prompt_index:06d}",
        prompt=normalize_text(prompt_text),
        negative_prompt=normalize_text(negative_prompt) if negative_prompt else None,
        language=str(tree.defaults.get("language", "en")),
        parameters={},
        metadata=metadata,
        version=tree.version,
    )


def _build_combination_draft(
    sample: SampledTheme,
    profile: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    return {
        "scene": _pick(profile.get("scene_pool"), rng, default="a grounded real-world scene"),
        "lighting": _pick(profile.get("lighting_pool"), rng, default="natural daylight"),
        "weather": _pick(profile.get("weather_pool"), rng, default="clear conditions"),
        "subject_state": _pick(profile.get("subject_state_pool"), rng, default="authentic movement"),
        "camera": _pick(profile.get("camera_pool"), rng, default="a realistic documentary-style camera angle"),
        "realism_anchor": _pick(profile.get("realism_anchors"), rng, default="photorealistic detail"),
    }


def _mutate_combination_draft(
    draft: dict[str, str],
    profile: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    mutated = dict(draft)
    rng = random.Random(seed)
    for key, pool_key in (
        ("scene", "scene_pool"),
        ("lighting", "lighting_pool"),
        ("weather", "weather_pool"),
        ("subject_state", "subject_state_pool"),
        ("camera", "camera_pool"),
    ):
        mutated[key] = _pick(profile.get(pool_key), rng, default=mutated[key])
    return mutated


def _render_internal_prompt(
    *,
    sample: SampledTheme,
    draft: dict[str, str],
    style_family_name: str,
) -> str:
    if style_family_name == "keyword_list":
        return normalize_text(
            ", ".join(
                [
                    sample.theme,
                    draft["scene"],
                    draft["lighting"],
                    draft["weather"],
                    draft["subject_state"],
                    draft["camera"],
                    draft["realism_anchor"],
                ]
            )
        )
    if style_family_name == "short_sentence":
        return (
            f"{sample.theme} in {draft['scene']}, with {draft['lighting']} and {draft['weather']}, "
            f"showing {draft['subject_state']} from {draft['camera']}."
        )
    return (
        f"{sample.theme}. {draft['scene']}. {draft['lighting']}. "
        f"{draft['weather']}. The subject shows {draft['subject_state']}. "
        f"Seen from {draft['camera']}. Preserve {draft['realism_anchor']} with grounded, realistic detail."
    )


def _render_llm_instruction(
    *,
    sample: SampledTheme,
    draft: dict[str, str],
    template: PromptTemplateConfig,
    style_family: PromptStyleFamilyConfig,
    few_shot_examples: list[dict[str, Any]],
    generation_profile_name: str,
    intended_modality: str,
    avoid_texts: list[str],
) -> str:
    avoid_block = "\n".join(f"- {text}" for text in avoid_texts) if avoid_texts else "- none"
    return render_instruction_template(
        template.instruction_template,
        values={
            "generation_profile_name": generation_profile_name,
            "intended_modality": intended_modality,
            "category": sample.category,
            "subcategory": sample.subcategory,
            "subtopic": sample.subtopic,
            "theme": sample.theme,
            "theme_path": " / ".join(sample.theme_path),
            "scene_guidance": draft["scene"],
            "lighting_guidance": draft["lighting"],
            "weather_guidance": draft["weather"],
            "subject_state_guidance": draft["subject_state"],
            "camera_guidance": draft["camera"],
            "realism_anchor": draft["realism_anchor"],
            "style_instruction": style_family.style_instruction,
            "few_shot_block": _render_few_shot_block(few_shot_examples),
            "output_contract": _render_output_contract(style_family),
            "avoid_block": avoid_block,
        },
    )


def _parse_llm_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {"prompt": "A photorealistic scene with grounded detail."}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return _normalize_llm_payload(payload)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return _normalize_llm_payload(payload)
        except json.JSONDecodeError:
            pass
    return {"prompt": normalize_text(text)}


def _normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_text = normalize_text(str(payload.get("prompt", "")))
    negative_prompt = payload.get("negative_prompt")
    annotation_hints = payload.get("annotation_hints")
    tags = payload.get("tags")
    return {
        "prompt": prompt_text or "A photorealistic real-world scene with grounded detail.",
        "negative_prompt": normalize_text(str(negative_prompt)) if negative_prompt else None,
        "annotation_hints": annotation_hints,
        "tags": tags,
    }


def _default_negative_prompt() -> str:
    return (
        "anime, illustration, stylized concept art, low quality, blurry, distorted, overprocessed"
    )


def _default_annotation_hints(*, sample: SampledTheme, draft: dict[str, str]) -> dict[str, Any]:
    return {
        "scene_category": sample.category,
        "subject_category": sample.subcategory,
        "realism_requirement": "photorealistic",
        "style_preference": "realistic",
        "camera_hint": draft["camera"],
        "lighting_hint": draft["lighting"],
    }


def _pick(values: Any, rng: random.Random, *, default: str) -> str:
    if not isinstance(values, list) or not values:
        return default
    return str(rng.choice(values))


def _normalize_prompt_for_similarity(value: str) -> str:
    normalized = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", normalized)


def _tokenize_for_similarity(value: str) -> set[str]:
    return {token for token in _normalize_prompt_for_similarity(value).split() if token}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _generate_bundle_id(tree_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", tree_name).strip("-").lower() or "prompt-run"
    return f"{slug}_{timestamp}"


def _resolve_prompt_generation_config(
    *,
    tree: ThemeTree,
    catalog: dict[str, Any],
    template_name: str | None,
    style_family_name: str | None,
    target_model_name: str | None,
) -> dict[str, Any]:
    templates: dict[str, PromptTemplateConfig] = dict(catalog["templates"])
    style_families: dict[str, PromptStyleFamilyConfig] = dict(catalog["style_families"])
    target_style_mappings: dict[str, str] = dict(catalog.get("target_style_mappings", {}))

    resolved_template_name = str(
        template_name
        or tree.defaults.get("prompt_template")
        or DEFAULT_TEMPLATE_NAME
    )
    try:
        template = templates[resolved_template_name]
    except KeyError as exc:
        raise PromptGenerationError(
            f"Unknown prompt template: {resolved_template_name}"
        ) from exc

    if style_family_name:
        resolved_style_name = str(style_family_name)
        resolved_style_source = "cli"
    elif tree.defaults.get("prompt_style_family"):
        resolved_style_name = str(tree.defaults.get("prompt_style_family"))
        resolved_style_source = "tree_default"
    elif target_model_name and target_style_mappings.get(str(target_model_name)):
        resolved_style_name = str(target_style_mappings[str(target_model_name)])
        resolved_style_source = "target_model_mapping"
    else:
        resolved_style_name = str(template.default_style_family or DEFAULT_STYLE_FAMILY_NAME)
        resolved_style_source = "template_default"

    try:
        style_family = style_families[resolved_style_name]
    except KeyError as exc:
        raise PromptGenerationError(
            f"Unknown prompt style family: {resolved_style_name}"
        ) from exc

    return {
        "template": template,
        "style_family": style_family,
        "target_model_name": str(target_model_name) if target_model_name else None,
        "resolved_style_source": resolved_style_source,
    }


def _select_few_shot_examples(
    *,
    style_family: PromptStyleFamilyConfig,
    intended_modality: str,
) -> list[dict[str, Any]]:
    modality = str(intended_modality)
    if style_family.supported_modalities and modality not in style_family.supported_modalities:
        return []
    selected: list[dict[str, Any]] = []
    for example in style_family.few_shot_examples:
        if not isinstance(example, dict):
            continue
        applicability = example.get("applicability", {})
        if isinstance(applicability, dict):
            modalities = applicability.get("modalities")
            if isinstance(modalities, list) and modalities and modality not in [str(item) for item in modalities]:
                continue
        selected.append(dict(example))
        if len(selected) >= style_family.max_examples_per_request:
            break
    return selected


def _render_output_contract(style_family: PromptStyleFamilyConfig) -> str:
    contract = dict(style_family.output_contract)
    if not contract:
        return "Return JSON only with keys: prompt, negative_prompt, annotation_hints, tags."
    return json.dumps(contract, ensure_ascii=False, sort_keys=True)


def _render_few_shot_block(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "No few-shot examples selected."
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        example_id = str(example.get("id", f"example_{index}"))
        input_payload = example.get("input", {})
        output_payload = example.get("output", {})
        blocks.append(
            "\n".join(
                [
                    f"Example {index} ({example_id})",
                    f"Input: {json.dumps(input_payload, ensure_ascii=False, sort_keys=True)}",
                    f"Output JSON: {json.dumps(output_payload, ensure_ascii=False, sort_keys=True)}",
                ]
            )
        )
    return "\n\n".join(blocks)
