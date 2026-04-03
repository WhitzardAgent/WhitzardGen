from __future__ import annotations

import json
from pathlib import Path

from whitzard.benchmarking.interfaces import BenchmarkBuildOutput, BenchmarkBuildRequest, BenchmarkBuilder
from whitzard.benchmarking.models import BenchmarkCase
from whitzard.benchmarking.service import slugify
from whitzard.prompt_generation import generate_prompt_bundle
from whitzard.prompts import load_prompts


class ThemeTreeBenchmarkBuilder(BenchmarkBuilder):
    builder_id = "theme_tree"
    description = "Build benchmark cases from the theme-tree prompt generation pipeline."

    def build(self, request: BenchmarkBuildRequest) -> BenchmarkBuildOutput:
        if request.source_path is None:
            raise RuntimeError("theme_tree builder requires --source <theme_tree.yaml>.")
        source_path = Path(request.source_path)
        benchmark_id = slugify(request.benchmark_name or source_path.stem)
        prompt_out_dir = None
        if request.out_dir is not None:
            prompt_out_dir = str(Path(request.out_dir) / "_prompt_builder")
        prompt_bundle = generate_prompt_bundle(
            tree_path=source_path,
            out_dir=prompt_out_dir,
            llm_model=request.llm_model,
            execution_mode=request.execution_mode,
            seed=request.seed,
            count_config_path=request.count_config_path,
            profile_path=request.profile_path,
            template_name=request.template_name,
            style_family_name=request.style_family_name,
            target_model_name=request.target_model_name,
            intended_modality=request.intended_modality,
            progress=request.progress,
        )
        prompt_records = load_prompts(prompt_bundle.prompts_path)
        manifest_payload = json.loads(Path(prompt_bundle.manifest_path).read_text(encoding="utf-8"))
        cases = [
            BenchmarkCase(
                benchmark_id=benchmark_id,
                case_id=record.prompt_id,
                input_type=str(manifest_payload.get("intended_modality") or request.intended_modality or "text"),
                prompt=record.prompt,
                instruction=None,
                metadata={
                    **dict(record.metadata),
                    "prompt_bundle_id": prompt_bundle.bundle_id,
                    "prompt_bundle_manifest": manifest_payload,
                },
                tags=list(dict(record.metadata).get("tags", []) or []),
                split=str(dict(record.metadata).get("split", "default")),
                context=None,
                expected_structure=None,
                case_version=record.version,
                source_builder=self.builder_id,
                language=record.language,
                parameters=dict(record.parameters),
            )
            for record in prompt_records
        ]
        return BenchmarkBuildOutput(
            cases=cases,
            source_path=str(source_path),
            build_mode="dynamic",
            extra_manifest={
                "prompt_bundle_id": prompt_bundle.bundle_id,
                "prompt_bundle_manifest_path": prompt_bundle.manifest_path,
                "prompt_template": prompt_bundle.prompt_template,
                "prompt_style_family": prompt_bundle.prompt_style_family,
                "llm_model": prompt_bundle.llm_model,
            },
        )
