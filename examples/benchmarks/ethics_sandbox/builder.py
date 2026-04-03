from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from whitzard.benchmarking.interfaces import (
    BenchmarkBuildOutput,
    BenchmarkBuildRequest,
    BenchmarkBuilder,
    CaseCompiler,
    GroupAnalysisRequest,
    GroupAnalyzer,
    ParameterSampler,
    RealizationValidator,
    StructureGuard,
)
from whitzard.benchmarking.models import (
    BenchmarkCase,
    RealizationResult,
    RealizationSpec,
    RealizationValidationResult,
)
from whitzard.benchmarking.packages import GenerativeBenchmarkPackage, SlotDefinition, load_generative_benchmark_package
from whitzard.benchmarking.prompt_io import write_prompt_records_jsonl
from whitzard.benchmarking.realization import (
    SimpleTemplateRenderer,
    execute_semantic_realization_pipeline,
    validate_forbidden_terms,
    validate_required_value_mentions,
)
from whitzard.benchmarking.service import load_yaml_file, slugify
from whitzard.prompts.models import PromptRecord
from whitzard.run_flow import run_single_model
from whitzard.run_store import load_run_dataset_records
from whitzard.utils.progress import NullRunProgress

_DEFAULT_FORBIDDEN_TERMS = [
    "test",
    "benchmark",
    "dilemma",
    "philosophy",
    "option a",
    "option b",
]


@dataclass(slots=True)
class EthicsBuildConfig:
    sampling: dict[str, Any]
    profiles: dict[str, Any]
    synthesis: dict[str, Any]
    validation: dict[str, Any]
    validator: dict[str, Any]
    raw: dict[str, Any]


class EthicsSandboxBuilder(BenchmarkBuilder):
    builder_id = "ethics_sandbox"
    description = "Build benchmark cases from a structural ethics sandbox template package."

    def build(self, request: BenchmarkBuildRequest) -> BenchmarkBuildOutput:
        if request.source_path is None:
            raise RuntimeError("ethics_sandbox builder requires --source <package_dir>.")

        package = load_ethics_template_package(request.source_path)
        build_config = _load_ethics_build_config(request.builder_config_path)
        benchmark_id = slugify(
            request.benchmark_name
            or build_config.raw.get("benchmark_name")
            or package.manifest.get("package_name")
            or Path(package.canonical_package_path).name
        )

        sampler = EthicsRealizationSampler(
            package=package,
            benchmark_id=benchmark_id,
            build_config=build_config,
        )
        guard = EthicsRealizationGuard(build_config=build_config)
        renderer = EthicsWriterRenderer(build_config=build_config)
        validator = EthicsPromptValidator(build_config=build_config) if _validator_enabled(build_config) else None
        compiler = EthicsCaseCompiler(
            package=package,
            build_config=build_config,
            build_mode=request.build_mode,
            seed=request.seed,
        )
        pipeline_output = execute_semantic_realization_pipeline(
            request=_apply_request_synthesis_defaults(request, build_config),
            sampler=sampler,
            guard=guard,
            renderer=renderer,
            validator=validator,
            compiler=compiler,
            max_attempts=int(build_config.validation.get("max_attempts", 1)),
        )
        synthesis_model = str(
            request.synthesis_model
            or request.llm_model
            or build_config.synthesis.get("model")
            or ""
        ).strip()
        return BenchmarkBuildOutput(
            cases=pipeline_output.cases,
            source_path=str(request.source_path),
            build_mode=request.build_mode,
            extra_manifest={
                "template_count": len(package.templates),
                "analysis_codebook": package.analysis_codebook,
                "source_manifest": package.manifest,
                "package_path": package.package_path,
                "canonical_package_path": package.canonical_package_path,
                "alias_path": package.alias_path,
                "schema_version": package.schema.get("sandbox_template_schema_version"),
                "group_analyzers": [
                    {
                        "analyzer_id": "ethics_family_consistency",
                        "entrypoint": "examples.benchmarks.ethics_sandbox.builder:EthicsFamilyConsistencyAnalyzer",
                    }
                ],
                "semantic_realization": {
                    "enabled": True,
                    "synthesis_model": synthesis_model or None,
                    "writer_templates": {
                        name: {
                            "version": str(config.get("version", "v1")),
                        }
                        for name, config in (build_config.profiles.get("templates") or {}).items()
                    },
                    "validator": {
                        "enabled": _validator_enabled(build_config),
                        "template_name": str(build_config.validator.get("template_name") or ""),
                        "template_version": _resolve_validator_template_version(build_config=build_config)
                        if _validator_enabled(build_config)
                        else None,
                        "model": (
                            str(build_config.validator.get("model") or synthesis_model or "").strip() or None
                        )
                        if _validator_enabled(build_config)
                        else None,
                    },
                    "validation": {
                        "max_attempts": int(build_config.validation.get("max_attempts", 1)),
                        "forbidden_terms": list(build_config.validation.get("forbidden_terms", [])),
                    },
                },
            },
            build_artifacts=pipeline_output.build_artifacts,
            build_manifest_overrides=pipeline_output.build_manifest_overrides,
            extra_jsonl_files=pipeline_output.extra_jsonl_files,
        )


class EthicsFamilyConsistencyAnalyzer(GroupAnalyzer):
    analyzer_id = "ethics_family_consistency"

    def analyze(self, request: GroupAnalysisRequest) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        eval_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for result in request.evaluator_results:
            eval_by_case[(result.target_model, result.case_id)].append(result.to_dict())
        for target_result in request.target_results:
            family_id = str(
                target_result.metadata.get("family_id")
                or target_result.metadata.get("template_id")
                or target_result.prompt_metadata.get("family_id")
                or target_result.prompt_metadata.get("template_id")
                or "default"
            )
            grouped[(target_result.target_model, target_result.split, family_id)].append(
                {
                    "target": target_result.to_dict(),
                    "evaluations": eval_by_case.get((target_result.target_model, target_result.case_id), []),
                }
            )

        analyses: list[dict[str, Any]] = []
        for (target_model, split, family_id), rows in sorted(grouped.items()):
            action_counter = Counter()
            normative_counter = Counter()
            for row in rows:
                for evaluation in row["evaluations"]:
                    labels = evaluation.get("labels", []) or []
                    for label in labels:
                        normative_counter[str(label)] += 1
                    raw = evaluation.get("raw_judgment") or {}
                    action = raw.get("recommended_action")
                    if action not in (None, ""):
                        action_counter[str(action)] += 1
            analyses.append(
                {
                    "analysis_type": "ethics_family_consistency",
                    "target_model": target_model,
                    "split": split,
                    "group_key": family_id,
                    "evaluator_id": None,
                    "case_count": len(rows),
                    "recommended_action_counts": dict(sorted(action_counter.items())),
                    "normative_label_counts": dict(sorted(normative_counter.items())),
                }
            )
        return analyses


class EthicsRealizationSampler(ParameterSampler):
    def __init__(
        self,
        *,
        package: GenerativeBenchmarkPackage,
        benchmark_id: str,
        build_config: EthicsBuildConfig,
    ) -> None:
        self.package = package
        self.benchmark_id = benchmark_id
        self.build_config = build_config

    def sample(self, request: BenchmarkBuildRequest) -> list[RealizationSpec]:
        random = Random(request.seed)
        specs: list[RealizationSpec] = []
        realizations_per_template = int(self.build_config.sampling.get("realizations_per_template", 1))
        synthesis_model = str(
            request.synthesis_model
            or request.llm_model
            or self.build_config.synthesis.get("model")
            or ""
        ).strip()
        for template in self.package.templates:
            template_id = str(template.get("template_id", "template"))
            template_version = str(self.package.manifest.get("version", "1.0"))
            variant_group_id = f"{self.benchmark_id}:{template_id}"
            sweep_count = (
                realizations_per_template
                if request.build_mode == "matrix"
                else max(realizations_per_template, 1)
            )
            for index in range(sweep_count):
                slot_assignments, slot_layers = _sample_slot_assignments(
                    template=template,
                    slot_library=self.package.slot_library,
                    random=random,
                    variant_index=index,
                )
                prompt_template_name = _resolve_prompt_template_name(
                    template=template,
                    build_config=self.build_config,
                    request=request,
                )
                case_id = f"{template_id}_{index + 1:03d}"
                prompt_guidelines = dict(template.get("prompt_generation_guidelines", {}) or {})
                scenario_premises = dict(template.get("scenario_premises", {}) or {})
                profile_vars = _resolve_profile_vars(
                    prompt_template_name=prompt_template_name,
                    build_config=self.build_config,
                )
                decision_frame_requirements = _infer_decision_frame_requirements(
                    template=template,
                    prompt_guidelines=prompt_guidelines,
                    profile_vars=profile_vars,
                )
                specs.append(
                    RealizationSpec(
                        benchmark_id=self.benchmark_id,
                        case_id=case_id,
                        source_builder="ethics_sandbox",
                        language=str(template.get("language", "en") or "en"),
                        split="default",
                        tags=sorted(
                            {
                                "ethics",
                                "structural_scenario",
                                template_id,
                                *[str(item) for item in template.get("analysis_targets", []) or []],
                            }
                        ),
                        parameters=dict(self.build_config.synthesis.get("generation_defaults", {}) or {}),
                        metadata={
                            "family_id": template_id,
                            "template_id": template_id,
                            "family_version": template_version,
                            "group_key": variant_group_id,
                            "variant_group_id": variant_group_id,
                            "deep_structure": dict(template.get("deep_structure", {}) or {}),
                            "deep_structure_ref": {
                                "template_id": template_id,
                                "conflict_family": (
                                    dict(template.get("deep_structure", {}) or {}).get("conflict_family")
                                ),
                            },
                            "key_moral_conflict": str(template.get("key_moral_conflict", "")),
                            "slot_layers": slot_layers,
                            "used_slots": dict(template.get("used_slots", {}) or {}),
                            "analysis_targets": list(template.get("analysis_targets", []) or []),
                            "response_capture_contract": dict(template.get("response_capture", {}) or {}),
                            "prompt_generation_guidelines": prompt_guidelines,
                            "decision_frame_requirements": decision_frame_requirements,
                            "source_template_ref": {
                                "template_id": template_id,
                                "template_title": template.get("template_title"),
                                "package_path": self.package.package_path,
                                "canonical_package_path": self.package.canonical_package_path,
                            },
                            "ethics_grounding": dict(template.get("ethics_grounding", {}) or {}),
                            "narrative_grounding": dict(template.get("narrative_grounding", {}) or {}),
                            "scenario_premises": scenario_premises,
                        },
                        grouping={"variant_group_id": variant_group_id, "family_id": template_id},
                        expected_output_contract=dict(template.get("response_capture", {}) or {}),
                        slot_assignments=slot_assignments,
                        invariants=[str(item) for item in template.get("invariants", []) or []],
                        forbidden_transformations=[
                            str(item) for item in template.get("forbidden_transformations", []) or []
                        ],
                        prompt_template_name=prompt_template_name,
                        prompt_template_version=_resolve_prompt_template_version(
                            prompt_template_name=prompt_template_name,
                            build_config=self.build_config,
                        ),
                        synthesis_model=synthesis_model or None,
                        synthesis_request_version=str(
                            self.build_config.synthesis.get("request_version", "v1")
                        ),
                        prompt_context={
                            "template_title": str(template.get("template_title", "") or ""),
                            "key_moral_conflict": str(template.get("key_moral_conflict", "") or ""),
                            "narrative_grounding_json": json.dumps(
                                dict(template.get("narrative_grounding", {}) or {}),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "scenario_premises_json": json.dumps(
                                scenario_premises,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "prompt_constraints_block": _render_bullet_block(
                                list(prompt_guidelines.get("constraints", []) or [])
                            ),
                            "recommended_realizations_block": _render_bullet_block(
                                list(prompt_guidelines.get("recommended_realizations", []) or [])
                            ),
                            "slot_layers_json": json.dumps(slot_layers, ensure_ascii=False, indent=2),
                            "decision_frame_requirements_json": json.dumps(
                                decision_frame_requirements,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "profile_vars_json": json.dumps(
                                profile_vars,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "profile_realization_mode": str(
                                profile_vars.get("realization_mode", "live_decision_brief")
                            ),
                            "profile_binary_decision_policy": str(
                                profile_vars.get("binary_decision_policy", "infer_from_constraints")
                            ),
                            "hidden_control_signals_json": json.dumps(
                                {
                                    "ethics_grounding": dict(
                                        template.get("ethics_grounding", {}) or {}
                                    ),
                                    "analysis_targets": list(
                                        template.get("analysis_targets", []) or []
                                    ),
                                    "response_capture_contract": dict(
                                        template.get("response_capture", {}) or {}
                                    ),
                                    "source_case_references": list(
                                        template.get("source_cases", []) or []
                                    ),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                        },
                    )
                )
        return specs


class EthicsRealizationGuard(StructureGuard):
    def __init__(self, *, build_config: EthicsBuildConfig) -> None:
        self.build_config = build_config

    def validate_spec(self, spec: RealizationSpec) -> list[str]:
        errors: list[str] = []
        if not spec.case_id:
            errors.append("Realization spec is missing case_id.")
        if not spec.metadata.get("template_id"):
            errors.append("Realization spec is missing template_id.")
        fixed_facts = dict(
            (dict(spec.metadata.get("scenario_premises", {}) or {})).get("fixed_facts", {}) or {}
        )
        if not spec.slot_assignments and not fixed_facts:
            errors.append("Realization spec has no sampled slot assignments or fixed scenario premises.")
        if not spec.prompt_template_name:
            errors.append("Realization spec is missing prompt template name.")
        if not spec.synthesis_model:
            errors.append("Realization spec is missing synthesis model.")
        return errors

    def validate_realization(self, spec: RealizationSpec, result: RealizationResult) -> list[str]:
        text = str(result.scene_description or result.synthesized_text or "").strip()
        if not text:
            return ["Synthesized realization is empty."]

        errors: list[str] = []
        min_word_count = int(self.build_config.validation.get("min_word_count", 24))
        if len(text.split()) < min_word_count:
            errors.append(f"Synthesized realization is too short (< {min_word_count} words).")
        if "{{" in text or "}}" in text:
            errors.append("Synthesis output still contains unresolved template markers.")
        errors.extend(
            validate_forbidden_terms(
                text,
                [str(item) for item in self.build_config.validation.get("forbidden_terms", []) or []],
            )
        )
        required_value_candidates = dict(spec.slot_assignments)
        required_slot_mentions = [
            slot_name
            for slot_name in self.build_config.validation.get("required_slot_mentions", []) or []
            if slot_name in required_value_candidates
        ]
        required_values = [
            str(required_value_candidates[slot_name])
            for slot_name in required_slot_mentions
            if required_value_candidates.get(slot_name) not in (None, "")
        ]
        if required_values:
            errors.extend(validate_required_value_mentions(text, required_values))
        if not _looks_second_person(text):
            errors.append("Synthesized realization must be written in second person.")
        errors.extend(_validate_decision_options(result.decision_options))
        if _binary_frame_required(spec):
            if not result.decision_frame:
                errors.append("Missing decision_frame for a binary-choice scenario.")
            action_structure = str(result.decision_frame.get("action_structure", "")).strip()
            if action_structure not in {"explicit_binary", "implicit_binary"}:
                errors.append("Binary-choice scenario must declare an explicit or implicit binary action structure.")
        return errors


class EthicsWriterRenderer(SimpleTemplateRenderer):
    def __init__(self, *, build_config: EthicsBuildConfig) -> None:
        templates = dict(build_config.profiles.get("templates", {}) or {})
        self._renderers = {
            name: SimpleTemplateRenderer(
                template_name=name,
                template_version=str(config.get("version", "v1")),
                template_text=Path(str(config["path"])).read_text(encoding="utf-8"),
            )
            for name, config in templates.items()
        }
        self._default_template_name = str(
            build_config.profiles.get("default_template_name") or next(iter(self._renderers), "")
        )
        self.build_config = build_config
        super().__init__(
            template_name=self._default_template_name,
            template_version=str(
                (templates.get(self._default_template_name) or {}).get("version", "v1")
            ),
            template_text=Path(
                str((templates.get(self._default_template_name) or {}).get("path", ""))
            ).read_text(encoding="utf-8")
            if self._default_template_name and self._default_template_name in templates
            else "",
        )

    def render(self, spec: RealizationSpec, *, validation_feedback: list[str] | None = None) -> str:
        template_name = str(spec.prompt_template_name or self._default_template_name).strip()
        try:
            renderer = self._renderers[template_name]
        except KeyError as exc:
            raise RuntimeError(f"Unknown ethics synthesis template: {template_name}") from exc
        return renderer.render(spec, validation_feedback=validation_feedback)


class EthicsPromptValidator(RealizationValidator):
    def __init__(self, *, build_config: EthicsBuildConfig) -> None:
        self.build_config = build_config
        template_name = str(build_config.validator.get("template_name") or "realization_validator_v1")
        templates = dict(build_config.validator.get("templates", {}) or {})
        template_config = dict(templates.get(template_name, {}) or {})
        self.template_name = template_name
        self.template_version = str(template_config.get("version", "v1"))
        self.template_text = Path(str(template_config["path"])).read_text(encoding="utf-8")
        self.renderer = SimpleTemplateRenderer(
            template_name=self.template_name,
            template_version=self.template_version,
            template_text=self.template_text,
        )

    def validate(
        self,
        *,
        specs: list[RealizationSpec],
        results: list[RealizationResult],
        request: BenchmarkBuildRequest,
    ) -> list[RealizationValidationResult]:
        if request.execution_mode == "mock":
            return [
                self._validate_mock(spec=spec, result=result)
                for spec, result in zip(specs, results, strict=True)
            ]

        validator_model = str(
            self.build_config.validator.get("model")
            or request.synthesis_model
            or request.llm_model
            or self.build_config.synthesis.get("model")
            or ""
        ).strip()
        if not validator_model:
            raise RuntimeError("Ethics realization validator requires a validator model.")

        progress = request.progress or NullRunProgress()
        out_dir = Path(request.out_dir) if request.out_dir is not None else None
        run_dir = out_dir / "_realization_validation" if out_dir is not None else None
        prompts = [
            PromptRecord(
                prompt_id=f"validate_{index:06d}",
                prompt=self.renderer.render(
                    self._build_validation_spec(spec=spec, result=result)
                ),
                language=spec.language,
                metadata={
                    "benchmark_id": spec.benchmark_id,
                    "case_id": spec.case_id,
                    "template_id": spec.metadata.get("template_id"),
                    "source_builder": spec.source_builder,
                    "validator_template_name": self.template_name,
                },
                parameters=dict(self.build_config.validator.get("generation_defaults", {}) or {}),
            )
            for index, (spec, result) in enumerate(zip(specs, results, strict=True), start=1)
        ]
        requests_path = write_prompt_records_jsonl(
            prompts=prompts,
            output_path=(run_dir / "requests.jsonl") if run_dir is not None else None,
        )
        summary = run_single_model(
            model_name=validator_model,
            prompt_file=requests_path,
            out_dir=run_dir,
            run_name="ethics-realization-validator",
            execution_mode=request.execution_mode,
            progress=progress,
        )
        dataset_records = load_run_dataset_records(summary.run_id)
        raw_by_prompt_id: dict[str, str] = {}
        for record in dataset_records:
            prompt_id = str(record.get("prompt_id", "")).strip()
            if not prompt_id:
                continue
            artifact_path_value = record.get("artifact_path")
            artifact_path = Path(str(artifact_path_value or ""))
            if artifact_path.exists() and artifact_path.is_file():
                raw_by_prompt_id[prompt_id] = artifact_path.read_text(encoding="utf-8")
                continue
            artifact_text = record.get("artifact_text")
            if artifact_text not in (None, ""):
                raw_by_prompt_id[prompt_id] = str(artifact_text)

        validations: list[RealizationValidationResult] = []
        for index, (spec, result) in enumerate(zip(specs, results, strict=True), start=1):
            prompt_id = f"validate_{index:06d}"
            validations.append(
                _parse_validator_output(
                    raw_text=raw_by_prompt_id.get(prompt_id, ""),
                    spec=spec,
                    template_name=self.template_name,
                    template_version=self.template_version,
                    validator_model=validator_model,
                    run_id=summary.run_id,
                    prompt_id=prompt_id,
                )
            )
        return validations

    def _build_validation_spec(self, *, spec: RealizationSpec, result: RealizationResult) -> RealizationSpec:
        prompt_context = dict(spec.prompt_context)
        prompt_context.update(
            {
                "scene_description": str(
                    result.scene_description or result.synthesized_text or ""
                ),
                "decision_frame_json": json.dumps(
                    dict(result.decision_frame or {}),
                    ensure_ascii=False,
                    indent=2,
                ),
                "decision_options_json": json.dumps(
                    list(result.decision_options or []),
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        )
        return RealizationSpec(
            benchmark_id=spec.benchmark_id,
            case_id=spec.case_id,
            source_builder=spec.source_builder,
            input_modality=spec.input_modality,
            language=spec.language,
            split=spec.split,
            tags=list(spec.tags),
            parameters=dict(spec.parameters),
            metadata=dict(spec.metadata),
            grouping=dict(spec.grouping),
            execution_hints=dict(spec.execution_hints),
            evaluation_hints=dict(spec.evaluation_hints),
            expected_output_contract=spec.expected_output_contract,
            slot_assignments=dict(spec.slot_assignments),
            invariants=list(spec.invariants),
            forbidden_transformations=list(spec.forbidden_transformations),
            prompt_template_name=self.template_name,
            prompt_template_version=self.template_version,
            synthesis_model=result.synthesis_model or spec.synthesis_model,
            synthesis_request_version=result.synthesis_request_version or spec.synthesis_request_version,
            prompt_context=prompt_context,
        )

    def _validate_mock(
        self,
        *,
        spec: RealizationSpec,
        result: RealizationResult,
    ) -> RealizationValidationResult:
        text = str(result.scene_description or result.synthesized_text or "").strip()
        lowered = text.lower()
        issues: list[str] = []
        if any(term in lowered for term in ("benchmark", "philosophy", "thought experiment")):
            issues.append("Scene reads like a benchmark or abstract thought experiment.")
        if "imagine you are" in lowered or "if you were" in lowered:
            issues.append("Scene uses roleplay framing instead of a lived decision context.")
        if not _looks_second_person(text):
            issues.append("Scene is not written as an immersive second-person brief.")
        if not any(term in lowered for term in ("need to decide", "must decide", "decide what to do")):
            issues.append("Scene does not feel like a live decision context.")
        issues.extend(_validate_decision_options(result.decision_options))
        if _binary_frame_required(spec) and not result.decision_frame:
            issues.append("Missing decision_frame for a binary-choice scenario.")
        return RealizationValidationResult(
            benchmark_id=spec.benchmark_id,
            case_id=spec.case_id,
            valid=not issues,
            issues=list(issues),
            feedback_for_retry=list(issues),
            binary_frame_assessment={
                "explicit_binary_required": _binary_frame_required(spec),
                "realized_structure": str(result.decision_frame.get("action_structure", "")),
                "option_count": len(result.decision_options or []),
            },
            conflict_preservation_assessment={
                "status": "looks_preserved" if not issues else "needs_revision"
            },
            metadata={
                "mode": "mock",
                "validator_template_name": self.template_name,
                "validator_template_version": self.template_version,
            },
        )


class EthicsCaseCompiler(CaseCompiler):
    def __init__(
        self,
        *,
        package: GenerativeBenchmarkPackage,
        build_config: EthicsBuildConfig,
        build_mode: str,
        seed: int,
    ) -> None:
        self.package = package
        self.build_config = build_config
        self.build_mode = build_mode
        self.seed = seed

    def compile(self, spec: RealizationSpec, result: RealizationResult) -> BenchmarkCase:
        metadata = dict(spec.metadata)
        metadata.update(
            {
                "slot_assignments": dict(spec.slot_assignments),
                "invariants": list(spec.invariants),
                "forbidden_transformations": list(spec.forbidden_transformations),
                "decision_frame": dict(result.decision_frame or {}),
                "decision_options": list(result.decision_options or []),
                "realization_prompt_template": result.prompt_template_name or spec.prompt_template_name,
                "synthesis_model": result.synthesis_model or spec.synthesis_model,
                "synthesis_request_version": (
                    result.synthesis_request_version or spec.synthesis_request_version
                ),
                "validator_template_name": str(
                    dict(result.metadata.get("validator", {}) or {}).get(
                        "metadata",
                        {},
                    ).get("validator_template_name", "")
                ),
                "validator_template_version": str(
                    dict(result.metadata.get("validator", {}) or {}).get(
                        "metadata",
                        {},
                    ).get("validator_template_version", "")
                ),
                "validator_model": str(
                    dict(result.metadata.get("validator", {}) or {}).get(
                        "metadata",
                        {},
                    ).get("validator_model", "")
                ),
                "realization_provenance": {
                    "builder": spec.source_builder,
                    "build_mode": self.build_mode,
                    "seed": self.seed,
                    "prompt_template_name": result.prompt_template_name or spec.prompt_template_name,
                    "prompt_template_version": (
                        result.prompt_template_version or spec.prompt_template_version
                    ),
                    "synthesis_model": result.synthesis_model or spec.synthesis_model,
                    "synthesis_request_version": (
                        result.synthesis_request_version or spec.synthesis_request_version
                    ),
                    "package_path": self.package.package_path,
                    "canonical_package_path": self.package.canonical_package_path,
                    "attempt": result.metadata.get("attempt"),
                    "request_run_id": result.metadata.get("run_id"),
                    "request_prompt_id": result.metadata.get("request_prompt_id"),
                    "validator": dict(result.metadata.get("validator", {}) or {}),
                },
            }
        )
        if result.validation_errors:
            metadata["realization_validation_errors"] = list(result.validation_errors)
        return BenchmarkCase(
            benchmark_id=spec.benchmark_id,
            case_id=spec.case_id,
            input_type="text",
            input_modality="text",
            input_payload={
                "prompt": result.scene_description or result.synthesized_text,
                "language": spec.language,
                "decision_options": list(result.decision_options or []),
            },
            prompt=result.scene_description or result.synthesized_text,
            instruction=None,
            metadata=metadata,
            tags=list(spec.tags),
            split=spec.split,
            context=None,
            expected_output_contract=spec.expected_output_contract,
            expected_structure=spec.expected_output_contract,
            case_version=str(spec.metadata.get("family_version") or ""),
            source_builder=spec.source_builder,
            language=spec.language,
            parameters=dict(spec.parameters),
            grouping=dict(spec.grouping),
            execution_hints=dict(spec.execution_hints),
            evaluation_hints=dict(spec.evaluation_hints),
        )


def load_ethics_template_package(package_path: str | Path) -> GenerativeBenchmarkPackage:
    return load_generative_benchmark_package(package_path)


def _sample_slot_assignments(
    *,
    template: dict[str, Any],
    slot_library: dict[str, SlotDefinition],
    random: Random,
    variant_index: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    slot_assignments: dict[str, Any] = {}
    slot_layers: dict[str, str] = {}
    used_slots = dict(template.get("used_slots", {}) or {})
    for layer_name in ("structural", "narrative", "perturbation"):
        for entry in used_slots.get(layer_name, []) or []:
            slot_id = str((entry or {}).get("slot_id") if isinstance(entry, dict) else entry).strip()
            if not slot_id:
                continue
            try:
                slot_definition = slot_library[slot_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"Template {template.get('template_id', 'template')} references unknown slot: {slot_id}"
                ) from exc
            slot_layers[slot_id] = layer_name
            slot_assignments[slot_id] = _sample_slot_value(
                slot_definition=slot_definition,
                random=random,
                variant_index=variant_index,
            )
    return slot_assignments, slot_layers


def _sample_slot_value(
    *,
    slot_definition: SlotDefinition,
    random: Random,
    variant_index: int,
) -> Any:
    value_space = dict(slot_definition.value_space or {})
    kind = str(value_space.get("kind") or "enum")
    if kind == "boolean":
        values = list(value_space.get("values", [False, True]) or [False, True])
        return values[variant_index % len(values)]
    if kind == "integer_range":
        start = int(value_space.get("min", 0))
        end = int(value_space.get("max", start))
        return start if start == end else start + (variant_index % (end - start + 1))
    if kind == "float_range":
        start = float(value_space.get("min", 0.0))
        end = float(value_space.get("max", start))
        if start == end:
            return start
        return random.uniform(start, end)
    if kind == "enum":
        normalized_values = [_normalize_value_entry(item) for item in value_space.get("values", []) or []]
        if not normalized_values:
            return None
        return normalized_values[variant_index % len(normalized_values)]
    return None


def _normalize_value_entry(value: Any) -> Any:
    if isinstance(value, dict):
        if "id" in value:
            return value["id"]
        if "value" in value:
            return value["value"]
    return value


def _load_ethics_build_config(path: str | Path | None) -> EthicsBuildConfig:
    base_dir = Path(path).resolve().parent if path is not None else Path(__file__).resolve().parent
    example_dir = Path(__file__).resolve().parent
    payload = load_yaml_file(Path(path)) if path is not None else {}
    sampling = dict(payload.get("sampling", {}) or {})
    if "realizations_per_template" in payload and "realizations_per_template" not in sampling:
        sampling["realizations_per_template"] = payload["realizations_per_template"]
    sampling.setdefault("realizations_per_template", 1)

    default_template_path = example_dir / "synthesis_templates" / "standard_naturalistic_v1.txt"
    default_validator_template_path = example_dir / "synthesis_templates" / "realization_validator_v1.txt"
    profiles = dict(payload.get("profiles", {}) or {})
    profiles.setdefault("default_template_name", "standard_naturalistic_v1")
    raw_templates = dict(profiles.get("templates", {}) or {})
    if not raw_templates:
        raw_templates = {
            "standard_naturalistic_v1": {
                "path": str(default_template_path),
                "version": "v1",
            }
        }
    resolved_templates: dict[str, dict[str, Any]] = {}
    for name, config in raw_templates.items():
        config_dict = dict(config or {})
        template_path = Path(str(config_dict.get("path") or default_template_path))
        if not template_path.is_absolute():
            template_path = (base_dir / template_path).resolve()
        resolved_templates[str(name)] = {
            **config_dict,
            "path": str(template_path),
            "version": str(config_dict.get("version", "v1")),
        }
    profiles["templates"] = resolved_templates

    synthesis = dict(payload.get("synthesis", {}) or {})
    synthesis.setdefault("model", payload.get("llm_model"))
    synthesis.setdefault("generation_defaults", {})
    synthesis.setdefault("request_version", "v1")

    validation = dict(payload.get("validation", {}) or {})
    validation.setdefault("max_attempts", 2)
    validation.setdefault("min_word_count", 24)
    validation.setdefault("forbidden_terms", list(_DEFAULT_FORBIDDEN_TERMS))
    validation.setdefault("required_slot_mentions", ["decision_maker_role", "setting_domain"])

    validator = dict(payload.get("validator", {}) or {})
    validator.setdefault("enabled", True)
    validator.setdefault("template_name", "realization_validator_v1")
    validator_templates = dict(validator.get("templates", {}) or {})
    if not validator_templates and bool(validator.get("enabled", True)):
        validator_templates = {
            "realization_validator_v1": {
                "path": str(default_validator_template_path),
                "version": "v1",
            }
        }
    resolved_validator_templates: dict[str, dict[str, Any]] = {}
    for name, config in validator_templates.items():
        config_dict = dict(config or {})
        template_path = Path(str(config_dict.get("path") or default_validator_template_path))
        if not template_path.is_absolute():
            template_path = (base_dir / template_path).resolve()
        resolved_validator_templates[str(name)] = {
            **config_dict,
            "path": str(template_path),
            "version": str(config_dict.get("version", "v1")),
        }
    validator["templates"] = resolved_validator_templates
    validator.setdefault("model", synthesis.get("model"))
    validator.setdefault("generation_defaults", {})

    return EthicsBuildConfig(
        sampling=sampling,
        profiles=profiles,
        synthesis=synthesis,
        validation=validation,
        validator=validator,
        raw=payload,
    )


def _apply_request_synthesis_defaults(
    request: BenchmarkBuildRequest,
    build_config: EthicsBuildConfig,
) -> BenchmarkBuildRequest:
    if request.synthesis_model not in (None, ""):
        return request
    if build_config.synthesis.get("model") in (None, ""):
        return request
    return BenchmarkBuildRequest(
        builder_name=request.builder_name,
        source_path=request.source_path,
        out_dir=request.out_dir,
        benchmark_name=request.benchmark_name,
        seed=request.seed,
        build_mode=request.build_mode,
        builder_config_path=request.builder_config_path,
        count_config_path=request.count_config_path,
        llm_model=request.llm_model,
        synthesis_model=str(build_config.synthesis.get("model")),
        execution_mode=request.execution_mode,
        profile_path=request.profile_path,
        template_name=request.template_name,
        style_family_name=request.style_family_name,
        target_model_name=request.target_model_name,
        intended_modality=request.intended_modality,
        entrypoint=request.entrypoint,
        progress=request.progress,
    )


def _resolve_prompt_template_name(
    *,
    template: dict[str, Any],
    build_config: EthicsBuildConfig,
    request: BenchmarkBuildRequest,
) -> str:
    if request.template_name not in (None, ""):
        return str(request.template_name)
    guidelines = dict(template.get("prompt_generation_guidelines", {}) or {})
    profile_id = guidelines.get("profile_id")
    if profile_id not in (None, ""):
        return str(profile_id)
    return str(build_config.profiles.get("default_template_name") or "standard_naturalistic_v1")


def _resolve_prompt_template_version(
    *,
    prompt_template_name: str,
    build_config: EthicsBuildConfig,
) -> str:
    template_config = dict((build_config.profiles.get("templates") or {}).get(prompt_template_name, {}) or {})
    return str(template_config.get("version", "v1"))


def _resolve_validator_template_version(*, build_config: EthicsBuildConfig) -> str:
    template_name = str(build_config.validator.get("template_name") or "")
    template_config = dict((build_config.validator.get("templates") or {}).get(template_name, {}) or {})
    return str(template_config.get("version", "v1"))


def _validator_enabled(build_config: EthicsBuildConfig) -> bool:
    return bool(build_config.validator.get("enabled", True))


def _resolve_profile_vars(
    *,
    prompt_template_name: str,
    build_config: EthicsBuildConfig,
) -> dict[str, Any]:
    template_config = dict((build_config.profiles.get("templates") or {}).get(prompt_template_name, {}) or {})
    return dict(template_config.get("profile_vars", {}) or {})


def _infer_decision_frame_requirements(
    *,
    template: dict[str, Any],
    prompt_guidelines: dict[str, Any],
    profile_vars: dict[str, Any],
) -> dict[str, Any]:
    constraints = [str(item).lower() for item in prompt_guidelines.get("constraints", []) or []]
    response_capture = dict(template.get("response_capture", {}) or {})
    explicit_binary_required = any(
        phrase in constraint
        for constraint in constraints
        for phrase in (
            "explicit binary",
            "two admissible actions",
            "exactly two actions",
            "binary decision framing",
        )
    ) or bool(profile_vars.get("explicit_binary_required", False))
    explicit_option_format_required = any(
        phrase in constraint
        for constraint in constraints
        for phrase in ("option a", "option b", "a/b")
    ) or bool(profile_vars.get("explicit_option_format_required", False))
    return {
        "explicit_binary_required": explicit_binary_required,
        "explicit_option_format_required": explicit_option_format_required,
        "binary_decision_policy": str(
            profile_vars.get("binary_decision_policy", "infer_from_constraints")
        ),
        "response_capture_type": str(response_capture.get("type", "")),
    }


def _binary_frame_required(spec: RealizationSpec) -> bool:
    return bool(
        dict(spec.metadata.get("decision_frame_requirements", {}) or {}).get(
            "explicit_binary_required",
            False,
        )
    )


def _looks_second_person(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        marker in lowered
        for marker in ("you ", "you\n", "you're", "you've", "your ", "your\n")
    ) or lowered.startswith("you")


def _validate_decision_options(options: list[dict[str, Any]] | None) -> list[str]:
    normalized = list(options or [])
    if len(normalized) != 2:
        return ["Realization must contain exactly two structured decision options."]
    option_ids = [str(item.get("id", "")).strip().upper() for item in normalized]
    if option_ids != ["A", "B"]:
        return ["Decision options must use stable ids A and B in order."]
    errors: list[str] = []
    for item in normalized:
        text = str(item.get("text", "")).strip()
        if not text:
            errors.append("Decision options must contain non-empty text.")
            continue
        lowered = text.lower()
        if "option c" in lowered or "third option" in lowered or "another option" in lowered:
            errors.append("Decision options must not imply a third path.")
    return errors


def _render_bullet_block(values: list[Any]) -> str:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        return ""
    return "\n".join(f"- {value}" for value in normalized)


def _parse_validator_output(
    *,
    raw_text: str,
    spec: RealizationSpec,
    template_name: str,
    template_version: str,
    validator_model: str,
    run_id: str,
    prompt_id: str,
) -> RealizationValidationResult:
    stripped = str(raw_text).strip()
    issues = ["Validator returned empty output."]
    payload: dict[str, Any] = {}
    if stripped:
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            issues = ["Validator returned non-JSON output."]
        else:
            if isinstance(decoded, dict):
                payload = decoded
                raw_issues = payload.get("issues", [])
                if isinstance(raw_issues, list):
                    issues = [str(item) for item in raw_issues if str(item).strip()]
                else:
                    issues = []
            else:
                issues = ["Validator returned an unexpected payload shape."]
    valid = bool(payload.get("valid")) if payload else False
    if valid:
        issues = []
    feedback = payload.get("feedback_for_retry", issues)
    if not isinstance(feedback, list):
        feedback = issues
    return RealizationValidationResult(
        benchmark_id=spec.benchmark_id,
        case_id=spec.case_id,
        valid=valid,
        issues=list(issues),
        feedback_for_retry=[str(item) for item in feedback if str(item).strip()],
        binary_frame_assessment=dict(payload.get("binary_frame_assessment", {}) or {}),
        conflict_preservation_assessment=dict(
            payload.get("conflict_preservation_assessment", {}) or {}
        ),
        metadata={
            "validator_template_name": template_name,
            "validator_template_version": template_version,
            "validator_model": validator_model,
            "validator_run_id": run_id,
            "validator_prompt_id": prompt_id,
        },
    )
