from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from aigc.benchmarking.interfaces import (
    BenchmarkBuildOutput,
    BenchmarkBuildRequest,
    BenchmarkBuilder,
    CaseCompiler,
    GroupAnalysisRequest,
    GroupAnalyzer,
    ParameterSampler,
    StructureGuard,
)
from aigc.benchmarking.models import BenchmarkCase, RealizationResult, RealizationSpec
from aigc.benchmarking.packages import GenerativeBenchmarkPackage, SlotDefinition, load_generative_benchmark_package
from aigc.benchmarking.realization import (
    SimpleTemplateRenderer,
    execute_semantic_realization_pipeline,
    validate_forbidden_terms,
    validate_required_value_mentions,
)
from aigc.benchmarking.service import load_yaml_file, slugify

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
        renderer = EthicsRealizationRenderer(build_config=build_config)
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
                    "prompt_templates": {
                        name: {
                            "version": str(config.get("version", "v1")),
                        }
                        for name, config in (build_config.profiles.get("templates") or {}).items()
                    },
                    "validation": {
                        "max_attempts": int(build_config.validation.get("max_attempts", 1)),
                        "forbidden_terms": list(build_config.validation.get("forbidden_terms", [])),
                    },
                },
            },
            build_artifacts=pipeline_output.build_artifacts,
            build_manifest_overrides=pipeline_output.build_manifest_overrides,
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
                            "merge_rationale": str(template.get("merge_rationale", "") or ""),
                            "source_cases_json": json.dumps(
                                list(template.get("source_cases", []) or []),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "ethics_grounding_json": json.dumps(
                                dict(template.get("ethics_grounding", {}) or {}),
                                ensure_ascii=False,
                                indent=2,
                            ),
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
        text = str(result.synthesized_text or "").strip()
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
        return errors


class EthicsRealizationRenderer(SimpleTemplateRenderer):
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
                "realization_prompt_template": result.prompt_template_name or spec.prompt_template_name,
                "synthesis_model": result.synthesis_model or spec.synthesis_model,
                "synthesis_request_version": (
                    result.synthesis_request_version or spec.synthesis_request_version
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
            prompt=result.synthesized_text,
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

    return EthicsBuildConfig(
        sampling=sampling,
        profiles=profiles,
        synthesis=synthesis,
        validation=validation,
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


def _render_bullet_block(values: list[Any]) -> str:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        return ""
    return "\n".join(f"- {value}" for value in normalized)
