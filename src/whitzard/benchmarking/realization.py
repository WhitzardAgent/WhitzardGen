from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from whitzard.benchmarking.interfaces import (
    BenchmarkBuildRequest,
    CaseCompiler,
    ParameterSampler,
    RealizationSynthesisBackend,
    RealizationTemplateRenderer,
    RealizationValidator,
    StructureGuard,
)
from whitzard.benchmarking.models import (
    BenchmarkCase,
    RealizationResult,
    RealizationSpec,
    RealizationValidationResult,
)
from whitzard.benchmarking.prompt_io import write_prompt_records_jsonl
from whitzard.prompt_generation.config import render_instruction_template
from whitzard.prompts.models import PromptRecord
from whitzard.run_flow import run_single_model
from whitzard.run_store import load_run_dataset_records
from whitzard.structured_io import build_json_object_output_spec, parse_structured_output, resolve_output_spec
from whitzard.utils.progress import NullRunProgress

_FORBIDDEN_TERM_PATTERNS = {
    "test": re.compile(r"\btest\b", re.IGNORECASE),
    "benchmark": re.compile(r"\bbenchmark\b", re.IGNORECASE),
    "dilemma": re.compile(r"\bdilemma\b", re.IGNORECASE),
    "philosophy": re.compile(r"\bphilosophy\b", re.IGNORECASE),
    "option a": re.compile(r"\boption\s*a\b", re.IGNORECASE),
    "option b": re.compile(r"\boption\s*b\b", re.IGNORECASE),
}


class SemanticRealizationError(RuntimeError):
    """Raised when a semantic benchmark realization fails."""


def execute_semantic_realization_pipeline(
    *,
    request: BenchmarkBuildRequest,
    sampler: ParameterSampler,
    guard: StructureGuard,
    renderer: RealizationTemplateRenderer,
    compiler: CaseCompiler,
    synthesis_backend: RealizationSynthesisBackend | None = None,
    validator: RealizationValidator | None = None,
    max_attempts: int = 1,
) -> BenchmarkBuildOutputLike:
    progress = request.progress or NullRunProgress()
    specs = sampler.sample(request)
    if not specs:
        raise SemanticRealizationError("Semantic realization pipeline produced no specs.")

    for spec in specs:
        validation_errors = _dedupe_errors(guard.validate_spec(spec))
        if validation_errors:
            raise SemanticRealizationError(
                f"Realization spec validation failed for {spec.case_id}: {'; '.join(validation_errors)}"
            )

    backend = synthesis_backend or RunKernelRealizationSynthesisBackend()
    max_attempts = max(int(max_attempts), 1)
    pending_specs = list(specs)
    feedback_by_case_id: dict[str, list[str]] = {}
    final_results: dict[str, RealizationResult] = {}
    validation_failures: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        if not pending_specs:
            break
        progress.env_message(
            f"[benchmark-build] semantic realization attempt {attempt}/{max_attempts} "
            f"for {len(pending_specs)} cases"
        )
        batch_results = backend.synthesize(
            specs=pending_specs,
            renderer=renderer,
            request=request,
            validation_feedback_by_case_id=feedback_by_case_id,
        )
        if len(batch_results) != len(pending_specs):
            raise SemanticRealizationError(
                f"Semantic realization backend returned {len(batch_results)} results for "
                f"{len(pending_specs)} specs."
            )

        next_pending: list[RealizationSpec] = []
        validation_records: list[RealizationValidationResult] = []
        if validator is not None:
            validation_records = validator.validate(
                specs=pending_specs,
                results=batch_results,
                request=request,
            )
            if len(validation_records) != len(pending_specs):
                raise SemanticRealizationError(
                    f"Semantic realization validator returned {len(validation_records)} results for "
                    f"{len(pending_specs)} specs."
                )
        else:
            validation_records = [
                RealizationValidationResult(
                    benchmark_id=spec.benchmark_id,
                    case_id=spec.case_id,
                    valid=True,
                )
                for spec in pending_specs
            ]
        for spec, result, validation_record in zip(
            pending_specs,
            batch_results,
            validation_records,
            strict=True,
        ):
            result.metadata.setdefault("attempt", attempt)
            result.metadata["validator"] = validation_record.to_dict()
            result.validation_errors = _dedupe_errors(
                list(result.validation_errors)
                + list(guard.validate_realization(spec, result))
                + list(validation_record.issues)
            )
            if result.validation_errors and attempt < max_attempts:
                retry_feedback = list(validation_record.feedback_for_retry) or list(result.validation_errors)
                feedback_by_case_id[spec.case_id] = _dedupe_errors(retry_feedback)
                next_pending.append(spec)
                continue

            if result.validation_errors:
                validation_failures.append(
                    {
                        "case_id": spec.case_id,
                        "attempt": attempt,
                        "errors": list(result.validation_errors),
                    }
                )
            final_results[spec.case_id] = result
        pending_specs = next_pending

    if pending_specs:
        unresolved = ", ".join(spec.case_id for spec in pending_specs)
        raise SemanticRealizationError(
            f"Semantic realization exhausted retries but did not finalize all cases: {unresolved}"
        )

    valid_specs: list[RealizationSpec] = []
    invalid_specs: list[RealizationSpec] = []
    raw_realizations: list[dict[str, Any]] = []
    rejected_realizations: list[dict[str, Any]] = []
    for spec in specs:
        result = final_results[spec.case_id]
        validation_errors = list(result.validation_errors)
        payload = {
            "benchmark_id": spec.benchmark_id,
            "case_id": spec.case_id,
            "valid": not validation_errors,
            "spec": spec.to_dict(),
            "result": result.to_dict(),
            "validation_errors": validation_errors,
        }
        raw_realizations.append(payload)
        if validation_errors:
            invalid_specs.append(spec)
            rejected_realizations.append(payload)
            continue
        valid_specs.append(spec)

    cases = [compiler.compile(spec, final_results[spec.case_id]) for spec in valid_specs]
    progress.env_message(
        f"[benchmark-build] semantic realization produced {len(cases)} valid cases "
        f"and {len(rejected_realizations)} rejected cases"
    )
    build_artifacts = {
        "realization_case_count": len(cases),
        "realization_valid_case_count": len(cases),
        "realization_rejected_case_count": len(rejected_realizations),
        "realization_raw_case_count": len(raw_realizations),
        "realization_attempt_count": max_attempts,
        "realization_validation_failure_count": len(validation_failures),
        "realization_validation_failures": validation_failures,
        "realization_validator_enabled": validator is not None,
    }
    return BenchmarkBuildOutputLike(
        cases=cases,
        build_artifacts=build_artifacts,
        extra_jsonl_files={
            "raw_realizations.jsonl": raw_realizations,
            "rejected_realizations.jsonl": rejected_realizations,
        },
    )


class SimpleTemplateRenderer(RealizationTemplateRenderer):
    def __init__(
        self,
        *,
        template_name: str,
        template_version: str,
        template_text: str,
        base_values: dict[str, Any] | None = None,
    ) -> None:
        self.template_name = template_name
        self.template_version = template_version
        self.template_text = template_text
        self.base_values = dict(base_values or {})

    def render(self, spec: RealizationSpec, *, validation_feedback: list[str] | None = None) -> str:
        values = dict(self.base_values)
        values.update(dict(spec.prompt_context))
        values.update(
            {
                "benchmark_id": spec.benchmark_id,
                "case_id": spec.case_id,
                "template_id": spec.metadata.get("template_id", ""),
                "family_id": spec.metadata.get("family_id", ""),
                "variant_group_id": spec.grouping.get("variant_group_id", ""),
                "slot_assignments_json": json.dumps(spec.slot_assignments, ensure_ascii=False, indent=2),
                "deep_structure_json": json.dumps(
                    spec.metadata.get("deep_structure", {}),
                    ensure_ascii=False,
                    indent=2,
                ),
                "invariants_block": _render_bullet_block(spec.invariants),
                "forbidden_transformations_block": _render_bullet_block(
                    spec.forbidden_transformations
                ),
                "analysis_targets_block": _render_bullet_block(
                    spec.metadata.get("analysis_targets", [])
                ),
                "response_capture_contract_json": json.dumps(
                    spec.metadata.get("response_capture_contract", {}),
                    ensure_ascii=False,
                    indent=2,
                ),
                "profile_template_name": spec.prompt_template_name or "",
                "validation_feedback_block": _render_bullet_block(validation_feedback or []),
            }
        )
        return render_instruction_template(self.template_text, values=values)


class RunKernelRealizationSynthesisBackend(RealizationSynthesisBackend):
    def synthesize(
        self,
        *,
        specs: list[RealizationSpec],
        renderer: RealizationTemplateRenderer,
        request: BenchmarkBuildRequest,
        validation_feedback_by_case_id: dict[str, list[str]] | None = None,
    ) -> list[RealizationResult]:
        feedback_by_case_id = dict(validation_feedback_by_case_id or {})
        if request.execution_mode == "mock":
            return [
                _build_mock_realization_result(
                    spec=spec,
                    request_prompt=renderer.render(
                        spec,
                        validation_feedback=feedback_by_case_id.get(spec.case_id),
                    ),
                )
                for spec in specs
            ]

        synthesis_model = str(request.synthesis_model or request.llm_model or "").strip()
        if not synthesis_model:
            raise SemanticRealizationError(
                "Semantic realization requires a synthesis model. "
                "Pass --synthesis-model/--llm-model or configure synthesis.model in the builder config."
            )

        out_dir = Path(request.out_dir) if request.out_dir is not None else None
        run_dir = out_dir / "_realization_synthesis" if out_dir is not None else None
        request_prompts = [
            PromptRecord(
                prompt_id=f"realize_{index:06d}",
                prompt=renderer.render(
                    spec,
                    validation_feedback=feedback_by_case_id.get(spec.case_id),
                ),
                language=spec.language,
                metadata={
                    "benchmark_id": spec.benchmark_id,
                    "case_id": spec.case_id,
                    "template_id": spec.metadata.get("template_id"),
                    "family_id": spec.metadata.get("family_id"),
                    "variant_group_id": spec.grouping.get("variant_group_id"),
                    "source_builder": spec.source_builder,
                },
                parameters=dict(spec.parameters),
            )
            for index, spec in enumerate(specs, start=1)
        ]
        requests_path = write_prompt_records_jsonl(
            prompts=request_prompts,
            output_path=(run_dir / "requests.jsonl") if run_dir is not None else None,
        )
        summary = run_single_model(
            model_name=synthesis_model,
            prompt_file=requests_path,
            out_dir=run_dir,
            run_name=f"{request.builder_name}-semantic-realization",
            execution_mode=request.execution_mode,
            progress=request.progress,
        )
        dataset_records = load_run_dataset_records(summary.run_id)
        text_by_prompt_id: dict[str, str] = {}
        for record in dataset_records:
            prompt_id = str(record.get("prompt_id", "")).strip()
            if not prompt_id:
                continue
            artifact_path_value = record.get("artifact_path")
            artifact_path = Path(str(artifact_path_value or ""))
            if artifact_path.exists() and artifact_path.is_file():
                text_by_prompt_id[prompt_id] = artifact_path.read_text(encoding="utf-8")
                continue
            text_candidate = record.get("artifact_text")
            if text_candidate not in (None, ""):
                text_by_prompt_id[prompt_id] = str(text_candidate)

        results: list[RealizationResult] = []
        for index, spec in enumerate(specs, start=1):
            prompt_id = f"realize_{index:06d}"
            request_prompt = request_prompts[index - 1].prompt
            raw_text = text_by_prompt_id.get(prompt_id, "")
            scene_description, structured_output, decision_frame, decision_options = _parse_synthesized_output(
                raw_text,
                output_spec=dict(spec.metadata.get("writer_output_spec", {}) or {}),
            )
            results.append(
                RealizationResult(
                    benchmark_id=spec.benchmark_id,
                    case_id=spec.case_id,
                    source_builder=spec.source_builder,
                    synthesized_text=scene_description,
                    scene_description=scene_description,
                    structured_output=structured_output,
                    decision_frame=decision_frame,
                    decision_options=decision_options,
                    prompt_template_name=spec.prompt_template_name,
                    prompt_template_version=spec.prompt_template_version,
                    synthesis_model=spec.synthesis_model or synthesis_model,
                    synthesis_request_version=spec.synthesis_request_version,
                    request_prompt=request_prompt,
                    validation_errors=[] if raw_text.strip() else ["Synthesis model returned empty text."],
                    metadata={
                        "run_id": summary.run_id,
                        "request_prompt_id": prompt_id,
                    },
                )
            )
        return results


def validate_forbidden_terms(text: str, forbidden_terms: list[str]) -> list[str]:
    errors: list[str] = []
    lowered = text.lower()
    for term in forbidden_terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        pattern = _FORBIDDEN_TERM_PATTERNS.get(normalized.lower())
        if pattern is not None:
            if pattern.search(text):
                errors.append(f"Forbidden exposed term found: {normalized}")
            continue
        if normalized.lower() in lowered:
            errors.append(f"Forbidden exposed term found: {normalized}")
    return errors


def validate_required_value_mentions(text: str, values: list[str]) -> list[str]:
    errors: list[str] = []
    lowered = text.lower()
    for value in values:
        normalized = str(value).strip().replace("_", " ").lower()
        if not normalized:
            continue
        if normalized not in lowered:
            errors.append(f"Expected realization to mention: {normalized}")
    return errors


def _normalize_decision_options(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    options: list[dict[str, Any]] = []
    for index, item in enumerate(value[:2], start=1):
        option_id = "A" if index == 1 else "B"
        text = ""
        if isinstance(item, dict):
            raw_id = str(item.get("id") or option_id).strip().upper()
            option_id = raw_id or option_id
            text = str(item.get("text") or "").strip()
        else:
            text = str(item).strip()
        if option_id not in {"A", "B"} or not text:
            return []
        options.append({"id": option_id, "text": text})
    if len(options) != 2:
        return []
    if {item["id"] for item in options} != {"A", "B"}:
        return []
    return sorted(options, key=lambda item: item["id"])


def _parse_synthesized_output(
    raw_text: str,
    *,
    output_spec: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    stripped = str(raw_text).strip()
    if not stripped:
        return "", {}, {}, []
    spec = resolve_output_spec(output_spec) if output_spec else build_json_object_output_spec()
    parsed = parse_structured_output(stripped, output_spec=spec)
    if isinstance(parsed.raw_payload, dict):
        payload = dict(parsed.fields)
        structured_output = dict(payload)
        decision_frame = dict(payload.get("decision_frame", {}) or {})
        decision_options = _normalize_decision_options(payload.get("decision_options"))
        for key in (
            "scene_description",
            "realized_prompt",
            "scenario",
            "scene",
            "prompt",
            "text",
        ):
            value = payload.get(key)
            if value not in (None, ""):
                return str(value).strip(), structured_output, decision_frame, decision_options
        return stripped, structured_output, decision_frame, decision_options
    return stripped, {}, {}, []


def _build_mock_realization_result(*, spec: RealizationSpec, request_prompt: str) -> RealizationResult:
    role = str(
        spec.slot_assignments.get("decision_maker_role")
        or spec.slot_assignments.get("role")
        or "person"
    ).replace("_", " ")
    domain = str(spec.slot_assignments.get("setting_domain") or "real-world setting").replace(
        "_", " "
    )
    conflict = str(spec.metadata.get("key_moral_conflict") or "").strip()
    synthesized_text = (
        f"You are in a difficult situation in {domain}. "
        f"As the {role}, you need to decide what to do next. "
        f"{conflict}"
    ).strip()
    decision_frame = {
        "explicit_binary_required": bool(
            spec.metadata.get("decision_frame_requirements", {}).get("explicit_binary_required", False)
        ),
        "action_structure": "explicit_binary",
    }
    decision_options = [
        {"id": "A", "text": "Take the first admissible action available to you in this situation."},
        {"id": "B", "text": "Take the second admissible action available to you in this situation."},
    ]
    return RealizationResult(
        benchmark_id=spec.benchmark_id,
        case_id=spec.case_id,
        source_builder=spec.source_builder,
        synthesized_text=synthesized_text,
        scene_description=synthesized_text,
        structured_output={
            "scene_description": synthesized_text,
            "decision_frame": decision_frame,
            "decision_options": decision_options,
        },
        decision_frame=decision_frame,
        decision_options=decision_options,
        prompt_template_name=spec.prompt_template_name,
        prompt_template_version=spec.prompt_template_version,
        synthesis_model=spec.synthesis_model,
        synthesis_request_version=spec.synthesis_request_version,
        request_prompt=request_prompt,
        metadata={"mode": "mock"},
    )


def _render_bullet_block(values: list[Any]) -> str:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        return ""
    return "\n".join(f"- {value}" for value in normalized)


def _dedupe_errors(errors: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for error in errors:
        normalized = str(error).strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


class BenchmarkBuildOutputLike:
    def __init__(
        self,
        *,
        cases: list[BenchmarkCase],
        build_artifacts: dict[str, Any] | None = None,
        build_manifest_overrides: dict[str, Any] | None = None,
        extra_jsonl_files: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.cases = cases
        self.build_artifacts = dict(build_artifacts or {})
        self.build_manifest_overrides = dict(build_manifest_overrides or {})
        self.extra_jsonl_files = dict(extra_jsonl_files or {})
