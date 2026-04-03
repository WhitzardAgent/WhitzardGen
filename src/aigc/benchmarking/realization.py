from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aigc.benchmarking.interfaces import (
    BenchmarkBuildRequest,
    CaseCompiler,
    ParameterSampler,
    RealizationSynthesisBackend,
    RealizationTemplateRenderer,
    StructureGuard,
)
from aigc.benchmarking.models import BenchmarkCase, RealizationResult, RealizationSpec
from aigc.prompt_generation.config import render_instruction_template
from aigc.prompts.models import PromptRecord
from aigc.run_flow import run_single_model
from aigc.run_store import load_run_dataset_records
from aigc.utils.progress import NullRunProgress

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
        for spec, result in zip(pending_specs, batch_results, strict=True):
            result.metadata.setdefault("attempt", attempt)
            result.validation_errors = _dedupe_errors(
                list(result.validation_errors) + list(guard.validate_realization(spec, result))
            )
            if result.validation_errors and attempt < max_attempts:
                feedback_by_case_id[spec.case_id] = list(result.validation_errors)
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

    invalid_results = [result for result in final_results.values() if result.validation_errors]
    if invalid_results:
        details = "; ".join(
            f"{result.case_id}: {', '.join(result.validation_errors)}" for result in invalid_results[:5]
        )
        raise SemanticRealizationError(
            "Semantic realization produced invalid outputs after retry budget was exhausted: "
            f"{details}"
        )

    cases = [compiler.compile(spec, final_results[spec.case_id]) for spec in specs]
    progress.env_message(f"[benchmark-build] semantic realization produced {len(cases)} cases")
    build_artifacts = {
        "realization_case_count": len(cases),
        "realization_attempt_count": max_attempts,
        "realization_validation_failure_count": len(validation_failures),
        "realization_validation_failures": validation_failures,
    }
    return BenchmarkBuildOutputLike(cases=cases, build_artifacts=build_artifacts)


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
        requests_path = _write_request_prompts(
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
            results.append(
                RealizationResult(
                    benchmark_id=spec.benchmark_id,
                    case_id=spec.case_id,
                    source_builder=spec.source_builder,
                    synthesized_text=_parse_synthesized_text(raw_text),
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


def _parse_synthesized_text(raw_text: str) -> str:
    stripped = str(raw_text).strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict):
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
                return str(value).strip()
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, str):
            return first.strip()
    return stripped


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
        f"I'm dealing with a difficult situation in {domain}. "
        f"As the {role}, I need to decide what to do next. "
        f"{conflict}"
    ).strip()
    return RealizationResult(
        benchmark_id=spec.benchmark_id,
        case_id=spec.case_id,
        source_builder=spec.source_builder,
        synthesized_text=synthesized_text,
        prompt_template_name=spec.prompt_template_name,
        prompt_template_version=spec.prompt_template_version,
        synthesis_model=spec.synthesis_model,
        synthesis_request_version=spec.synthesis_request_version,
        request_prompt=request_prompt,
        metadata={"mode": "mock"},
    )


def _write_request_prompts(*, prompts: list[PromptRecord], output_path: Path | None) -> Path:
    if output_path is None:
        output_path = Path.cwd() / ".codex_benchmark_realization_requests.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "language": prompt.language,
                        "metadata": prompt.metadata,
                        "parameters": prompt.parameters,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path


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
    ) -> None:
        self.cases = cases
        self.build_artifacts = dict(build_artifacts or {})
        self.build_manifest_overrides = dict(build_manifest_overrides or {})
