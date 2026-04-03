import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aigc.benchmarking.interfaces import (
    BenchmarkBuildRequest,
    CaseCompiler,
    ParameterSampler,
    RealizationValidator,
    StructureGuard,
)
from aigc.benchmarking.discovery import discover_example_builder_specs
from aigc.benchmarking.compiler import DefaultTaskCompiler
from aigc.benchmarking.packages import load_generative_benchmark_package
from aigc.benchmarking.models import (
    BenchmarkCase,
    CaseSourceRef,
    EvalTask,
    RealizationResult,
    RealizationSpec,
    RealizationValidationResult,
    ScoreRecord,
)
from aigc.benchmarking.realization import (
    BenchmarkBuildOutputLike,
    execute_semantic_realization_pipeline,
)
from aigc.benchmarking.service import build_benchmark, evaluate_benchmark, normalize_case_payload


def _build_run_summary(*, run_id: str, output_dir: Path, export_path: Path):
    return type(
        "Summary",
        (),
        {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "export_path": str(export_path),
            "status": "completed",
            "to_dict": lambda self: {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "export_path": str(export_path),
                "status": "completed",
            },
        },
    )()


class BenchmarkingTests(unittest.TestCase):
    def test_semantic_realization_pipeline_retries_until_validation_passes(self) -> None:
        class DummySampler(ParameterSampler):
            def sample(self, request: BenchmarkBuildRequest) -> list[RealizationSpec]:
                return [
                    RealizationSpec(
                        benchmark_id="suite",
                        case_id="case_001",
                        source_builder="dummy",
                        slot_assignments={"decision_maker_role": "doctor", "setting_domain": "ER"},
                        prompt_template_name="template_v1",
                        synthesis_model="Qwen3-32B",
                        prompt_context={},
                    )
                ]

        class DummyGuard(StructureGuard):
            def validate_spec(self, spec: RealizationSpec) -> list[str]:
                return []

            def validate_realization(self, spec: RealizationSpec, result: RealizationResult) -> list[str]:
                errors: list[str] = []
                if "benchmark" in result.synthesized_text.lower():
                    errors.append("Forbidden exposed term found: benchmark")
                return errors

        class DummyRenderer:
            def render(self, spec: RealizationSpec, *, validation_feedback=None) -> str:
                if validation_feedback:
                    return "retry prompt"
                return "initial prompt"

        class DummyBackend:
            def __init__(self) -> None:
                self.calls = 0

            def synthesize(
                self,
                *,
                specs: list[RealizationSpec],
                renderer,
                request: BenchmarkBuildRequest,
                validation_feedback_by_case_id: dict[str, list[str]] | None = None,
            ) -> list[RealizationResult]:
                self.calls += 1
                text = "This benchmark prompt is invalid." if self.calls == 1 else "A doctor in the ER needs advice."
                return [
                    RealizationResult(
                        benchmark_id="suite",
                        case_id="case_001",
                        source_builder="dummy",
                        synthesized_text=text,
                    )
                ]

        class DummyCompiler(CaseCompiler):
            def compile(self, spec: RealizationSpec, result: RealizationResult) -> BenchmarkCase:
                return BenchmarkCase(
                    benchmark_id=spec.benchmark_id,
                    case_id=spec.case_id,
                    input_modality="text",
                    input_payload={"prompt": result.synthesized_text},
                    prompt=result.synthesized_text,
                    source_builder=spec.source_builder,
                    metadata={"slot_assignments": dict(spec.slot_assignments)},
                )

        output = execute_semantic_realization_pipeline(
            request=BenchmarkBuildRequest(builder_name="dummy", execution_mode="mock"),
            sampler=DummySampler(),
            guard=DummyGuard(),
            renderer=DummyRenderer(),
            compiler=DummyCompiler(),
            synthesis_backend=DummyBackend(),
            max_attempts=2,
        )

        self.assertIsInstance(output, BenchmarkBuildOutputLike)
        self.assertEqual(len(output.cases), 1)
        self.assertIn("doctor", output.cases[0].prompt.lower())

    def test_semantic_realization_pipeline_uses_model_validator_feedback_for_retry(self) -> None:
        class DummySampler(ParameterSampler):
            def sample(self, request: BenchmarkBuildRequest) -> list[RealizationSpec]:
                return [
                    RealizationSpec(
                        benchmark_id="suite",
                        case_id="case_001",
                        source_builder="dummy",
                        slot_assignments={"decision_maker_role": "doctor"},
                        prompt_template_name="writer_v1",
                        synthesis_model="Qwen3-32B",
                    )
                ]

        class DummyGuard(StructureGuard):
            def validate_spec(self, spec: RealizationSpec) -> list[str]:
                return []

            def validate_realization(self, spec: RealizationSpec, result: RealizationResult) -> list[str]:
                return []

        class DummyRenderer:
            def render(self, spec: RealizationSpec, *, validation_feedback=None) -> str:
                if validation_feedback:
                    return "retry writer prompt"
                return "initial writer prompt"

        class DummyBackend:
            def __init__(self) -> None:
                self.calls = 0

            def synthesize(
                self,
                *,
                specs: list[RealizationSpec],
                renderer,
                request: BenchmarkBuildRequest,
                validation_feedback_by_case_id: dict[str, list[str]] | None = None,
            ) -> list[RealizationResult]:
                self.calls += 1
                if self.calls == 1:
                    return [
                        RealizationResult(
                            benchmark_id="suite",
                            case_id="case_001",
                            source_builder="dummy",
                            synthesized_text="This reads like a classroom puzzle.",
                            scene_description="This reads like a classroom puzzle.",
                            structured_output={"scene_description": "This reads like a classroom puzzle."},
                        )
                    ]
                return [
                    RealizationResult(
                        benchmark_id="suite",
                        case_id="case_001",
                        source_builder="dummy",
                        synthesized_text="A doctor has to decide immediately whether to disclose the risk.",
                        scene_description="A doctor has to decide immediately whether to disclose the risk.",
                        structured_output={
                            "scene_description": "A doctor has to decide immediately whether to disclose the risk.",
                            "decision_frame": {
                                "explicit_binary_required": True,
                                "action_structure": "implicit_binary",
                            },
                        },
                        decision_frame={
                            "explicit_binary_required": True,
                            "action_structure": "implicit_binary",
                        },
                    )
                ]

        class DummyValidator(RealizationValidator):
            def validate(
                self,
                *,
                specs: list[RealizationSpec],
                results: list[RealizationResult],
                request: BenchmarkBuildRequest,
            ) -> list[RealizationValidationResult]:
                del request
                validations: list[RealizationValidationResult] = []
                for spec, result in zip(specs, results, strict=True):
                    is_invalid = "classroom puzzle" in (result.scene_description or "").lower()
                    validations.append(
                        RealizationValidationResult(
                            benchmark_id=spec.benchmark_id,
                            case_id=spec.case_id,
                            valid=not is_invalid,
                            issues=["Scene reads like a benchmark item."] if is_invalid else [],
                            feedback_for_retry=["Rewrite as a live decision brief, not a classroom puzzle."]
                            if is_invalid
                            else [],
                        )
                    )
                return validations

        class DummyCompiler(CaseCompiler):
            def compile(self, spec: RealizationSpec, result: RealizationResult) -> BenchmarkCase:
                return BenchmarkCase(
                    benchmark_id=spec.benchmark_id,
                    case_id=spec.case_id,
                    input_modality="text",
                    input_payload={"prompt": result.scene_description or result.synthesized_text},
                    prompt=result.scene_description or result.synthesized_text,
                    source_builder=spec.source_builder,
                    metadata={"decision_frame": dict(result.decision_frame)},
                )

        output = execute_semantic_realization_pipeline(
            request=BenchmarkBuildRequest(builder_name="dummy", execution_mode="mock"),
            sampler=DummySampler(),
            guard=DummyGuard(),
            renderer=DummyRenderer(),
            validator=DummyValidator(),
            compiler=DummyCompiler(),
            synthesis_backend=DummyBackend(),
            max_attempts=2,
        )

        self.assertEqual(len(output.cases), 1)
        self.assertIn("doctor", output.cases[0].prompt.lower())
        self.assertTrue(output.build_artifacts["realization_validator_enabled"])

    def test_normalize_case_payload_supports_structured_v2_fields(self) -> None:
        case = normalize_case_payload(
            {
                "case_id": "case_structured_001",
                "input_modality": "text",
                "input_payload": {
                    "instruction": "Explain the tradeoff.",
                    "context": {"audience": "research"},
                    "language": "en",
                },
                "expected_output_contract": {"format": "json"},
                "metadata": {"family_id": "ethics_family_alpha"},
                "tags": ["ethics", "structured"],
                "split": "eval",
                "grouping": {"variant_group_id": "vg_001"},
                "execution_hints": {"temperature": 0.2},
                "evaluation_hints": {"judge_profile": "strict"},
            },
            benchmark_id="structured_suite",
            default_builder="static_jsonl",
            default_split="default",
        )

        self.assertEqual(case.benchmark_id, "structured_suite")
        self.assertEqual(case.input_modality, "text")
        self.assertEqual(case.input_payload["instruction"], "Explain the tradeoff.")
        self.assertEqual(case.expected_output_contract, {"format": "json"})
        self.assertEqual(case.grouping["variant_group_id"], "vg_001")
        self.assertEqual(case.execution_hints["temperature"], 0.2)
        self.assertEqual(case.evaluation_hints["judge_profile"], "strict")

    def test_example_builder_discovery_exposes_ethics_and_theme_tree(self) -> None:
        specs = discover_example_builder_specs()

        self.assertIn("ethics_sandbox", specs)
        self.assertIn("theme_tree", specs)
        self.assertEqual(specs["ethics_sandbox"].source, "example")

    def test_ethics_package_loader_resolves_canonical_and_legacy_alias_paths(self) -> None:
        canonical = load_generative_benchmark_package(
            "examples/benchmarks/ethics_sandbox/package"
        )
        legacy = load_generative_benchmark_package(
            "docs/ethics_design/sandbox_template"
        )

        self.assertTrue(canonical.canonical_package_path.endswith("examples/benchmarks/ethics_sandbox/package"))
        self.assertEqual(legacy.canonical_package_path, canonical.canonical_package_path)
        self.assertIsNotNone(legacy.alias_path)
        self.assertIn("decision_maker_role", canonical.slot_library)

    def test_canonical_ethics_templates_use_used_slots_not_inline_value_constraints(self) -> None:
        template_path = Path(
            "examples/benchmarks/ethics_sandbox/package/templates/competent_refusal_of_treatment.yaml"
        )
        raw = template_path.read_text(encoding="utf-8")

        self.assertIn("used_slots:", raw)
        self.assertNotIn("parameter_slots:", raw)
        self.assertNotIn("allowed_values:", raw)
        self.assertNotIn("fixed_value:", raw)
        self.assertNotIn("range:", raw)

    def test_writer_prompt_uses_hidden_control_signal_framing(self) -> None:
        template_path = Path(
            "examples/benchmarks/ethics_sandbox/synthesis_templates/standard_naturalistic_v1.txt"
        )
        raw = template_path.read_text(encoding="utf-8")

        self.assertIn("You write realistic English decision briefs.", raw)
        self.assertIn("Use these only as hidden control signals.", raw)
        self.assertIn("\"decision_frame\": {", raw)
        self.assertNotIn("Analysis targets:", raw)
        self.assertNotIn("Response capture contract:", raw)
        self.assertNotIn("Source-case references:", raw)

    def test_build_ethics_example_benchmark_writes_structural_metadata(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        builder_config_path = tmpdir / "builder.yaml"
        builder_config_path.write_text(
            "\n".join(
                [
                    "sampling:",
                    "  realizations_per_template: 2",
                    "synthesis:",
                    "  model: Qwen3-32B",
                    "validation:",
                    "  max_attempts: 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = build_benchmark(
            builder_name="ethics_sandbox",
            source_path="examples/benchmarks/ethics_sandbox/package",
            out_dir=tmpdir / "benchmark_bundle",
            benchmark_name="ethics_suite",
            builder_config_path=builder_config_path,
            build_mode="matrix",
            execution_mode="mock",
        )

        self.assertEqual(summary.builder_name, "ethics_sandbox")
        self.assertEqual(summary.build_mode, "matrix")
        cases = [
            json.loads(line)
            for line in Path(summary.cases_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreater(len(cases), 0)
        first_case = cases[0]
        self.assertEqual(first_case["benchmark_id"], "ethics_suite")
        self.assertIn("variant_group_id", first_case["metadata"])
        self.assertIn("slot_assignments", first_case["metadata"])
        self.assertIn("realization_prompt_template", first_case["metadata"])
        self.assertIn("synthesis_model", first_case["metadata"])
        self.assertIn("realization_provenance", first_case["metadata"])
        self.assertIn("decision_frame", first_case["metadata"])
        self.assertEqual(first_case["source_builder"], "ethics_sandbox")
        manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
        self.assertIn("build_artifacts", manifest)
        self.assertTrue(manifest["semantic_realization"]["enabled"])
        self.assertIn("validator", manifest["semantic_realization"])

    def test_task_compiler_builds_execution_requests_with_lineage(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        source_path = tmpdir / "cases.jsonl"
        source_path.write_text(
            json.dumps(
                {
                    "benchmark_id": "static_suite",
                    "case_id": "case_001",
                    "input_modality": "text",
                    "input_payload": {"prompt": "Describe the scenario.", "language": "en"},
                    "metadata": {"family_id": "fam_001"},
                    "grouping": {"variant_group_id": "vg_001"},
                    "execution_hints": {"temperature": 0.1},
                    "evaluation_hints": {"review_mode": "strict"},
                    "parameters": {"max_new_tokens": 128},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        benchmark_summary = build_benchmark(
            builder_name="static_jsonl",
            source_path=source_path,
            out_dir=tmpdir / "benchmark_bundle",
            benchmark_name="static_suite",
        )
        task = EvalTask(
            task_id="task_static_suite",
            case_source=CaseSourceRef(source_type="benchmark_bundle", source_path=benchmark_summary.benchmark_dir),
            case_set_path=benchmark_summary.benchmark_dir,
            target_models=["Qwen3-32B"],
            execution_policy={},
        )

        compiled = DefaultTaskCompiler().compile(task)

        self.assertEqual(compiled.case_set.benchmark_id, "static_suite")
        self.assertEqual(len(compiled.execution_requests), 1)
        request = compiled.execution_requests[0]
        self.assertEqual(request.task_id, "task_static_suite")
        self.assertEqual(request.case_id, "case_001")
        self.assertEqual(request.target_model, "Qwen3-32B")
        self.assertEqual(request.request_id, "task_static_suite:Qwen3-32B:case_001")
        self.assertEqual(request.runtime_hints["temperature"], 0.1)
        self.assertEqual(request.metadata["grouping"]["variant_group_id"], "vg_001")
        self.assertEqual(request.expected_output_contract, None)

    def test_evaluate_benchmark_reuses_target_run_and_evaluator_layer(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        runs_root = tmpdir / "runs"
        config_path = tmpdir / "local_runtime.yaml"
        config_path.write_text(
            json.dumps({"paths": {"runs_root": str(runs_root)}}),
            encoding="utf-8",
        )
        builder_config_path = tmpdir / "builder.yaml"
        builder_config_path.write_text(
            "\n".join(
                [
                    "sampling:",
                    "  realizations_per_template: 1",
                    "synthesis:",
                    "  model: Qwen3-32B",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        benchmark_summary = build_benchmark(
            builder_name="ethics_sandbox",
            source_path="docs/ethics_design/sandbox_template",
            out_dir=tmpdir / "benchmark_bundle",
            benchmark_name="ethics_suite_eval",
            builder_config_path=builder_config_path,
            build_mode="static",
            execution_mode="mock",
        )

        def fake_run_single_model(**kwargs):
            run_id = "run_eval_target_001"
            run_root = runs_root / run_id
            export_path = run_root / "exports" / "dataset.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_records = [
                json.loads(line)
                for line in Path(kwargs["prompt_file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows = []
            for index, prompt_record in enumerate(prompt_records, start=1):
                artifact_path = run_root / "workdir" / f"{prompt_record['prompt_id']}.txt"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(
                    "\n".join(
                        [
                            f"Recommended action: action_for_{prompt_record['prompt_id']}",
                            "Because: minimizing direct harm to the larger group.",
                            "Reasoning: compare the outcomes and choose the lower total harm.",
                            "Confidence: 0.8",
                        ]
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "record_id": f"rec_{index:08d}",
                        "run_id": run_id,
                        "task_id": f"task_{index:06d}",
                        "prompt_id": prompt_record["prompt_id"],
                        "prompt": prompt_record["prompt"],
                        "negative_prompt": None,
                        "language": prompt_record.get("language", "en"),
                        "model_name": kwargs["model_name"],
                        "task_type": "t2t",
                        "artifact_type": "text",
                        "artifact_path": str(artifact_path),
                        "artifact_metadata": {"format": "txt"},
                        "generation_params": {},
                        "prompt_metadata": prompt_record.get("metadata", {}),
                    }
                )
            export_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            (run_root / "run_manifest.json").write_text(
                json.dumps({"run_id": run_id, "export_path": str(export_path)}),
                encoding="utf-8",
            )
            (run_root / "failures.json").write_text("[]", encoding="utf-8")
            return _build_run_summary(run_id=run_id, output_dir=run_root, export_path=export_path)

        def fake_score_target_results(**kwargs):
            target_results = kwargs["target_results"]
            self.assertEqual(kwargs["source_run_id"], "run_eval_target_001")
            results = [
                ScoreRecord(
                    task_id=kwargs["task"].task_id,
                    benchmark_id=item.benchmark_id,
                    case_id=item.case_id,
                    case_version=item.case_version,
                    request_id=item.request_id,
                    target_model=item.target_model,
                    scorer_id="ethics_structural_judge",
                    status="success",
                    labels=["utilitarian_aggregation"],
                    scores={"confidence": 0.9},
                    rationale="mock rationale",
                    raw_judgment={"recommended_action": "action_a"},
                    scorer_metadata={"type": "judge"},
                    split=item.split,
                    tags=list(item.tags),
                    source_record_id=item.source_record_id,
                )
                for item in target_results
            ]
            return results, []

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(config_path)}, clear=False):
            with patch("aigc.benchmarking.gateway.run_single_model", side_effect=fake_run_single_model):
                with patch("aigc.benchmarking.runner.score_target_results", side_effect=fake_score_target_results):
                    summary = evaluate_benchmark(
                        benchmark_path=benchmark_summary.benchmark_dir,
                        target_models=["Qwen3-32B"],
                        normalizer_ids=["ethics_structural_normalizer"],
                        evaluator_ids=["ethics_structural_judge"],
                        analysis_plugin_ids=[
                            "ethics_family_consistency",
                            "ethics_slot_sensitivity",
                        ],
                        out_dir=tmpdir / "experiment_bundle",
                        execution_mode="mock",
                    )

        self.assertEqual(summary.target_run_count, 1)
        self.assertGreater(summary.normalized_result_count, 0)
        self.assertGreater(summary.score_record_count, 0)
        self.assertGreater(summary.group_analysis_record_count, 0)
        self.assertGreater(summary.analysis_plugin_result_count, 0)
        self.assertTrue(Path(summary.execution_requests_path).exists())
        self.assertTrue(Path(summary.normalized_results_path).exists())
        self.assertTrue(Path(summary.score_records_path).exists())
        self.assertTrue(Path(summary.analysis_plugin_results_path).exists())
        self.assertTrue(Path(summary.compiled_task_plan_path).exists())
        self.assertTrue(Path(summary.experiment_log_path).exists())
        report_text = Path(summary.report_path).read_text(encoding="utf-8")
        self.assertIn("Experiment Report", report_text)
        self.assertIn("Per-Normalizer Counts", report_text)
        self.assertIn("Analysis Plugin Counts", report_text)
