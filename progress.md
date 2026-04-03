# Progress

## Current Phase
- Phase 36 — Semantic Benchmark Realization Pipeline for Ethics and Safety Builders

## Completed
- Added generic semantic-realization core contracts so benchmark builders can express sampled structured specs, synthesis requests, and realized outputs without baking workload semantics into core:
  - [src/aigc/benchmarking/models.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/models.py)
  - [src/aigc/benchmarking/interfaces.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/interfaces.py)
- Implemented the reusable semantic realization pipeline and run-kernel-backed synthesis orchestration in:
  - [src/aigc/benchmarking/realization.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/realization.py)
- Wired benchmark build and CLI to support `synthesis_model` and build-manifest realization metadata:
  - [src/aigc/benchmarking/service.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/service.py)
  - [src/aigc/cli/main.py](/Users/morinop/coding/whitzardgen/src/aigc/cli/main.py)
- Reworked the ethics sandbox builder from mechanical prompt stitching to:
  - slot sampling
  - structure validation
  - template-driven LLM realization
  - final case compilation
  in [examples/benchmarks/ethics_sandbox/builder.py](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/builder.py)
- Added semantic-build config and realization template for the ethics example package:
  - [examples/benchmarks/ethics_sandbox/example_build.yaml](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/example_build.yaml)
  - [examples/benchmarks/ethics_sandbox/synthesis_templates/standard_naturalistic_v1.txt](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/synthesis_templates/standard_naturalistic_v1.txt)
- Updated the ethics runbook/docs so the user-facing workflow now reflects semantic benchmark realization instead of mechanical prompt stitching:
  - [docs/ethics_conflict_eval_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/docs/ethics_conflict_eval_runbook.zh-CN.md)
  - [examples/benchmarks/ethics_sandbox/README.md](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/README.md)
  - [examples/experiments/ethics_structural_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural_runbook.zh-CN.md)
  - [examples/experiments/ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)
- Added regression and pipeline tests covering semantic-realization retries, metadata lineage, CLI passthrough, and mock-mode ethics benchmark build:
  - [tests/test_benchmarking.py](/Users/morinop/coding/whitzardgen/tests/test_benchmarking.py)
  - [tests/test_cli_benchmark.py](/Users/morinop/coding/whitzardgen/tests/test_cli_benchmark.py)
- Replaced benchmark/evaluation core contracts with V2 task-first types in [src/aigc/benchmarking/models.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/models.py):
  - `CaseSourceRef`
  - `CaseSet`
  - `EvalTask`
  - `ExecutionRequest`
  - `CompiledTaskPlan`
  - `TargetResult`
  - `NormalizedResult`
  - `ScoreRecord`
  - `GroupAnalysisRecord`
  - `ExperimentLogEvent`
  - `ExperimentBundleManifest`
- Reworked benchmark interfaces in [src/aigc/benchmarking/interfaces.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/interfaces.py) around:
  - `TaskCompiler`
  - `RunEngineGateway`
  - `ExperimentRunner`
  - scorer-oriented contracts
  - V2 normalization / analysis requests
- Added V2 planning / execution scaffolding:
  - [src/aigc/benchmarking/compiler.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/compiler.py)
  - [src/aigc/benchmarking/gateway.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/gateway.py)
  - [src/aigc/benchmarking/runner.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/runner.py)
  - [src/aigc/benchmarking/resolution.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/resolution.py)
- Updated normalization and scoring substrate toward V2:
  - [src/aigc/normalizers/service.py](/Users/morinop/coding/whitzardgen/src/aigc/normalizers/service.py)
  - [src/aigc/evaluators/models.py](/Users/morinop/coding/whitzardgen/src/aigc/evaluators/models.py)
  - [src/aigc/evaluators/service.py](/Users/morinop/coding/whitzardgen/src/aigc/evaluators/service.py)
  - [examples/normalizers/ethics_structural/normalizer.py](/Users/morinop/coding/whitzardgen/examples/normalizers/ethics_structural/normalizer.py)
- Cut the benchmark artifact layer and orchestration over to V2-first outputs and flow:
  - [src/aigc/benchmarking/bundle.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/bundle.py)
  - [src/aigc/benchmarking/service.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/service.py)
  - [src/aigc/analysis/service.py](/Users/morinop/coding/whitzardgen/src/aigc/analysis/service.py)
  - [src/aigc/cli/main.py](/Users/morinop/coding/whitzardgen/src/aigc/cli/main.py)
- Updated example plugins and tests to the new primary V2 names:
  - [examples/analysis_plugins/ethics_family_consistency/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_family_consistency/plugin.py)
  - [examples/analysis_plugins/ethics_slot_sensitivity/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_slot_sensitivity/plugin.py)
  - [tests/test_benchmarking.py](/Users/morinop/coding/whitzardgen/tests/test_benchmarking.py)
  - [tests/test_cli_benchmark.py](/Users/morinop/coding/whitzardgen/tests/test_cli_benchmark.py)
- Fixed one unrelated-but-real recovery regression discovered during full-suite verification:
  - [src/aigc/run_flow.py](/Users/morinop/coding/whitzardgen/src/aigc/run_flow.py)
- Added current-architecture ethics evaluation docs in `docs/`:
  - [docs/ethics_conflict_eval_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/docs/ethics_conflict_eval_runbook.zh-CN.md)
  - [docs/ethics_benchmark_spec.md](/Users/morinop/coding/whitzardgen/docs/ethics_benchmark_spec.md)
- Fixed remote/source-install example discovery so `examples.*` entrypoints can load even when `examples` is not installed as a site-package:
  - [src/aigc/benchmarking/discovery.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/discovery.py)

## Files Added/Modified
- Modified:
  - [progress.md](/Users/morinop/coding/whitzardgen/progress.md)
  - [src/aigc/benchmarking/__init__.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/__init__.py)
  - [src/aigc/benchmarking/models.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/models.py)
  - [src/aigc/benchmarking/interfaces.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/interfaces.py)
  - [src/aigc/benchmarking/realization.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/realization.py)
  - [src/aigc/benchmarking/bundle.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/bundle.py)
  - [src/aigc/benchmarking/service.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/service.py)
  - [src/aigc/benchmarking/runner.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/runner.py)
  - [src/aigc/analysis/service.py](/Users/morinop/coding/whitzardgen/src/aigc/analysis/service.py)
  - [src/aigc/cli/main.py](/Users/morinop/coding/whitzardgen/src/aigc/cli/main.py)
  - [src/aigc/normalizers/service.py](/Users/morinop/coding/whitzardgen/src/aigc/normalizers/service.py)
  - [src/aigc/evaluators/models.py](/Users/morinop/coding/whitzardgen/src/aigc/evaluators/models.py)
  - [src/aigc/evaluators/service.py](/Users/morinop/coding/whitzardgen/src/aigc/evaluators/service.py)
  - [src/aigc/run_flow.py](/Users/morinop/coding/whitzardgen/src/aigc/run_flow.py)
  - [src/aigc/benchmarking/discovery.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/discovery.py)
  - [docs/ethics_benchmark_spec.md](/Users/morinop/coding/whitzardgen/docs/ethics_benchmark_spec.md)
  - [docs/ethics_conflict_eval_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/docs/ethics_conflict_eval_runbook.zh-CN.md)
  - [examples/benchmarks/ethics_sandbox/builder.py](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/builder.py)
  - [examples/benchmarks/ethics_sandbox/example_build.yaml](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/example_build.yaml)
  - [examples/benchmarks/ethics_sandbox/README.md](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/README.md)
  - [examples/experiments/ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)
  - [examples/experiments/ethics_structural_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural_runbook.zh-CN.md)
  - [examples/analysis_plugins/ethics_family_consistency/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_family_consistency/plugin.py)
  - [examples/analysis_plugins/ethics_slot_sensitivity/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_slot_sensitivity/plugin.py)
  - [examples/normalizers/ethics_structural/normalizer.py](/Users/morinop/coding/whitzardgen/examples/normalizers/ethics_structural/normalizer.py)
  - [tests/test_benchmarking.py](/Users/morinop/coding/whitzardgen/tests/test_benchmarking.py)
  - [tests/test_cli_benchmark.py](/Users/morinop/coding/whitzardgen/tests/test_cli_benchmark.py)
- Added:
  - [examples/benchmarks/ethics_sandbox/synthesis_templates/standard_naturalistic_v1.txt](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/synthesis_templates/standard_naturalistic_v1.txt)
  - [src/aigc/benchmarking/compiler.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/compiler.py)
  - [src/aigc/benchmarking/gateway.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/gateway.py)
  - [src/aigc/benchmarking/runner.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/runner.py)
  - [src/aigc/benchmarking/resolution.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/resolution.py)
  - [docs/ethics_conflict_eval_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/docs/ethics_conflict_eval_runbook.zh-CN.md)

## Current Status
- Phase 36 semantic benchmark realization is functionally in place.
- `ethics_sandbox` no longer mechanically stitches final prompts; it now samples structured slot assignments, renders synthesis requests, calls the existing T2T run kernel, validates outputs, and compiles final `BenchmarkCase`s with realization lineage.
- Benchmark bundles remain lightweight, while final case metadata now preserves:
  - `slot_assignments`
  - `realization_prompt_template`
  - `synthesis_model`
  - `synthesis_request_version`
  - `realization_provenance`
- Local regression coverage for semantic realization, ethics benchmark build, CLI passthrough, prompt generation, annotation, and run CLI paths is passing.

## Blockers
- No confirmed blocker.
- Remaining work is follow-up polish and expansion:
  - update user-facing docs/runbooks to emphasize `--synthesis-model` and the new build config shape
  - add a second non-ethics generative builder example to further validate the generic pipeline
  - optionally add richer realization validators and retry policies

## Next Task
- Update the benchmark/ethics runbook docs for the semantic build pipeline and consider adding a second generative example builder beyond ethics.
