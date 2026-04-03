# Progress

## Current Phase
- Phase 35 — V2 Task-First Core Refactor (`EvalTask` + `ExecutionRequest` + `RunEngineGateway`)

## Completed
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
  - [src/aigc/benchmarking/__init__.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/__init__.py)
  - [src/aigc/benchmarking/models.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/models.py)
  - [src/aigc/benchmarking/interfaces.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/interfaces.py)
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
  - [examples/analysis_plugins/ethics_family_consistency/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_family_consistency/plugin.py)
  - [examples/analysis_plugins/ethics_slot_sensitivity/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_slot_sensitivity/plugin.py)
  - [examples/normalizers/ethics_structural/normalizer.py](/Users/morinop/coding/whitzardgen/examples/normalizers/ethics_structural/normalizer.py)
  - [tests/test_benchmarking.py](/Users/morinop/coding/whitzardgen/tests/test_benchmarking.py)
  - [tests/test_cli_benchmark.py](/Users/morinop/coding/whitzardgen/tests/test_cli_benchmark.py)
- Added:
  - [src/aigc/benchmarking/compiler.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/compiler.py)
  - [src/aigc/benchmarking/gateway.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/gateway.py)
  - [src/aigc/benchmarking/runner.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/runner.py)
  - [src/aigc/benchmarking/resolution.py](/Users/morinop/coding/whitzardgen/src/aigc/benchmarking/resolution.py)
  - [docs/ethics_conflict_eval_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/docs/ethics_conflict_eval_runbook.zh-CN.md)

## Current Status
- Phase 35 core cutover is functionally in place:
- Phase 35 core cutover is functionally in place:
  - `evaluate_benchmark(...)` now resolves an `EvalTask`, compiles a `CompiledTaskPlan`, and runs it through `DefaultExperimentRunner`
  - V2 bundle outputs now include `execution_requests.jsonl`, `score_records.jsonl`, `experiment_log.jsonl`, and `compiled_task_plan.json`
  - examples and CLI surfaces are using V2-first names with compatibility fallbacks where needed
- Benchmarking-focused tests and broader regression suites are passing locally.
- `docs/` now includes a dedicated runbook for operating ethics-conflict evaluation on the current V2 architecture.
- Example discovery is now more robust on remote machines that run from source trees or editable installs.

## Blockers
- No confirmed blocker.
- Remaining work is mostly polish:
  - optionally migrate more text/UI wording from `evaluator` to `scorer`
  - optionally add more V2-specific coverage around bundle compatibility readers and `ExperimentLog` semantics
  - remote machine behavior should now be rechecked with a real `aigc benchmark build --builder ethics_sandbox ...` invocation

## Next Task
- If we continue from here, the best next slice is post-cutover cleanup:
  - tighten V2 docs/spec wording around `ScoreRecord` / `ExperimentLog`
  - add explicit compatibility-reader tests for older `evaluator_results.jsonl` bundles
  - standardize remaining CLI/user-facing text from “evaluator” to “scorer” where appropriate
