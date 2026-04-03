# Requirements Traceability Matrix

This file maps your original intent to concrete documents in this package.

## Requirement A
Need a complete pipeline from sandbox-template-based scenario variation to multi-model local inference and automated analysis.

Covered by:
- `docs/03_pipeline_architecture.md`
- `docs/05_prompt_compiler_and_variant_generation.md`
- `docs/06_local_vllm_model_serving.md`
- `docs/07_request_orchestrator.md`
- `docs/08_llm_analysis_pipeline.md`
- `docs/16_cli_entrypoints_and_config_contracts.md`

## Requirement B
Need Codex to be able to implement the system incrementally rather than hallucinate a monolith.

Covered by:
- `AGENTS.md`
- `.codex/PLANS.template.md`
- `tasks/01_...` through `tasks/10_...`
- `docs/11_incremental_implementation_plan.md`
- `docs/19_repository_delivery_checklist.md`

## Requirement C
Need support for local multi-model requests via vLLM.

Covered by:
- `docs/06_local_vllm_model_serving.md`
- `examples/model_registry.example.yaml`
- `docs/16_cli_entrypoints_and_config_contracts.md`
- `schemas/model_request.schema.yaml`
- `schemas/model_response.schema.yaml`

## Requirement D
Need analysis-ready metadata for decisions, rationale, reasoning traces if available, consistency, and value preference.

Covered by:
- `docs/04_data_contracts.md`
- `docs/08_llm_analysis_pipeline.md`
- `docs/15_analysis_dimensions_and_metrics.md`
- `docs/18_reasoning_trace_policy.md`
- `schemas/normalized_response.schema.yaml`
- `schemas/analysis_result.schema.yaml`

## Requirement E
Need the package to remain aligned with Codex best practices.

Covered by:
- `docs/01_codex_best_practices.md`
- `AGENTS.md`
- `.codex/config.toml.example`
- task briefs in `tasks/`

## Requirement F
Need the pipeline to be reproducible, resumable, and suitable for publication-quality research operations.

Covered by:
- `docs/09_storage_and_reproducibility.md`
- `docs/17_failure_taxonomy_and_resume_strategy.md`
- `schemas/run_manifest.schema.yaml`
- `docs/19_repository_delivery_checklist.md`
