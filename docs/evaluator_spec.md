# Evaluator Specification

## Layering

WhitzardGen separates reusable post-target-processing into three generic layers.

### 1. Normalizer

Consumes:

- one target result

Produces:

- schema-light extracted fields
- optional refusal / decision / confidence signals
- optional reasoning-trace text

This layer is intentionally generic. Domain-specific extraction policies belong in example packages.

### 2. Record Evaluator

Consumes:

- one benchmark case
- one target result
- optional normalized result(s)

Produces:

- structured labels
- extracted fields
- scores
- rationale or notes

Supported styles:

- rule-based evaluator
- LLM-as-a-judge
- VLM-as-a-judge

### 3. Group Evaluator / Analysis Plugin

Consumes:

- multiple record evaluations from the same family, group, or experiment slice
- optional normalized outputs
- optional previous plugin outputs

Produces:

- stability metrics
- robustness metrics
- sensitivity metrics
- contradiction summaries

## Canonical Record Evaluation Output

Recommended envelope:

- `source_record_id`
- `source_prompt_id`
- `target_model`
- `status`
- `labels`
- `scores`
- `rationale`
- `raw_judgment`
- `evaluator_metadata`

Example packages may define richer structured fields inside `raw_judgment`, but those domain schemas must not leak into the generic core contract.

## Canonical Group Analysis / Plugin Output

Recommended fields:

- `benchmark_id`
- `target_model`
- `plugin_id`
- `group_key` or comparable slice identifier
- `case_count`
- `status`
- `labels`
- `scores`
- `output`
- `metadata`

## Implementation Mapping

Current implementation mapping:

- `normalize_target_results(...)` is the reusable normalizer path
- `annotate_run(...)` is the reusable record-evaluator path
- benchmark experiment aggregation computes generic group analyses after record evaluation
- plugin-style comparative analysis runs after record evaluation and can depend on normalized results
