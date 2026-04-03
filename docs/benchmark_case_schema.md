# Benchmark Case Schema

## Canonical Fields

Each benchmark case should preserve enough structure for both execution and downstream analysis.

Required fields:

- `benchmark_id`
- `case_id`
- `family_id`
- `family_version`
- `variant_group_id`
- `input_type`
- `prompt`
- `language`
- `metadata`

Recommended fields:

- `split`
- `tags`
- `parameters`
- `version`

## Structural Benchmark Metadata

For structural scenario workloads, `metadata` should preserve:

- `deep_structure`
- `key_moral_conflict`
- `slot_assignments`
- `slot_layers`
- `invariants`
- `forbidden_transformations`
- `analysis_targets`
- `response_capture_contract`
- `realization_provenance`

## Compatibility with Prompt Execution

Benchmark cases may also be serialized in a prompt-compatible JSONL shape by keeping:

- `prompt_id = case_id`
- `prompt`
- `language`
- `parameters`
- benchmark metadata inside `metadata`

This allows benchmark cases to flow directly into the existing run kernel.
