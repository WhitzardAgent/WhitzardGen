# Experiment Specification

## Definition

An experiment is one execution of:

```text
benchmark x target_models x normalizers x evaluators x analysis_plugins
```

It produces an immutable result bundle.

## Required Inputs

- benchmark path
- target model list
- optional normalizer set
- evaluator set or optional evaluator model/profile/template
- optional analysis plugin set
- execution mode
- optional recipe config

## Required Outputs

- `cases.jsonl`
- `target_results.jsonl`
- `normalized_results.jsonl`
- `evaluator_results.jsonl`
- `group_analyses.jsonl`
- `analysis_plugin_results.jsonl`
- `experiment_manifest.json`
- `summary.json`
- `report.md`
- `failures.json`

## Execution Plan

1. load benchmark cases
2. execute target models
3. normalize target results
4. optionally normalize target results
5. run record-level evaluation
6. run generic group analyses
7. run plugin-style analysis
8. write summary and report

## Recipe-Driven Execution

Experiments may be launched directly from CLI arguments or from reusable recipe configs under `configs/experiments/` and `examples/experiments/`.

Recipes can specify:

- benchmark source/builder
- target models
- normalizers
- evaluator sets
- analysis plugin sets
- output locations
- optional auto-launch preferences

## Structural Benchmark Notes

For structural scenario benchmarks, experiments must retain:

- family/template identity
- sibling group identity
- slot metadata
- response capture contract

Without these fields, consistency and sensitivity analysis becomes impossible.
