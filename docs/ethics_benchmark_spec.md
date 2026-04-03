# Structural Ethics Benchmark Specification

## Purpose

This document describes one example benchmark family built on the generic framework: structural ethics scenario packages such as:

- [package](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/package)
- [examples/benchmarks/ethics_sandbox](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox)

These packages are not ordinary QA datasets. They encode:

- deep moral structure
- controlled slot-based variation
- invariant constraints
- forbidden transformations
- analysis targets

## Source Package Components

The sandbox package typically includes:

- `manifest.yaml`
- `templates/*.yaml`
- `slot_library.yaml`
- `analysis_codebook.yaml`

## Build Flow

The benchmark builder should:

1. load the package manifest and templates
2. sample or sweep structural, narrative, and perturbation slots
3. render a writer prompt that turns the structured spec into a second-person live decision brief plus structured A/B decision options
4. run a validator prompt that checks realism, benchmark-feel leakage, conflict preservation, second-person immersion, and binary framing
5. retry when validation feedback indicates the scene needs revision
6. preserve family, variant, and build-time decision-frame lineage in case metadata

## Execution Flow

1. run target models on the realized cases
2. capture outputs with full prompt metadata
3. run record-level extraction/judging
4. aggregate sibling cases into group analyses

## Analysis Targets

Typical outputs include:

- recommended action
- justification types
- uncertainty
- stakeholder prioritization
- normative labels
- stability under slot changes
- robustness to narrative perturbation
- consistency across template families

## Current CLI Path

For an architecture-focused Chinese runbook on the current V2 task-first stack, see:

- [docs/ethics_conflict_eval_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/docs/ethics_conflict_eval_runbook.zh-CN.md)

Build:

```bash
aigc benchmark build \
  --builder ethics_sandbox \
  --source examples/benchmarks/ethics_sandbox/package \
  --config examples/benchmarks/ethics_sandbox/example_build.yaml \
  --synthesis-model Qwen3-32B \
  --build-mode matrix
```

Evaluate:

```bash
aigc evaluate run \
  --benchmark runs/benchmarks/<bundle_id> \
  --targets Qwen3-32B \
  --evaluator-model Qwen3-32B \
  --evaluator-profile ethics_structural_review \
  --evaluator-template ethics_structural_review_v1
```

Current V2 experiment bundles are score-first rather than evaluator-first. The main artifacts are:

- `cases.jsonl`
- `execution_requests.jsonl`
- `target_results.jsonl`
- `normalized_results.jsonl`
- `score_records.jsonl`
- `group_analysis_records.jsonl`
- `analysis_plugin_results.jsonl`
- `experiment_log.jsonl`
- `compiled_task_plan.json`
- `experiment_manifest.json`
- `summary.json`
- `report.md`
- `failures.json`

## Key Invariant

The framework must never discard the template-family and slot lineage when a structural ethics benchmark is realized into prompts. That structure is what makes downstream consistency and sensitivity analysis meaningful.
