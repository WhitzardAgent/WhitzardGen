# Benchmark Framework Specification

## Overview

WhitzardGen now treats benchmark execution as the top-level workflow.

The framework must support:

- loading static benchmarks
- building synthetic benchmarks
- executing one or more target models
- normalizing raw target outputs into reusable intermediate records
- evaluating outputs with rule or judge-based evaluators
- running plugin-style comparative or post-processing analyses
- aggregating experiment reports

## Core Abstractions

- **Benchmark**
  - one named case collection with provenance
- **Benchmark Case**
  - one executable unit for target models
- **Benchmark Builder**
  - produces canonical cases from packages, templates, or Python code
- **Target Model**
  - the model under test
- **Evaluator**
  - transforms target outputs into analysis-ready judgments
- **Normalizer**
  - transforms raw target outputs into a schema-light normalized envelope
- **Analysis Plugin**
  - performs deterministic, judge-backed, or comparative post-processing on experiment artifacts
- **Experiment**
  - one run of `benchmark x targets x evaluators`

Core framework modules must remain workload-agnostic. Concrete benchmark families, normalizers, evaluator presets, analysis plugins, and experiment recipes should live under `examples/` and implement the generic builder/evaluator/analyzer interfaces.

## Supported Benchmark Families

- core generic sources such as static JSONL prompt/case sets
- example builders such as theme-tree
- example structural scenario packages such as the ethics sandbox templates

## Required Artifact Families

### Benchmark Bundle

- `cases.jsonl`
- `benchmark_manifest.json`
- `stats.json`

### Experiment Bundle

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

## Product Mapping

- `prompts generate` becomes a benchmark-builder internal path
- `run` becomes the target-execution kernel
- `normalizers` become reusable pre-evaluation layers
- `annotate` becomes a reusable record evaluator path
- `analysis plugins` become reusable post-evaluation layers
- `export dataset` remains an artifact-centric lower layer
