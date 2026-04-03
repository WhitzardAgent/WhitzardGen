# WhitzardGen System Specification

## 1. Positioning

WhitzardGen is a **benchmark-centric multimodal evaluation framework** built on top of a reusable generation/runtime kernel.

It supports two complementary product views:

- **benchmark/evaluation framework**
  - build benchmark cases
  - execute one or more target models
  - run rule-based or model-based evaluators
  - aggregate structured experiment results
- **generation/data-production kernel**
  - prompt generation
  - multimodal model execution
  - artifact collection
  - dataset export

The framework is therefore dual-use, but the primary user-facing abstraction is now:

```text
benchmark -> target execution -> evaluation -> experiment report
```

## 2. Core Objectives

The system is designed to support:

- static QA-style benchmarks
- synthetic slot/template benchmarks
- multimodal model benchmarking across `t2t`, `t2i`, and `t2v`
- scalable local multi-GPU execution
- reproducible case generation and evaluation lineage
- structured analysis bundles rather than only raw generated media

Representative workloads include:

- unsafe prompt benchmark evaluation for image/video models
- LLM benchmark execution with structured output analysis
- structural ethics benchmark generation from example scenario-template packages
- cross-model comparison with per-family and per-group consistency analysis

## 3. Top-Level Architecture

The system is organized into five logical layers:

```text
CLI
  ↓
Benchmark / Experiment Orchestrator
  ↓
Execution Kernel
  ↓
Evaluator Layer
  ↓
Bundles / Reports / Exports
```

Main subsystems:

1. **Benchmark builders**
   - core generic sources such as static JSONL
   - example builders such as theme-tree and structural ethics packages
2. **Execution kernel**
   - model registry
   - adapters
   - runtime scheduling
   - persistent workers
   - recovery
3. **Evaluators**
   - rule-based evaluators
   - LLM-as-a-judge evaluators
   - VLM-as-a-judge evaluators
4. **Experiment aggregation**
   - record-level evaluation outputs
   - group-level consistency / sensitivity analyses
   - reports and summaries
5. **Lower-level data-production outputs**
   - run ledgers
   - dataset export bundles
   - recovery artifacts

## 4. Canonical Concepts

### 4.1 Benchmark

A **benchmark** is a named collection of canonical cases plus provenance and metadata.

Examples:

- static JSONL prompt benchmarks
- theme-tree prompt bundles used as benchmark inputs
- example structural scenario packages realized into benchmark cases

### 4.2 Benchmark Case

A **benchmark case** is one execution unit for target models.

Each case must be traceable and analysis-ready. In addition to prompt text, cases may carry:

- `benchmark_id`
- `case_id`
- `family_id` / `template_id`
- `variant_group_id`
- `input_type`
- `metadata`
- `tags`
- `split`

Structural scenario workloads additionally preserve:

- deep conflict structure
- slot assignments
- invariants
- forbidden transformations
- analysis targets
- response capture contract

### 4.3 Target Execution

A **target execution** is one model-under-test applied to one or more benchmark cases through the existing run kernel.

This layer reuses:

- registry resolution
- batching
- persistent workers
- recovery
- telemetry
- run manifests and logs

### 4.4 Evaluator

An **evaluator** transforms target outputs into analysis-ready results.

Two evaluator levels are first-class:

- **record evaluator**
  - evaluates one target output at a time
  - extracts labels, scores, rationales, or structured judgments
- **group evaluator**
  - analyzes multiple sibling outputs together
  - measures consistency, robustness, stability, and sensitivity

### 4.5 Experiment

An **experiment** is one run of:

```text
benchmark x target_models x evaluators
```

An experiment produces a standalone result bundle with:

- raw target results
- record-level evaluations
- group-level analyses
- manifest
- summary
- report
- failures

## 5. Mapping from Existing Subsystems

The existing subsystems remain important, but their product meaning changes:

- `prompt generation` -> **benchmark / scenario builder internals**
- `run` -> **execution kernel**
- `annotate` -> **record evaluator path**
- `export dataset` -> **lower-level result/export layer**

This means older commands remain useful, but the higher-level workflow is now benchmark-centric.

## 6. Core vs Examples

The benchmark core must remain workload-agnostic.

Core responsibilities:

- canonical schemas
- builder/evaluator/analyzer interfaces
- benchmark and experiment bundle I/O
- orchestration across target and evaluator planes
- reuse of the execution kernel

Concrete workload families belong in `examples/`, not in core package types.

## 7. Structural Scenario Example Support

Structural ethics scenario packages are implemented as an example benchmark-builder family.

The framework must preserve that these workloads are not plain QA prompts. Each realized case belongs to a scenario family and retains:

- template family identity
- structural / narrative / perturbation slot assignments
- invariant constraints
- forbidden transformations
- analysis targets

This enables:

- scenario realization from reusable template packages
- target-model execution on naturalistic prompts
- record-level extraction of decisions and reasoning signals
- group-level comparison across sibling variants

For this example benchmark family, the framework must support:

- controlled realization from template packages
- traceable case grouping
- cross-variant aggregation
- consistency and sensitivity reporting

## 8. Outputs

The framework now has two major artifact families.

### 7.1 Run Artifacts

Produced by the execution kernel:

- `run_manifest.json`
- `samples.jsonl`
- `runtime_status.json`
- `failures.json`
- dataset export files

### 7.2 Experiment Artifacts

Produced by benchmark/evaluation workflows:

- `cases.jsonl`
- `target_results.jsonl`
- `evaluator_results.jsonl`
- `group_analyses.jsonl`
- `experiment_manifest.json`
- `summary.json`
- `report.md`
- `failures.json`

## 9. Scope Boundaries

This framework is not trying to be:

- a generic hosted model-serving platform
- a full distributed orchestration stack
- a free-form notebook analysis environment

It is a structured local/cluster evaluation framework with a strong execution kernel and traceable experiment bundles.
