# Pipeline DAG Specification

## 1. Purpose

This document defines the high-level workflow DAG for WhitzardGen as a benchmark-centric evaluation system.

The framework still reuses the existing generation kernel, but the primary DAG is no longer just:

```text
prompt -> model -> artifact
```

It is now:

```text
benchmark build -> target execution -> record evaluation -> group analysis -> report
```

## 2. Design Principles

The pipeline must be:

- composable
- deterministic
- stage-isolated
- resumable where practical
- artifact- and lineage-preserving
- compatible with both static and synthetic benchmarks

## 3. High-Level DAG

```text
Benchmark Source
    ↓
Benchmark Load / Build
    ↓
Case Validation
    ↓
Target Execution
    ↓
Target Result Normalization
    ↓
Record Evaluation
    ↓
Group Analysis
    ↓
Summary / Report / Export
```

## 4. Stage Definitions

### 4.1 Benchmark Source

Supported sources include:

- static JSONL benchmark cases
- synthetic benchmark builders discovered from `examples/`
- structural scenario template packages handled by example builders

Examples:

- theme-tree prompt bundles
- ethics sandbox template packages

### 4.2 Benchmark Load / Build

This stage produces canonical benchmark cases.

It may:

- load already realized cases
- synthesize new cases from templates and slot policies
- preserve family / variant provenance

For structural scenario packages, this stage must also preserve:

- family/template identity
- slot assignments
- invariants
- forbidden transformations
- analysis targets

### 4.3 Case Validation

Validates canonical benchmark cases before execution.

Checks include:

- required IDs
- prompt/instruction presence
- split and metadata shape
- case-family lineage fields for structural benchmarks

### 4.4 Target Execution

Runs benchmark cases through target models using the existing runtime kernel.

This stage reuses:

- registry resolution
- batching
- persistent workers
- recovery
- telemetry
- run manifests

### 4.5 Target Result Normalization

Produces normalized target result records suitable for evaluators and reporting.

These records should preserve:

- case identity
- target model identity
- source prompt metadata
- artifact path
- artifact metadata
- generation params

### 4.6 Record Evaluation

Evaluates one target result at a time.

Supported evaluator types:

- rule-based
- LLM-as-a-judge
- VLM-as-a-judge

Outputs:

- labels
- scores
- rationales
- structured JSON judgments

### 4.7 Group Analysis

Aggregates multiple evaluated records together.

This stage is required for workloads where single-record judging is not enough, especially:

- structural ethics scenario families
- perturbation robustness studies
- within-family consistency checks

Typical outputs:

- consistency metrics
- sensitivity metrics
- contradiction detection
- per-family summaries

### 4.8 Summary / Report / Export

Produces experiment-facing outputs:

- summary JSON
- report markdown
- structured result JSONL files
- failure records

Lower-level dataset export remains available as a separate layer when the user needs artifact-centric outputs.

## 5. Example Structural Ethics Workflow

For the `examples/benchmarks/ethics_sandbox` workload, the DAG specializes as:

```text
Sandbox Template Package
    ↓
Scenario Family Load
    ↓
Slot Sampling / Matrix Expansion
    ↓
Scenario Realization
    ↓
Target Model Execution
    ↓
Record-Level Extraction / Judging
    ↓
Group-Level Consistency / Sensitivity Analysis
    ↓
Experiment Report
```

The key rule is that the framework must not collapse these cases into anonymous prompts. The family and variant structure must survive all downstream stages.

## 6. Relationship to Existing Subsystems

The older pipeline names still map cleanly into the new DAG:

- prompt generation -> benchmark/scenario realization
- run -> target execution
- annotate -> record evaluation
- export -> lower-level artifact export

This keeps the operationally mature kernel intact while changing the top-level product structure.
