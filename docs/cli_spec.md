# CLI Specification

## 1. Purpose

The CLI is the primary interface for running benchmark construction, target execution, evaluation, and lower-level generation workflows.

The top-level product view is now benchmark-centric:

```text
whitzard benchmark ...
whitzard evaluate ...
whitzard experiments ...
```

The legacy/kernel-oriented commands remain available:

```text
whitzard prompts ...
whitzard run ...
whitzard annotate ...
whitzard runs ...
whitzard export ...
```

## 2. Design Principles

The CLI must be:

- scriptable
- deterministic
- explicit about inputs and outputs
- stable enough for long-running experiment automation
- able to expose both high-level benchmark workflows and low-level runtime utilities

## 3. Top-Level Groups

Current command groups:

```text
whitzard
 ├── benchmark
 ├── evaluate
 ├── experiments
 ├── prompts
 ├── annotate
 ├── models
 ├── run
 ├── runs
 ├── export
 ├── doctor
 └── version
```

Recommended user workflow starts with `benchmark`, `evaluate`, and `experiments`.

## 4. Benchmark-Centric Commands

### 4.1 `whitzard benchmark list`

Lists supported benchmark builders discovered from the core and `examples/`.

Current sources:

- core generic builders such as `static_jsonl`
- example builders such as `theme_tree` and `ethics_sandbox`

### 4.2 `whitzard benchmark build`

Builds a benchmark bundle.

Current supported shape:

```bash
whitzard benchmark build \
  --builder ethics_sandbox \
  --source examples/benchmarks/ethics_sandbox/package \
  --config examples/benchmarks/ethics_sandbox/example_build.yaml \
  --build-mode matrix
```

Current outputs:

- `cases.jsonl`
- `benchmark_manifest.json`
- `stats.json`

### 4.3 `whitzard benchmark inspect`

Inspects a benchmark bundle and reports:

- benchmark id
- builder
- source package
- case count
- counts by family
- counts by split

## 5. Evaluation Commands

### 5.1 `whitzard evaluate run`

Runs one benchmark across one or more target models and optionally applies:

- normalizers
- record evaluators
- analysis plugins
- recipe-driven wiring for all of the above

Current shape:

```bash
whitzard evaluate run \
  --recipe examples/experiments/ethics_structural.yaml
```

Behavior:

1. load benchmark cases
2. run each target model through the existing execution kernel
3. optionally normalize target outputs
4. run record-level evaluation
5. aggregate generic group-level analyses
6. run plugin-style comparative analysis
7. write an experiment bundle

Current bundle outputs:

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

### 5.2 `whitzard experiments list`

Lists recorded experiment bundles.

### 5.3 `whitzard experiments report`

Prints experiment manifest and summary information and points to the generated report.

### 5.4 `whitzard evaluate export`

Exports one completed experiment bundle into analysis-friendly tables.

Recommended usage:

```bash
whitzard evaluate export <experiment_id_or_path> --format both
```

Outputs:

- `dataset.jsonl`
- `dataset.csv`
- `export_manifest.json`
- `README.md`

Each exported row merges:

- benchmark case data
- execution request data
- target model output text when the artifact is readable as UTF-8
- matching normalized results
- matching score records

If an experiment finished with a partial V2 bundle, export still works as long as:

- `experiment_manifest.json` exists
- `target_results.jsonl` exists

Missing optional layers such as `normalized_results.jsonl` or `score_records.jsonl` are exported as empty fields rather than failing the whole export.

## 6. Kernel-Oriented Commands

These commands remain supported because they expose useful lower-level workflow components.

### 6.1 `whitzard prompts ...`

Prompt/theme-tree generation remains available, but it should be understood as a benchmark/scenario builder path rather than the primary product concept.

### 6.2 `whitzard run ...`

The low-level multimodal execution kernel remains available for direct model runs.

### 6.3 `whitzard annotate ...`

Annotation remains available and now acts as the reusable record-evaluator path behind higher-level evaluation workflows.

### 6.4 `whitzard export ...`

Dataset export remains a lower-level result/export layer and is still useful for dataset production workflows.

## 7. Output Modes

Every major command should support:

- `--output text`
- `--output json`

`text` is for operators.

`json` is for automation and pipelines.

## 8. Future Direction

The CLI should keep moving toward:

- benchmark builders as first-class workflows
- experiment config files
- reusable normalizer / evaluator / analysis-plugin sets
- stronger multi-target / multi-evaluator reporting

But it should do so without hiding or breaking the lower-level runtime controls that make the framework operationally useful.
