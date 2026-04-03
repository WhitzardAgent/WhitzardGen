# CLI Specification

## 1. Purpose

The CLI is the primary interface for running benchmark construction, target execution, evaluation, and lower-level generation workflows.

The top-level product view is now benchmark-centric:

```text
aigc benchmark ...
aigc evaluate ...
aigc experiments ...
```

The legacy/kernel-oriented commands remain available:

```text
aigc prompts ...
aigc run ...
aigc annotate ...
aigc runs ...
aigc export ...
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
aigc
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

### 4.1 `aigc benchmark list`

Lists supported benchmark builders discovered from the core and `examples/`.

Current sources:

- core generic builders such as `static_jsonl`
- example builders such as `theme_tree` and `ethics_sandbox`

### 4.2 `aigc benchmark build`

Builds a benchmark bundle.

Current supported shape:

```bash
aigc benchmark build \
  --builder ethics_sandbox \
  --source docs/ethics_design/sandbox_template \
  --config examples/benchmarks/ethics_sandbox/example_build.yaml \
  --build-mode matrix
```

Current outputs:

- `cases.jsonl`
- `benchmark_manifest.json`
- `stats.json`

### 4.3 `aigc benchmark inspect`

Inspects a benchmark bundle and reports:

- benchmark id
- builder
- source package
- case count
- counts by family
- counts by split

## 5. Evaluation Commands

### 5.1 `aigc evaluate run`

Runs one benchmark across one or more target models and optionally applies:

- normalizers
- record evaluators
- analysis plugins
- recipe-driven wiring for all of the above

Current shape:

```bash
aigc evaluate run \
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

### 5.2 `aigc experiments list`

Lists recorded experiment bundles.

### 5.3 `aigc experiments report`

Prints experiment manifest and summary information and points to the generated report.

## 6. Kernel-Oriented Commands

These commands remain supported because they expose useful lower-level workflow components.

### 6.1 `aigc prompts ...`

Prompt/theme-tree generation remains available, but it should be understood as a benchmark/scenario builder path rather than the primary product concept.

### 6.2 `aigc run ...`

The low-level multimodal execution kernel remains available for direct model runs.

### 6.3 `aigc annotate ...`

Annotation remains available and now acts as the reusable record-evaluator path behind higher-level evaluation workflows.

### 6.4 `aigc export ...`

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
