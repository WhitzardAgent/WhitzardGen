# Pipeline DAG Specification

## 1. Purpose

The **Pipeline DAG** defines how the framework executes generation workflows from prompt input to artifact output.

This document specifies:

- pipeline stages
- stage dependencies
- optional vs required stages
- task expansion rules
- batch execution behavior
- retry and resume behavior
- artifact flow

The DAG is designed for a **modular generation framework**, not for a single rigid end-to-end workflow.  
Different jobs may enable different subsets of the pipeline.

For MVP, the pipeline is intentionally minimal and focuses on:

- prompt loading
- preprocessing
- model execution
- artifact collection
- dataset export

The MVP does **not require labeling**.

---

# 2. Design Principles

The pipeline must follow these principles:

1. **Composable**
   - not every job must run every stage

2. **Deterministic**
   - same prompt set + same model + same parameters should produce the same task graph

3. **Batch-aware**
   - batch-capable models should receive prompt batches

4. **Resumable**
   - completed stages should not be recomputed unnecessarily

5. **Artifact-first**
   - generated files are first-class outputs of the pipeline

6. **Stage-isolated**
   - each stage should have clear inputs and outputs

---

# 3. Pipeline Scope

The framework supports multiple job types.

Examples:

- text-to-image generation
- text-to-video generation
- text-to-audio generation
- text-to-text generation

A single job usually targets:

- one modality
- one task type
- multiple models

Example:

- one prompt list for multiple video models
- one prompt list for multiple image models

It is **not required** that one job mixes image, video, audio, and text.

---

# 4. High-Level DAG

The general pipeline DAG is:

```text
Prompt Source
    ↓
Prompt Load
    ↓
Prompt Validation
    ↓
Prompt Preprocessing
    ↓
Prompt Batching
    ↓
Task Expansion
    ↓
Model Execution
    ↓
Artifact Collection
    ↓
Artifact Validation
    ↓
Dataset Export
````

Optional future stages:

```text
Artifact Labeling
Dataset Curation
Evaluation
```

---

# 5. Stage Definitions

## 5.1 Prompt Source

The source of prompts for a run.

Supported MVP source types:

* JSONL file

Future source types:

* generated prompt sets
* database-backed prompt stores
* API sources

### Input

None

### Output

A raw prompt file path or prompt dataset handle

---

## 5.2 Prompt Load

Loads prompt records into the framework.

### Input

* prompt source

### Output

* list of prompt records

### Rules

* every record must parse as valid JSON
* every record must include `prompt_id`
* every record must include `prompt`
* language defaults may be filled in here if missing

### Example output

```json
[
  {
    "prompt_id": "p001",
    "prompt": "a futuristic city",
    "language": "en"
  },
  {
    "prompt_id": "p002",
    "prompt": "一只可爱的猫",
    "language": "zh"
  }
]
```

---

## 5.3 Prompt Validation

Validates prompts against the prompt schema.

### Input

* loaded prompt records

### Output

* validated prompt records
* validation errors, if any

### Validation checks

* `prompt_id` exists and is unique
* `prompt` is non-empty
* `language` is supported
* `negative_prompt` is valid if present
* optional parameters follow expected types

### Failure behavior

* invalid prompts may either:

  * fail the job immediately
  * be skipped and logged

This behavior should be configurable.

---

## 5.4 Prompt Preprocessing

Transforms validated prompts into model-ready prompts.

### Input

* validated prompt records
* target model definition

### Output

* model-ready prompt records

### Responsibilities

* prompt normalization
* language adaptation
* translation if needed
* optional prompt rewriting
* negative prompt normalization
* token-length enforcement

### Examples

#### Whitespace normalization

```text
"  a   futuristic city  " → "a futuristic city"
```

#### Translation

```text
"生成一张未来城市的图片" → "Generate an image of a futuristic city"
```

### Notes

Preprocessing may be model-specific.

For example:

* model A accepts Chinese directly
* model B requires English
* model C requires negative prompts
* model D has a short token limit

---

## 5.5 Prompt Batching

Groups prompts into batches for models that support batch inference.

### Input

* preprocessed prompts
* adapter capabilities
* scheduler constraints

### Output

* prompt batches

### Rules

If model supports batch prompts:

```text
batch_size = min(model.max_batch_size, scheduler.batch_limit)
```

If model does not support batch prompts:

```text
batch_size = 1
```

### Example output

```json
[
  {
    "batch_id": "b001",
    "prompt_ids": ["p001", "p002", "p003"]
  },
  {
    "batch_id": "b002",
    "prompt_ids": ["p004"]
  }
]
```

### Important

Batching happens **per model**, because different models have different capabilities.

---

## 5.6 Task Expansion

Expands prompt batches into executable tasks.

### Input

* selected models
* prompt batches
* job parameters

### Output

* task records

### Task definition

A task is:

```text
(model, prompt_batch, execution_params)
```

### Example

```json
{
  "task_id": "task_001",
  "model": "Z-Image",
  "batch_id": "b001",
  "prompt_ids": ["p001", "p002", "p003"],
  "params": {
    "guidance_scale": 4.0,
    "steps": 50
  }
}
```

### Notes

Task expansion is where the system creates the real execution graph.

For example:

* 100 prompts
* 2 models
* batch size 4 for model A
* batch size 1 for model B

This will produce different task counts per model.

---

## 5.7 Model Execution

Executes each task using the model adapter and executor.

### Input

* task record
* model adapter
* execution environment

### Output

* raw execution result
* logs
* generated files in workdir

### Execution modes

#### In-process execution

Used for:

* Diffusers pipelines
* Transformers APIs

Execution flow:

```text
adapter.prepare()
executor calls adapter.execute()
adapter.collect()
```

#### External-process execution

Used for:

* `python script.py ...`
* `torchrun ...`
* repository demo scripts

Execution flow:

```text
adapter.prepare()
executor runs subprocess
adapter.collect()
```

### Important

Execution ownership belongs to:

* framework executor for external-process mode
* adapter execute() for in-process mode

This boundary must remain strict.

---

## 5.8 Artifact Collection

Collects generated artifacts from the task workdir and converts them into normalized records.

### Input

* execution result
* workdir
* prompt_ids
* model adapter

### Output

* artifact records
* per-prompt result mapping

### Artifact examples

* PNG
* JPG
* MP4
* WAV
* TXT

### Example output

```json
{
  "status": "success",
  "batch_items": [
    {
      "prompt_id": "p001",
      "artifacts": [
        {
          "type": "image",
          "path": "runs/run_001/Z-Image/p001.png",
          "metadata": {
            "resolution": "1024x1024"
          }
        }
      ]
    }
  ]
}
```

### Responsibilities

* locate output files
* map outputs to prompt IDs
* extract metadata
* detect missing artifacts
* support partial batch success

---

## 5.9 Artifact Validation

Verifies that collected artifacts are structurally valid.

### Input

* artifact records

### Output

* validated artifact records
* validation errors

### Checks

Images:

* file exists
* readable format
* width and height can be extracted

Videos:

* file exists
* duration can be read
* fps can be read
* output is non-empty

Audio:

* file exists
* duration can be read
* sample rate can be read

### Failure behavior

Invalid artifacts may be:

* marked failed
* retried
* excluded from export

---

## 5.10 Dataset Export

Exports final records in dataset-ready format.

### Input

* validated artifacts
* prompt records
* model metadata
* run metadata

### Output

* JSONL
* Parquet

### Example exported record

```json
{
  "prompt_id": "p001",
  "prompt": "a futuristic city",
  "model": "Z-Image",
  "artifact_path": "runs/run_001/Z-Image/p001.png",
  "artifact_type": "image",
  "metadata": {
    "resolution": "1024x1024"
  }
}
```

---

# 6. Optional Stages

The pipeline is modular. Some stages are optional.

## 6.1 Postprocessing

Optional for MVP.

Examples:

* resize image
* transcode video
* compress outputs
* add thumbnails

This stage should not be required to complete generation jobs.

---

## 6.2 Labeling

Optional and out of MVP scope.

Future examples:

* safety labeling
* semantic labeling
* quality scoring

This stage should be attached as an additional pipeline branch, not embedded into the core generation DAG.

---

## 6.3 Evaluation

Optional future stage.

Examples:

* prompt adherence scoring
* quality metrics
* diversity metrics

---

# 7. Stage Dependency Rules

The DAG dependency rules are:

```text
Prompt Load depends on Prompt Source
Prompt Validation depends on Prompt Load
Prompt Preprocessing depends on Prompt Validation
Prompt Batching depends on Prompt Preprocessing
Task Expansion depends on Prompt Batching
Model Execution depends on Task Expansion
Artifact Collection depends on Model Execution
Artifact Validation depends on Artifact Collection
Dataset Export depends on Artifact Validation
```

Optional stages depend on Artifact Collection or Artifact Validation depending on their purpose.

---

# 8. Execution Model

## 8.1 Run

A **run** is a top-level execution instance.

A run includes:

* job configuration
* selected models
* input prompt source
* all generated tasks
* final exported dataset

### Example

```json
{
  "run_id": "run_001",
  "job_name": "video_generation_mvp"
}
```

---

## 8.2 Task

A **task** is the smallest schedulable execution unit.

A task includes:

* model
* prompt batch
* params
* workdir
* state

### Task states

Allowed states:

```text
pending
running
success
partial_success
failed
skipped
```

---

## 8.3 Workdir

Each task must use an isolated working directory.

Example:

```text
runs/run_001/workdir/task_001/
```

This prevents collisions between model outputs.

---

# 9. Retry and Resume Rules

## 9.1 Retry

Retries apply at the **task level**, not the whole run level.

A task may be retried if:

* subprocess crashed
* CUDA OOM
* artifact missing
* timeout

Retry limit example:

```text
max_retries = 3
```

## 9.2 Resume

The pipeline must support resuming interrupted runs.

Resume behavior:

* completed tasks are skipped
* failed tasks may be retried
* pending tasks continue
* exported outputs are not regenerated unless requested

---

# 10. Parallelism and Concurrency

## 10.1 Inter-model Parallelism

Different models may run in parallel if resources allow.

Example:

* Z-Image batch running on GPU 0
* LongCat-Video running on GPU 1

## 10.2 Intra-model Batch Parallelism

Some models support batch prompt execution in one inference call.

Examples:

* Diffusers image generation
* Transformer generation APIs

## 10.3 Concurrency Responsibility

The scheduler controls:

* how many tasks run simultaneously
* how prompts are batched
* which GPU/worker executes which task

The pipeline DAG defines dependencies, but not the scheduling algorithm itself.

---

# 11. Configurability

A pipeline should be configurable per run.

Example job config:

```yaml
run_name: image_generation_mvp

prompt_source:
  type: jsonl
  path: prompts/image_prompts.jsonl

models:
  - Z-Image
  - FLUX.1-dev

pipeline:
  enable_postprocessing: false
  enable_labeling: false
  enable_artifact_validation: true

execution:
  retry_limit: 3
  batch_limit: 4
```

---

# 12. MVP Pipeline Definition

The MVP pipeline is:

```text
Prompt Source
    ↓
Prompt Load
    ↓
Prompt Validation
    ↓
Prompt Preprocessing
    ↓
Prompt Batching
    ↓
Task Expansion
    ↓
Model Execution
    ↓
Artifact Collection
    ↓
Artifact Validation
    ↓
Dataset Export
```

Not included in MVP:

* labeling
* evaluation
* distributed cluster orchestration
* web UI

---

# 13. Non-Goals

The pipeline DAG does not define:

* model adapter internals
* registry structure
* environment manager internals
* CLI command syntax
* scheduling algorithm details

Those belong to:

* `model_adapter_spec.md`
* `model_registry_spec.md`
* `env_manager_spec.md`
* `cli_spec.md`
* `scheduler_spec.md`

---

# 14. Summary

The Pipeline DAG defines the execution graph for generation jobs in the framework.

It provides:

* clear stage boundaries
* modular execution
* support for optional stages
* batch-aware task expansion
* resumable execution
* artifact-centered outputs

This DAG is intentionally designed to support both:

* a minimal MVP generation system
* future expansion into labeling, evaluation, and dataset curation
