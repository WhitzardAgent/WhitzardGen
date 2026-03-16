# CLI Specification

## 1. Purpose

The **CLI** is the primary user interface for the framework.

It must provide a consistent and ergonomic way to:

- inspect available models
- run generation jobs
- manage runs
- export datasets
- inspect failures
- interact with the environment manager indirectly

The CLI should be:

- composable
- script-friendly
- deterministic
- stable across releases
- aligned with the system specs

This document defines:

- command groups
- command syntax
- argument conventions
- output expectations
- MVP command scope

---

# 2. Design Principles

The CLI must follow these principles.

## 2.1 Job-oriented

The CLI should focus on **running jobs**, not forcing the user to manually invoke individual model scripts.

Preferred:

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts.jsonl
````

Not preferred:

```bash
python model_a.py ...
python model_b.py ...
```

---

## 2.2 Scriptable

All commands must be usable from shell scripts and automation pipelines.

Requirements:

* machine-readable output option
* predictable exit codes
* no unnecessary interactive prompts

---

## 2.3 Stable

Command names and flags should remain stable once published.

Breaking changes should be minimized.

---

## 2.4 Explicit

The CLI should make execution scope obvious.

Users should be able to clearly see:

* which models are used
* which prompt dataset is used
* where outputs go
* what run ID is assigned

---

# 3. Top-Level Command Structure

The CLI is organized into command groups.

```text
aigc
 ├── models
 ├── run
 ├── runs
 ├── export
 ├── doctor
 └── version
```

MVP scope:

* `models`
* `run`
* `runs`
* `export`
* `doctor`
* `version`

---

# 4. Global Conventions

## 4.1 Binary Name

The CLI executable is:

```text
aigc
```

---

## 4.2 Global Flags

All commands may support:

```text
--help
--verbose
--quiet
--output [text|json]
```

### Meanings

* `--help`: show command help
* `--verbose`: show more logs
* `--quiet`: suppress non-essential output
* `--output json`: emit machine-readable structured output

Example:

```bash
aigc models list --output json
```

---

## 4.3 Exit Codes

Recommended exit code behavior:

| Exit Code | Meaning                    |
| --------- | -------------------------- |
| 0         | success                    |
| 1         | general error              |
| 2         | invalid arguments          |
| 3         | missing config/resource    |
| 4         | run failed                 |
| 5         | partial success / warnings |

---

# 5. models Command Group

The `models` group is used for model discovery and inspection.

## 5.1 models list

List all registered models.

### Syntax

```bash
aigc models list
```

### Optional Flags

```text
--modality [image|video|audio|text]
--task-type [t2i|t2v|t2a|t2t|i2v]
--output [text|json]
```

### Examples

```bash
aigc models list
aigc models list --modality image
aigc models list --task-type t2v --output json
```

### Text Output Example

```text
MODEL              MODALITY   TASK_TYPE   EXECUTION_MODE
Z-Image            image      t2i         in_process
FLUX.1-dev         image      t2i         in_process
LongCat-Video      video      t2v         external_process
```

---

## 5.2 models inspect

Show detailed information for one model.

### Syntax

```bash
aigc models inspect <model_name>
```

### Example

```bash
aigc models inspect Z-Image
```

### Example Output

```text
Model: Z-Image
Version: 1.0
Modality: image
Task Type: t2i
Adapter: ZImageAdapter
Execution Mode: in_process
Batch Support: yes
Max Batch Size: 8
HF Repo: Tongyi-MAI/Z-Image
```

---

# 6. run Command

The `run` command starts a generation run.

This is the most important CLI entrypoint.

## 6.1 run

### Syntax

```bash
aigc run --models <model_list> --prompts <prompt_file>
```

### Required Flags

```text
--models
--prompts
```

### Optional Flags

```text
--run-name <name>
--out <output_dir>
--batch-limit <int>
--max-workers <int>
--retry-limit <int>
--fail-on-invalid-prompts
--output [text|json]
```

### Flag Descriptions

* `--models`: comma-separated model names
* `--prompts`: path to prompt dataset JSONL
* `--run-name`: optional user-friendly run name
* `--out`: output directory
* `--batch-limit`: upper bound on batch size
* `--max-workers`: maximum concurrent tasks
* `--retry-limit`: retry limit per task
* `--fail-on-invalid-prompts`: abort if any prompt is invalid

### Examples

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts/image_prompts.jsonl
```

```bash
aigc run \
  --models LongCat-Video,Wan2.2-T2V-A14B-Diffusers \
  --prompts prompts/video_prompts.jsonl \
  --run-name video_mvp \
  --out runs/video_mvp \
  --max-workers 2
```

### Example Success Output

```text
Run created: run_001
Prompt file: prompts/image_prompts.jsonl
Models: Z-Image, FLUX.1-dev
Output dir: runs/run_001
Tasks scheduled: 120
```

### JSON Output Example

```json
{
  "run_id": "run_001",
  "models": ["Z-Image", "FLUX.1-dev"],
  "prompt_file": "prompts/image_prompts.jsonl",
  "tasks_scheduled": 120,
  "output_dir": "runs/run_001"
}
```

---

# 7. runs Command Group

The `runs` group manages previously created runs.

## 7.1 runs list

List all runs.

### Syntax

```bash
aigc runs list
```

### Optional Flags

```text
--status [running|success|failed|partial_success]
--output [text|json]
```

### Example

```bash
aigc runs list
```

### Example Output

```text
RUN_ID    STATUS    TASKS    CREATED_AT
run_001   success   120      2026-03-16T10:00:00Z
run_002   running   84       2026-03-16T11:10:00Z
```

---

## 7.2 runs inspect

Show detailed information about one run.

### Syntax

```bash
aigc runs inspect <run_id>
```

### Example

```bash
aigc runs inspect run_001
```

### Example Output

```text
Run ID: run_001
Status: success
Prompt File: prompts/image_prompts.jsonl
Models: Z-Image, FLUX.1-dev
Tasks: 120
Succeeded: 118
Failed: 2
Output Dir: runs/run_001
```

---

## 7.3 runs retry

Retry failed tasks in a run.

### Syntax

```bash
aigc runs retry <run_id>
```

### Optional Flags

```text
--failed-only
--task-id <task_id>
--max-workers <int>
```

### Examples

```bash
aigc runs retry run_001 --failed-only
```

```bash
aigc runs retry run_001 --task-id task_0042
```

### Behavior

* retries only failed tasks by default when `--failed-only` is provided
* preserves original run metadata
* records retry attempts in runtime state

---

## 7.4 runs resume

Resume an interrupted run.

### Syntax

```bash
aigc runs resume <run_id>
```

### Example

```bash
aigc runs resume run_002
```

### Behavior

* skips completed tasks
* continues pending tasks
* may retry interrupted tasks depending on policy

---

## 7.5 runs failures

List failed tasks for a run.

### Syntax

```bash
aigc runs failures <run_id>
```

### Optional Flags

```text
--output [text|json]
```

### Example

```bash
aigc runs failures run_001
```

### Example Output

```text
TASK_ID      MODEL            ERROR
task_0042    Z-Image          CUDA OOM
task_0057    FLUX.1-dev       artifact missing
```

---

# 8. export Command Group

The `export` group exports dataset records from a run.

## 8.1 export dataset

### Syntax

```bash
aigc export dataset <run_id>
```

### Optional Flags

```text
--format [jsonl|parquet]
--out <file_path>
--view [full|minimal]
--only-success
```

### Examples

```bash
aigc export dataset run_001 --format jsonl --out exports/run_001.jsonl
```

```bash
aigc export dataset run_001 --format parquet --view full
```

### Output Example

```text
Dataset exported:
Run ID: run_001
Format: jsonl
Path: exports/run_001.jsonl
Records: 118
```

---

# 9. doctor Command Group

The `doctor` command checks runtime readiness.

This is especially useful because the framework relies on:

* conda environments
* model dependencies
* GPU availability
* file paths

## 9.1 doctor

### Syntax

```bash
aigc doctor
```

### Optional Flags

```text
--model <model_name>
--output [text|json]
```

### Examples

```bash
aigc doctor
aigc doctor --model Z-Image
```

### Example Output

```text
Environment Check: OK
GPU Check: OK
Registry Check: OK
Model Z-Image: OK
Model LongCat-Video: missing environment
```

---

# 10. version Command

Show CLI/framework version.

### Syntax

```bash
aigc version
```

### Example Output

```text
aigc 0.1.0
```

---

# 11. Output Style

## 11.1 Text Output

Default human-readable format.

Should be:

* concise
* aligned
* stable
* not overly verbose

---

## 11.2 JSON Output

Used for scripting and tooling integration.

Example:

```bash
aigc runs inspect run_001 --output json
```

Example result:

```json
{
  "run_id": "run_001",
  "status": "success",
  "tasks_total": 120,
  "tasks_success": 118,
  "tasks_failed": 2
}
```

---

# 12. MVP Command Scope

The MVP should support the following commands only:

```text
aigc models list
aigc models inspect <model_name>

aigc run --models ... --prompts ...

aigc runs list
aigc runs inspect <run_id>
aigc runs retry <run_id>
aigc runs resume <run_id>
aigc runs failures <run_id>

aigc export dataset <run_id>

aigc doctor
aigc version
```

This is sufficient to support:

* model discovery
* run creation
* run management
* dataset export
* environment debugging

---

# 13. Non-Goals

The CLI does not define:

* scheduler internals
* pipeline DAG logic
* adapter behavior
* model registry storage
* environment manager implementation details

These belong to other specifications.

---

# 14. Future Extensions

Possible future CLI additions:

* `aigc prompts ...`
* `aigc labels ...`
* `aigc env ...` (if direct env ops are ever exposed)
* `aigc eval ...`
* `aigc config ...`

These are not part of MVP.

---

# 15. Summary

The CLI provides the main operational interface for the generation framework.

It is designed to support:

* model inspection
* job execution
* run lifecycle management
* dataset export
* runtime diagnostics

This command structure is intentionally minimal, stable, and aligned with the MVP architecture.