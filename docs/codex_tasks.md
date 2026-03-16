# codex_tasks.md

## 1. Purpose

This document breaks the project into **concrete implementation tasks** for Codex.

It is not another architecture spec.  
It is the **execution plan** that connects the existing documentation to actual code delivery.

This document defines:

- MVP scope
- implementation phases
- task ordering
- dependencies
- acceptance criteria
- model integration order
- what must work before implementation is considered successful

This is the primary task-planning document for coding.

---

# 2. MVP Definition

The MVP must support the following end-to-end workflow:

1. user provides a prompt list as:
   - `.txt` file, one prompt per line
   - or `.csv` file
   - or `.jsonl` file
2. framework loads and validates prompts
3. framework resolves selected models from registry
4. framework ensures required Conda environments exist
5. framework runs real model inference
6. framework collects real artifacts
7. framework exports dataset records with correct traceability

The MVP must support **all currently listed target models**.

---

## 2.1 Target Models Required in MVP

### Image Models
- FLUX.1-dev
- stable-diffusion-xl-base-1.0
- Qwen-Image-2512
- Z-Image-Turbo
- Z-Image
- HunyuanImage-3.0

### Video Models
- LongCat-Video
- Wan2.2-TI2V-5B
- Wan2.2-T2V-A14B-Diffusers
- MOVA-720p
- HunyuanVideo-1.5

These models must be:

- registered
- environment-resolved
- adapter-backed
- executable through the framework
- capable of producing real artifacts

---

## 2.2 MVP Input Requirements

The MVP must accept prompt input in the following forms:

### TXT
Format:

```text
a futuristic city at night
a cat sitting on a chair
一只可爱的猫
````

Rules:

* one prompt per line
* blank lines ignored
* prompt IDs auto-generated if not provided

### CSV

Minimum supported forms:

#### Form A

```csv
prompt
a futuristic city at night
a cat sitting on a chair
```

#### Form B

```csv
prompt_id,prompt,language
p001,a futuristic city at night,en
p002,一只可爱的猫,zh
```

Rules:

* if `prompt_id` missing, auto-generate
* if `language` missing, infer or default
* support UTF-8

### JSONL

Already defined in `prompt_spec.md`

---

## 2.3 MVP Success Criteria

The MVP is complete only if all of the following are true:

1. `aigc models list` shows all target models
2. `aigc models inspect <model>` works for all target models
3. prompt loader supports `.txt`, `.csv`, and `.jsonl`
4. `aigc run --models ... --prompts ...` works for image jobs
5. `aigc run --models ... --prompts ...` works for video jobs
6. environments are created automatically
7. at least one batch-capable image model actually runs in batch mode
8. all target models have real integration paths
9. each generated artifact maps to:

   * `prompt_id`
   * `model_name`
   * `run_id`
10. dataset export works and follows `dataset_schema.md`

---

# 3. Implementation Principles

## 3.1 MVP-first

Do not implement future systems before the MVP path works.

Prioritize:

* real execution
* correctness
* traceability
* inspectability

---

## 3.2 Real model integration over abstraction

A model is not considered integrated unless it can actually run.

Not enough:

* adapter skeleton
* registry entry
* placeholder output

Required:

* real artifact generation
* real end-to-end execution path

---

## 3.3 One successful path first

Before optimizing or generalizing:

1. make one image model work
2. make one video model work
3. then generalize the framework

---

## 3.4 Respect subsystem boundaries

Use the docs as authority:

* `spec.md`
* `prompt_spec.md`
* `pipeline_dag_spec.md`
* `model_adapter.md`
* `model_registry_spec.md`
* `scheduler_spec.md`
* `dataset_schema.md`
* `cli_spec.md`
* `env_manager_spec.md`

Do not mix responsibilities across subsystems.

---

# 4. Global Dependency Order

Implementation should follow this order:

```text
project skeleton
  ↓
prompt loading
  ↓
model registry
  ↓
environment manager
  ↓
worker runtime
  ↓
adapter base classes
  ↓
first image adapter
  ↓
first video adapter
  ↓
scheduler core
  ↓
dataset export
  ↓
CLI wiring
  ↓
remaining model integrations
```

---

# 5. Phase Breakdown

## Phase 0 — Repository Skeleton

### Goal

Create the minimal code structure required for implementation.

### Tasks

#### T0.1 Create source tree

Suggested structure:

```text
src/
  aigc/
    cli/
    prompts/
    registry/
    adapters/
    env/
    runtime/
    scheduler/
    exporters/
    utils/
tests/
configs/
envs/
```

### Done when

* package imports work
* module directories exist
* test runner can discover tests

---

## Phase 1 — Prompt Input System

### Goal

Support `.txt`, `.csv`, and `.jsonl` prompt inputs.

### Tasks

#### T1.1 Implement TXT prompt loader

Rules:

* one prompt per line
* ignore blank lines
* auto-generate `prompt_id`

#### T1.2 Implement CSV prompt loader

Support:

* `prompt`
* optional `prompt_id`
* optional `language`
* optional `negative_prompt`

#### T1.3 Implement JSONL prompt loader

Follow `prompt_spec.md`

#### T1.4 Implement prompt normalization

* strip whitespace
* ignore blank lines
* normalize encoding assumptions

#### T1.5 Implement prompt validation

* prompt not empty
* unique `prompt_id`
* language field handled

### Done when

* `.txt` works
* `.csv` works
* `.jsonl` works
* prompt records conform to `prompt_spec.md`

---

## Phase 2 — Model Registry

### Goal

Load all target models from registry definitions.

### Tasks

#### T2.1 Implement registry loader

* load model config
* expose list / inspect APIs

#### T2.2 Register all MVP models

Must include all target models listed in section 2.1

#### T2.3 Bind models to adapters

Even if adapter not fully implemented yet, registry structure must exist

### Done when

* `aigc models list` works
* `aigc models inspect <model>` works
* all target models are discoverable

---

## Phase 3 — Environment Manager

### Goal

Automatically create and validate Conda environments per model or per env spec.

### Tasks

#### T3.1 Implement env spec resolution

Map model → env spec

#### T3.2 Implement deterministic environment identity

Use env-spec-based identity

#### T3.3 Implement environment existence checks

#### T3.4 Implement environment creation

From `environment.yml` and optional pip requirements

#### T3.5 Implement environment validation

Smoke-test imports

#### T3.6 Implement command wrapping under Conda

For worker launches and external-process models

### Done when

* selecting a model causes its environment to be resolved automatically
* missing environments are created automatically
* `aigc doctor` can report env readiness

---

## Phase 4 — Runtime Worker

### Goal

Create an environment-scoped worker that executes tasks.

### Tasks

#### T4.1 Implement task worker entrypoint

Worker should:

* load task payload
* load adapter
* run task
* write task result

#### T4.2 Implement result serialization

Worker returns:

* status
* artifacts
* logs
* metadata

#### T4.3 Implement workdir isolation

One workdir per task

### Done when

* one worker can run one task in an isolated workdir
* outputs and logs are captured

---

## Phase 5 — Adapter Base System

### Goal

Implement the adapter abstraction correctly.

### Tasks

#### T5.1 Implement base adapter interface

Follow `model_adapter.md`

#### T5.2 Implement execution plan objects

#### T5.3 Implement artifact record types

#### T5.4 Implement batch item result mapping

### Done when

* adapter base types exist
* both execution modes are supported:

  * in-process
  * external-process

---

## Phase 6 — First Real Image Model

### Goal

Run one real image model end-to-end.

### Recommended first model

`Z-Image`

### Tasks

#### T6.1 Implement Z-Image environment spec

#### T6.2 Implement Z-Image adapter

#### T6.3 Implement artifact collection for image outputs

#### T6.4 Validate real generation with sample prompt list

### Done when

Command works:

```bash
aigc run --models Z-Image --prompts prompts/example.txt
```

And produces:

* real image artifact(s)
* dataset exportable record(s)

---

## Phase 7 — Remaining Image Models

### Goal

Integrate all MVP image models.

### Tasks

#### T7.1 FLUX.1-dev

#### T7.2 stable-diffusion-xl-base-1.0

#### T7.3 Qwen-Image-2512

#### T7.4 Z-Image-Turbo

#### T7.5 HunyuanImage-3.0

### Notes

These models may belong to different adapter families:

* Diffusers-style
* Transformers-style
* repo-specific logic

### Done when

Each model satisfies:

* registry entry exists
* environment spec exists
* adapter exists
* `aigc models inspect` works
* real run generates at least one valid image artifact

---

## Phase 8 — First Real Video Model

### Goal

Run one real video model end-to-end.

### Recommended first model

`Wan2.2-T2V-A14B-Diffusers`

### Tasks

#### T8.1 Implement environment spec

#### T8.2 Implement adapter

#### T8.3 Implement video artifact collection

#### T8.4 Extract video metadata

### Done when

Command works:

```bash
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/video_prompts.txt
```

And produces:

* real video file
* artifact metadata
* dataset export record

---

## Phase 9 — Remaining Video Models

### Goal

Integrate all remaining MVP video models.

### Tasks

#### T9.1 LongCat-Video

#### T9.2 Wan2.2-TI2V-5B

#### T9.3 MOVA-720p

#### T9.4 HunyuanVideo-1.5

### Notes

These are likely higher-complexity integrations.

Expect:

* external-process adapters
* repo-specific scripts
* heavier env specs
* special artifact collection logic

### Done when

Each model satisfies:

* registry entry exists
* environment spec exists
* adapter exists
* real run produces at least one valid video artifact

---

## Phase 10 — Scheduler Core

### Goal

Implement batching, task expansion, and run orchestration.

### Tasks

#### T10.1 Implement run object

#### T10.2 Implement task expansion

#### T10.3 Implement batch-aware task creation

#### T10.4 Implement max worker control

#### T10.5 Implement per-model concurrency limits

#### T10.6 Implement basic retry

#### T10.7 Implement resume support

### Done when

* multiple models can be run in one invocation
* batch-capable models actually receive prompt batches
* failed tasks can be retried
* interrupted runs can resume

---

## Phase 11 — Dataset Export

### Goal

Export results into canonical dataset records.

### Tasks

#### T11.1 Implement dataset record builder

Follow `dataset_schema.md`

#### T11.2 Implement JSONL export

#### T11.3 Implement Parquet export

Optional if time permits in MVP, but recommended

#### T11.4 Ensure prompt → artifact mapping correctness

### Done when

* export file exists
* one record per artifact
* all required fields present

---

## Phase 12 — CLI MVP

### Goal

Implement the user-facing CLI defined in `cli_spec.md`

### Tasks

#### T12.1 `aigc models list`

#### T12.2 `aigc models inspect <model>`

#### T12.3 `aigc run --models ... --prompts ...`

#### T12.4 `aigc runs list`

#### T12.5 `aigc runs inspect <run_id>`

#### T12.6 `aigc runs failures <run_id>`

#### T12.7 `aigc runs retry <run_id>`

#### T12.8 `aigc runs resume <run_id>`

#### T12.9 `aigc export dataset <run_id>`

#### T12.10 `aigc doctor`

### Done when

All MVP commands work and align with `cli_spec.md`

---

# 6. Model Integration Matrix

## 6.1 Required MVP Matrix

| Model                        | Registry |      Env |  Adapter | Real Run |   Export |
| ---------------------------- | -------- | -------: | -------: | -------: | -------: |
| FLUX.1-dev                   | required | required | required | required | required |
| stable-diffusion-xl-base-1.0 | required | required | required | required | required |
| Qwen-Image-2512              | required | required | required | required | required |
| Z-Image-Turbo                | required | required | required | required | required |
| Z-Image                      | required | required | required | required | required |
| HunyuanImage-3.0             | required | required | required | required | required |
| LongCat-Video                | required | required | required | required | required |
| Wan2.2-TI2V-5B               | required | required | required | required | required |
| Wan2.2-T2V-A14B-Diffusers    | required | required | required | required | required |
| MOVA-720p                    | required | required | required | required | required |
| HunyuanVideo-1.5             | required | required | required | required | required |

A model is not MVP-complete until all five columns are satisfied.

---

# 7. Input Support Tasks

Because you explicitly require `.txt` and `.csv`, these are mandatory MVP tasks.

## TXT Support

Done when:

```bash
aigc run --models Z-Image --prompts prompts/example.txt
```

works successfully.

## CSV Support

Done when:

```bash
aigc run --models Z-Image --prompts prompts/example.csv
```

works successfully.

CSV must support at least:

* `prompt`
* optional `prompt_id`
* optional `language`

---

# 8. Acceptance Criteria by Subsystem

## Prompt Input System

* TXT supported
* CSV supported
* JSONL supported
* prompt IDs generated when absent

## Registry

* all target models discoverable
* inspect output correct

## Environment Manager

* envs auto-created
* envs validated
* tasks launched under correct env

## Adapter Layer

* supports in-process and external-process
* supports batch-capable models
* correct prompt-to-output mapping

## Scheduler

* supports multi-model runs
* supports batching
* supports retry and resume

## Export

* one record per artifact
* traceable to prompt_id/model/run_id

## CLI

* matches `cli_spec.md`
* usable for real runs

---

# 9. Suggested Execution Order for Codex

Codex should work in this order:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9
11. Phase 10
12. Phase 11
13. Phase 12

Do not jump directly to full scheduler or all model integrations before the first real end-to-end run succeeds.

---

# 10. Deferred Scope

The following are explicitly deferred beyond MVP:

* automatic safety labeling
* semantic labeling
* quality scoring
* distributed cluster execution
* remote APIs / model serving
* web UI
* advanced evaluation metrics
* prompt generation by LLM
* active dataset balancing

Do not spend MVP time on these.

---

# 11. First Concrete Milestones

## Milestone A

One image model runs end-to-end from `.txt`

Example:

```bash
aigc run --models Z-Image --prompts prompts/example.txt
```

## Milestone B

Two image models run from `.csv`

Example:

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts/example.csv
```

## Milestone C

One video model runs end-to-end

Example:

```bash
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/video_prompts.txt
```

## Milestone D

All image models integrated

## Milestone E

All video models integrated

## Milestone F

All target MVP models integrated and exportable

---

# 12. Final Definition of MVP Completion

The MVP is complete only when all of the following are true:

1. prompt loading works for:

   * `.txt`
   * `.csv`
   * `.jsonl`
2. all target models are registered
3. all required environments are auto-managed
4. all target models can be executed through the framework
5. image models generate real images
6. video models generate real videos
7. dataset export works
8. CLI commands work
9. retries and resume exist at basic usable level
10. outputs are traceable and conform to schema

---

# 13. Summary

This task plan converts the architecture and subsystem specs into a concrete execution plan for implementation.

It should be used as the coding roadmap for Codex.

The priority is not elegance first.
The priority is a **real, functioning MVP** that can run the required models on real prompt lists and produce real dataset outputs.

