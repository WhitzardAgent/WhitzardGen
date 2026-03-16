
# START_HERE.md

## Purpose

This file tells an AI coding agent exactly how to begin implementation with minimal confusion.

Read this file before writing any code.

---

# 1. First Read These Docs

Read these documents in this exact order:

1. `docs/spec.md`
2. `docs/model_adapter.md`
3. `docs/model_registry_spec.md`
4. `docs/env_manager_spec.md`
5. `docs/prompt_spec.md`
6. `docs/pipeline_dag_spec.md`
7. `docs/scheduler_spec.md`
8. `docs/dataset_schema.md`
9. `docs/cli_spec.md`

These documents define the intended architecture.

Do not start implementation before reading them.

---

# 2. Then Check Model References

If available, inspect:

```text
docs/references/
````

Look for model-specific materials such as:

* Hugging Face model cards
* repository README files
* copied usage examples
* install instructions
* inference scripts

These are necessary to implement real adapters.

---

# 3. Primary MVP Objective

The first practical goal is **not** to build the entire framework.

The first practical goal is to build the **minimum end-to-end execution path** for real model runs.

That means:

* load prompts
* resolve model from registry
* ensure Conda environment exists
* execute one task for one model
* collect real artifacts
* export one dataset record

Do this first for an image model.

Then extend to a video model.

---

# 4. Recommended MVP Model Order

Implement in this order:

## Step 1: Image models

* Z-Image
* FLUX.1-dev
* SDXL
* HunyuanImage-3.0

## Step 2: Video model

* Wan2.2-T2V-A14B-Diffusers

## Step 3: Script-heavy video models

* LongCat-Video
* HunyuanVideo-1.5
* MOVA-720p
* Wan2.2-TI2V-5B

This order minimizes risk while covering the major execution patterns.

---

# 5. First Components To Build

Build these first:

## 5.1 Core package skeleton

Create a clean internal structure for:

* registry
* adapters
* env manager
* scheduler
* runtime worker
* CLI
* exporters

Do not over-engineer.

---

## 5.2 Prompt loader

Implement:

* JSONL prompt loading
* schema validation
* language field handling
* `prompt_id` uniqueness checks

Follow `docs/prompt_spec.md`.

---

## 5.3 Model registry loader

Implement:

* loading model registry definitions
* listing models
* resolving adapter class
* exposing model capabilities

Follow `docs/model_registry_spec.md`.

---

## 5.4 Environment manager

Implement:

* mapping model → env_spec
* automatic Conda env creation
* validation
* environment-aware worker launching

Follow `docs/env_manager_spec.md`.

This is critical.

---

## 5.5 Adapter base classes

Implement the adapter abstractions exactly as described in:

```text
docs/model_adapter.md
```

Respect:

* prepare / execute / collect boundaries
* in-process vs external-process modes
* batch support
* prompt-to-output mapping

---

## 5.6 First real adapter

Implement one real image adapter first.

Recommended first choice:

```text
Z-Image
```

Why:

* important target model
* diffusers-style
* good multilingual support
* useful for validating batch-capable image execution

---

## 5.7 Worker runtime

Implement a worker process that:

* runs under the correct Conda environment
* loads adapter
* executes task
* writes outputs to workdir
* serializes task result

This is needed because environment isolation is essential.

---

# 6. Recommended Repo Structure

A reasonable MVP code structure may look like:

```text
src/
  aigc/
    __init__.py
    cli/
    registry/
    adapters/
    env/
    scheduler/
    runtime/
    prompts/
    exporters/
    models/
tests/
configs/
envs/
docs/
```

This is only a suggestion. Keep it simple.

---

# 7. What To Avoid At The Beginning

Do **not** start with:

* labeling system
* distributed workers
* web UI
* optimization before correctness
* complicated plugin architecture beyond what is necessary
* general-purpose orchestration abstractions with no real model run

The first milestone must be a **real successful run**.

---

# 8. First Success Milestone

The first meaningful milestone is:

## Milestone A

A command like this should work:

```bash
aigc run --models Z-Image --prompts prompts/example.jsonl
```

And produce:

* a run directory
* at least one generated artifact
* metadata linking artifact to prompt_id
* a dataset export record

Until this works, nothing else matters.

---

# 9. Second Success Milestone

## Milestone B

A command like this should work:

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts/example.jsonl
```

With:

* model registry-driven resolution
* separate task expansion per model
* batch behavior where supported
* exported records for both models

---

# 10. Third Success Milestone

## Milestone C

A command like this should work:

```bash
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/video_example.jsonl
```

With:

* real video artifact output
* correct artifact metadata
* exported dataset record

---

# 11. Fourth Success Milestone

## Milestone D

At least one script-driven model should run successfully:

```text
LongCat-Video
or
HunyuanVideo-1.5
or
MOVA-720p
```

This validates the external-process path.

---

# 12. Implementation Strategy

Use this strategy:

1. make the smallest path work
2. verify with a real model
3. generalize only after real success
4. keep specs aligned while coding

If in doubt, prefer simple explicit code over speculative abstraction.

---

# 13. What To Produce Early

The first code deliverables should likely be:

* prompt loader
* registry loader
* env manager skeleton
* adapter base classes
* worker executor
* Z-Image adapter
* minimal CLI `models list` and `run`

---

# 14. Final Reminder

The framework must remain aligned with the docs.

If implementation pressure creates ambiguity:

* do not guess blindly
* inspect `docs/`
* inspect `docs/references/`
* choose the simplest design that preserves the documented boundaries