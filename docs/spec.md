# AIGC Multimodal Data Generation Framework
# System Specification (spec.md)

## 1. Introduction

This project implements a **multimodal AIGC data generation framework** designed to generate **large-scale synthetic datasets** using a diverse set of open-source generative models.

The system orchestrates multiple models to generate:

- images
- videos
- audio
- text

The primary goal is to produce **high-quality, diverse, labeled synthetic datasets** for downstream applications such as **AIGC safety detection and evaluation**.

The framework focuses on:

- scalable dataset generation
- multi-model orchestration
- reproducibility
- prompt traceability
- modular model integration

---

# 2. Project Goals

The system aims to produce:

```

1M+ synthetic samples

```

Characteristics:

- high diversity
- multi-model generation
- structured metadata
- reproducible prompts
- traceable artifacts

Primary dataset types:

| Modality | Task |
|--------|--------|
Image | text-to-image |
Video | text-to-video |
Audio | text-to-audio |
Text | text-to-text |

---

# 3. System Architecture

The system consists of several major components.

```

CLI
↓
Scheduler / Engine
↓
Pipeline Execution
↓
Model Registry
↓
Model Adapter
↓
Model Execution
↓
Artifact Storage

```

Major components:

1. CLI interface
2. Prompt dataset loader
3. Scheduler / execution engine
4. Model registry
5. Model adapters
6. Execution environment manager
7. Artifact storage system

---

# 4. Core Concepts

## 4.1 Prompt

A **prompt** is the fundamental generation input.

Prompts are defined using the schema in:

```

prompt_spec.md

````

Each prompt must include:

- prompt_id
- prompt text
- language
- optional parameters

Example:

```json
{
  "prompt_id": "p001",
  "prompt": "a futuristic cyberpunk city",
  "language": "en"
}
````

---

## 4.2 Task

A **task** represents the execution of one model on a batch of prompts.

Task definition:

```
task = (model, prompt_batch, parameters)
```

Example:

```
task_001:
  model: Z-Image
  prompts: [p001, p002, p003]
```

---

## 4.3 Artifact

An **artifact** is generated content.

Examples:

| Type  | Example |
| ----- | ------- |
| Image | PNG     |
| Video | MP4     |
| Audio | WAV     |

Artifacts must be linked to:

```
prompt_id
model
run_id
```

---

# 5. Pipeline Overview

The generation pipeline contains four stages.

```
Prompt Load
 ↓
Prompt Preprocessing
 ↓
Model Execution
 ↓
Artifact Collection
```

MVP **does NOT include labeling**.

---

## 5.1 Prompt Loading

Prompts are loaded from a dataset.

Supported formats:

```
JSONL
```

Example dataset:

```json
{"prompt_id":"p1","prompt":"a cat","language":"en"}
{"prompt_id":"p2","prompt":"一只猫","language":"zh"}
```

---

## 5.2 Prompt Preprocessing

Preprocessing includes:

* language adaptation
* prompt normalization
* prompt batching

Language translation occurs when the model does not support the prompt language.

---

## 5.3 Model Execution

Execution is handled by the **adapter + executor**.

Models fall into two categories:

### In-process models

Examples:

* Diffusers pipelines
* Transformers generation APIs

Execution occurs inside the framework process.

---

### External-process models

Examples:

* repository demo scripts
* torchrun pipelines

Execution occurs via subprocess.

---

# 6. Model Registry

All models are registered in the **Model Registry**.

Defined in:

```
model_registry_spec.md
```

The registry defines:

* model name
* adapter class
* capabilities
* runtime requirements
* weights location

Example:

```yaml
Z-Image:
  adapter: ZImageAdapter
  modality: image
  task_type: t2i
```

---

# 7. Model Adapters

Adapters translate framework tasks into model-specific execution.

Defined in:

```
model_adapter_spec.md
```

Adapters handle:

* input preparation
* execution configuration
* artifact collection
* metadata extraction

---

# 8. Prompt Batching

Many models support batch inference.

Examples:

* Diffusers pipelines
* transformer generation models

The scheduler groups prompts into batches based on adapter capabilities.

```
batch_size = min(adapter.max_batch_size, scheduler_limit)
```

---

# 9. Scheduler / Engine

The scheduler manages:

* task creation
* prompt batching
* model execution
* retries
* concurrency

Responsibilities:

| Responsibility | Description          |
| -------------- | -------------------- |
| Batching       | group prompts        |
| Scheduling     | assign tasks         |
| Retries        | retry failed prompts |
| Concurrency    | GPU utilization      |

---

# 10. Execution Environment

Models may require different dependencies.

The framework manages environments using:

```
conda environments
```

Each model specifies:

```
environment.yaml
```

Example:

```
envs/
  zimage_env.yml
  longcat_env.yml
```

The environment manager:

* creates environments automatically
* activates correct env for execution

---

# 11. Artifact Storage

Artifacts are stored in structured directories.

Example:

```
runs/
  run_001/
    Z-Image/
      p001.png
      p002.png
    LongCatVideo/
      p001.mp4
```

Artifacts must include metadata linking to prompts.

---

# 12. Dataset Export

Generated results can be exported as datasets.

Supported formats:

```
JSONL
Parquet
```

Example record:

```json
{
  "prompt_id": "p001",
  "model": "Z-Image",
  "artifact": "runs/run1/zimage/p001.png"
}
```

---

# 13. CLI Interface

CLI provides the primary user interface.

Example commands:

### Run generation

```
aigc run \
  --models Z-Image,LongCat-Video \
  --prompts prompts.jsonl
```

### List models

```
aigc models list
```

### Inspect model

```
aigc models inspect Z-Image
```

---

# 14. Error Handling

Common failure cases:

* CUDA OOM
* model crash
* artifact missing

Retries:

```
max_retries = 3
```

Failed prompts are logged and may be retried.

---

# 15. Non-Goals (MVP)

The MVP **does not include**:

* automated safety labeling
* distributed cluster scheduling
* web UI
* remote model APIs

These may be added later.

---

# 16. Future Extensions

Planned improvements:

* automatic prompt generation using LLMs
* safety labeling pipeline
* distributed GPU execution
* dataset evaluation tools
* prompt diversity metrics

---

# 17. Summary

This framework provides a scalable system for generating multimodal synthetic datasets using heterogeneous generative models.

Key features:

* modular model adapters
* dynamic model registry
* prompt traceability
* batch-aware execution
* structured artifact storage

```

---

# What Documentation Should Come Next (Before Implementation)

You are **very close** to being ready for implementation. However a few **critical docs are still missing**.

These are the final pieces needed to avoid architecture drift during coding.

---

# 1️⃣ CLI Specification (VERY IMPORTANT)

You already referenced CLI commands, but you need a **real CLI spec**.

Create:

```

cli_spec.md

```

It should define:

- commands
- flags
- examples
- expected outputs

Example sections:

```

run
models list
models inspect
runs list
runs inspect
resume
retry

```

This prevents CLI chaos during development.

---

# 2️⃣ Scheduler / Engine Spec

Right now the scheduler is only briefly mentioned.

You should define:

```

scheduler_spec.md

```

It should specify:

- task creation
- batching logic
- concurrency control
- retry logic
- run state tracking

This is **one of the most complex parts of the system**.

---

# 3️⃣ Artifact & Dataset Schema

Right now artifacts are loosely defined.

Create:

```

dataset_schema.md

```

Define:

```

artifact record
dataset record
run metadata
model metadata

```

This avoids breaking dataset compatibility later.

---

# 4️⃣ Environment Manager Spec

Since you're using **conda**, define:

```

env_manager_spec.md

```

Describe:

- environment creation
- caching
- activation strategy
- dependency conflicts

---

# Final Recommended Doc List

Your documentation set should be:

```

spec.md
prompt_spec.md
pipeline_dag_spec.md
model_adapter_spec.md
model_registry_spec.md
cli_spec.md
scheduler_spec.md
dataset_schema.md
env_manager_spec.md
