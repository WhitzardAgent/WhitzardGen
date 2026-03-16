```markdown
# AGENT.md

## Purpose

This document gives an AI coding agent a high-level understanding of:

- what this project is
- what the documentation set means
- which docs are authoritative for which subsystem
- how to approach implementation without introducing architectural drift

This repository contains the design for a **multimodal AIGC synthetic data generation framework**.

The framework is intended to:

- integrate heterogeneous open-source generative models
- run them under Conda-managed isolated environments
- support both script-based and in-process model execution
- batch prompts where supported
- generate structured artifact datasets
- serve as the foundation for a larger AIGC safety data pipeline

The immediate implementation goal is an **MVP** that can actually run the target models and produce real artifacts with traceable metadata.

---

# Project Summary

The framework is designed around the following concepts:

- **Prompt datasets** are the input
- **Model registry** defines what models exist
- **Model adapters** define how each model is executed
- **Scheduler** expands prompts into tasks and dispatches them
- **Environment manager** ensures the right Conda env exists and is used
- **Pipeline DAG** defines execution stages
- **Dataset schema** defines exported output records
- **CLI** is the primary user interface

This is not a general model serving platform.  
It is a **dataset generation framework**.

The MVP does **not** include:

- automated labeling
- web UI
- distributed cluster support
- remote inference services

---

# Documentation Map

The `docs/` directory is the source of truth.

## docs/spec.md

Top-level system overview.

Use this to understand:

- project goals
- system architecture
- major subsystems
- MVP scope

This is the best starting point for project orientation.

---

## docs/prompt_spec.md

Defines how prompts are represented and processed.

Use this for:

- prompt record structure
- language handling
- negative prompts
- batching semantics
- prompt → artifact traceability

Any prompt loading, preprocessing, or export logic must follow this document.

---

## docs/pipeline_dag_spec.md

Defines the execution stages and dependencies of the generation pipeline.

Use this for:

- pipeline stage boundaries
- required vs optional stages
- task expansion flow
- artifact flow
- MVP stage scope

This document is important for structuring runtime orchestration.

---

## docs/model_adapter.md

Defines the adapter abstraction between framework and model implementation.

Use this for:

- execution ownership boundaries
- in-process vs external-process execution
- batch-capable model behavior
- artifact collection
- metadata extraction
- prompt-to-output mapping

This is the most important document for model integration work.

---

## docs/model_registry_spec.md

Defines how models are registered and discovered.

Use this for:

- model metadata
- adapter binding
- runtime requirements
- capability exposure
- model lookup

Do not hard-code model logic outside the registry unless absolutely necessary.

---

## docs/scheduler_spec.md

Defines scheduler behavior.

Use this for:

- task creation
- batching rules
- concurrency control
- retry policy
- run lifecycle tracking
- task state transitions

This should guide implementation of the orchestration core.

---

## docs/dataset_schema.md

Defines the canonical export format.

Use this for:

- output record structure
- artifact metadata layout
- prompt/model/run traceability
- JSONL and Parquet export assumptions

Any exported records must conform to this document.

---

## docs/cli_spec.md

Defines the user-facing CLI surface.

Use this for:

- supported commands
- flags
- output expectations
- MVP CLI scope

The implementation should match this spec closely.

---

## docs/env_manager_spec.md

Defines Conda environment management behavior.

Use this for:

- environment spec resolution
- environment lifecycle
- validation
- worker launching under Conda
- environment-aware execution model

This document is essential because the project relies on environment isolation.

---

# References

If present, `docs/references/` contains model-specific documentation and copied READMEs.

Examples may include:

- Hugging Face model card excerpts
- repository README files
- usage examples
- installation notes
- supported parameters

These files are reference material for implementing actual adapters and environment specs.

They are not higher-level framework specs, but they are authoritative for model-specific execution behavior.

When integrating a model:

1. read `docs/model_adapter.md`
2. read `docs/model_registry_spec.md`
3. read the relevant file in `docs/references/`
4. implement adapter and registry entry accordingly

---

# Engineering Rules

## Rule 1: Respect execution boundaries

Do not blur:

- scheduler responsibilities
- adapter responsibilities
- environment manager responsibilities
- CLI responsibilities

Keep each subsystem aligned with its spec.

---

## Rule 2: Prefer framework-level consistency over one-off hacks

If a model needs special handling, first ask:

- should this be represented as adapter capability?
- should this be a registry field?
- should this be an environment-spec difference?

Avoid hard-coded branches scattered across the codebase.

---

## Rule 3: Treat artifacts as first-class outputs

The system is artifact-centric.

Every successful generation must produce:

- a real artifact file
- traceable prompt_id
- model_name
- run_id
- metadata record

No fake placeholder outputs.

---

## Rule 4: Batch where appropriate

Some models support multi-prompt batch inference.

This must be represented as:

- adapter capability
- scheduler behavior
- batch-aware result mapping

Do not force all models into batch mode.

---

## Rule 5: Environment isolation is mandatory

The framework assumes Conda-based isolation.

Do not assume:

- one global Python environment
- in-process model loading inside the main scheduler process
- dependency compatibility between unrelated models

Prefer environment-scoped workers.

---

## Rule 6: MVP first

The MVP should prioritize:

- correctness
- end-to-end execution
- real model integration
- minimal but stable CLI
- simple and inspectable runtime state

Do not implement future features prematurely.

---

# Recommended Implementation Order

1. basic project skeleton
2. model registry loader
3. prompt loader / validator
4. environment manager
5. task worker launcher
6. adapter base classes
7. first image adapter
8. first video adapter
9. scheduler core
10. dataset export
11. CLI wiring

---

# Initial Target Models for MVP

The MVP should prioritize models that cover the main execution patterns.

Recommended order:

## First wave
- Z-Image
- FLUX.1-dev
- SDXL
- HunyuanImage-3.0

These cover:
- diffusers-style in-process image models
- transformers-style image generation
- batch-capable image generation

## Second wave
- Wan2.2-T2V-A14B-Diffusers

This gives:
- a real video model
- a first video generation adapter path

## Third wave
- LongCat-Video
- HunyuanVideo-1.5
- MOVA-720p
- Wan2.2-TI2V-5B

These validate:
- script-based and heavier video integrations

---

# What the Agent Should Do First

When starting implementation:

1. read `docs/spec.md`
2. read `docs/model_adapter.md`
3. read `docs/model_registry_spec.md`
4. read `docs/env_manager_spec.md`
5. read `docs/scheduler_spec.md`
6. inspect `docs/references/` for actual model usage examples
7. propose or create the repo skeleton
8. implement the minimal runtime path for one real image model
9. iterate toward multi-model support

---

# Success Criteria for Early Implementation

The system is on the right path when all of the following are true:

- models can be discovered from the registry
- prompt datasets load correctly
- environments are created automatically
- one image model runs end-to-end
- one video model runs end-to-end
- artifacts are exported into the dataset schema
- the CLI can launch and inspect runs

---

# Final Note

If there is any ambiguity between framework design and model-specific docs:

- follow the framework specs for system behavior
- follow the model reference docs for model invocation details

Never invent model invocation behavior that contradicts the actual model docs.
