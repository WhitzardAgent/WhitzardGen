# Model Integration Checklist

## Purpose

This document is a practical checklist for adding or hardening a model integration in WhitzardGen.

Use it when you are:

- adding a new model
- moving a model from placeholder to real execution
- switching a model between external-process and in-process execution
- hardening a cluster deployment for an existing model

This is not a replacement for:

- [model_adapter.md](/Users/morinop/coding/whitzardgen/docs/model_adapter.md)
- [model_registry_spec.md](/Users/morinop/coding/whitzardgen/docs/model_registry_spec.md)
- [runtime_spec.md](/Users/morinop/coding/whitzardgen/docs/runtime_spec.md)

Instead, this file is the operational checklist that helps turn those specs into a working model integration.

## High-Level Flow

Recommended order:

1. understand the model's real inference path
2. choose the correct execution pattern
3. add the registry entry
4. add machine-local deployment fields
5. prepare and validate the Conda env
6. implement or reuse the adapter
7. verify prompt-to-artifact mapping
8. verify batch behavior and worker strategy
9. add tests
10. do mock validation first, then real cluster validation

## 1. Understand the Real Inference Path

Before writing code, identify which of these patterns the model actually wants:

- diffusers in-process pipeline
- repo-based Python in-process pipeline
- external-process repo script
- torchrun / distributed script

Questions to answer first:

- what is the true Python entrypoint?
- does it support batching?
- does it support negative prompts?
- does it support deterministic seeds?
- what files are required locally?
- what parts belong to:
  - repository checkout
  - weights directory
  - optional cache directory

Do not guess from model names alone. Follow the reference docs and actual inference code.

## 2. Choose the Execution Pattern

Use the simplest correct pattern.

### Prefer in-process when:

- the model has a stable Python API
- one loaded pipeline can serve many prompts
- persistent workers materially reduce repeated startup cost

### Prefer external-process when:

- the repo only provides a stable script path
- the execution stack relies on fragile CLI orchestration
- distributed launch is mandatory and not yet represented cleanly in-process

### Persistent worker suitability

The best candidates are heavy in-process models where:

- model load is expensive
- many prompts will be processed in one run
- one loaded pipeline can be reused safely

## 3. Add the Registry Entry

Edit the matching registry fragment under [configs/models](/Users/morinop/coding/whitzardgen/configs/models):

- `t2i.yaml`
- `t2v.yaml`
- `t2t.yaml`
- `t2a.yaml`

Required fields typically include:

- `version`
- `adapter`
- `modality`
- `task_type`
- `capabilities`
- `runtime`
- `weights`

Typical capability fields:

- `supports_batch_prompts`
- `max_batch_size`
- `preferred_batch_size`
- `supports_negative_prompt`
- `supports_seed`
- `output_types`
- `supported_languages`

Typical runtime fields:

- `execution_mode`
- `gpu_required`
- `min_vram_gb`
- `recommended_vram_gb`
- `env_spec`
- `worker_strategy`
- `gpus_per_replica`
- `supports_multi_replica`

Typical weights fields:

- `hf_repo`
- optional hints such as `local_dir_hint`

Keep registry entries machine-independent whenever possible.

## 4. Add Machine-Local Deployment Fields

Edit the matching local override fragment under [configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models) only when the current machine needs local deployment overrides.

Use this file for:

- `conda_env_name`
- `local_path`
- `weights_path`
- `repo_path`
- `script_root`
- `hf_cache_dir`
- `max_gpus`

Recommended rule:

- `configs/models/` defines what a model is
- `configs/local_models/` defines how this machine finds and runs it

Do not repeat registry defaults in `configs/local_models/` unless the machine truly overrides them.

## 5. Prepare and Validate the Conda Environment

Current policy:

- users prepare Conda envs manually
- the framework checks and uses them
- the framework does not auto-create envs during `aigc run`

Make sure the model has a usable env name.

Then verify:

```bash
aigc doctor --model <model_name>
```

Also inspect effective model config:

```bash
aigc models inspect <model_name>
```

Checklist:

- `conda_env_name` resolves correctly
- env exists
- local paths exist
- weights path exists
- repo path exists when required
- validation imports pass

## 6. Implement or Reuse an Adapter

Typical adapter locations:

- [src/aigc/adapters/images](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images)
- [src/aigc/adapters/videos](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos)

Recommended layout:

- shared image helpers in [src/aigc/adapters/images/base.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/base.py)
- model-specific image integrations in focused modules such as:
  - [src/aigc/adapters/images/zimage.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/zimage.py)
  - [src/aigc/adapters/images/flux.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/flux.py)
- shared video helpers in [src/aigc/adapters/videos/base.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/base.py) and [src/aigc/adapters/videos/diffusers_base.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/diffusers_base.py)
- model-specific video integrations in focused modules such as:
  - [src/aigc/adapters/videos/wan_t2v.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/wan_t2v.py)
  - [src/aigc/adapters/videos/longcat.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/longcat.py)
  - [src/aigc/adapters/videos/helios.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/helios.py)

Checklist:

- choose the right adapter base
- keep prompt-to-output mapping deterministic
- keep model loading separate from task execution where persistent workers are used
- ensure artifact collection returns stable paths and metadata
- preserve negative prompt semantics if supported
- preserve prompt-level parameter behavior

## 7. Verify Prompt-to-Artifact Mapping

Every generated artifact must remain traceable to:

- `run_id`
- `task_id`
- `prompt_id`
- `model_name`

Double-check:

- batch inference does not scramble prompt ordering
- task result `batch_items` map correctly back to prompts
- artifact type is correct
- artifact metadata is present where available

## 8. Verify Batch Behavior and Worker Strategy

If the model supports batching:

- make sure prompts with incompatible effective params are not batched together
- verify negative prompt handling for batch mode
- verify seed behavior remains correct

If the model should use persistent workers:

- load once
- run many tasks
- avoid hidden re-loads inside task execution

If the model should support multi-replica:

- set `gpus_per_replica`
- verify sharding is safe
- verify sequential warmup behavior is reasonable

## 9. Add Validation and Tests

At minimum add:

- registry load coverage
- adapter unit test coverage
- mock run-flow smoke coverage
- doctor/env coverage if new local path semantics were introduced

Useful test targets:

- correct capability flags
- batch behavior
- prompt/result mapping
- worker strategy selection
- config/path validation

## 10. Validate in This Order

### Step 1. Static config inspection

```bash
aigc models inspect <model_name>
```

### Step 2. Environment readiness

```bash
aigc doctor --model <model_name>
```

### Step 3. Canary validation

Prefer the dedicated canary command:

```bash
aigc models canary <model_name>
```

Optional forms:

```bash
aigc models canary <model_name> --mock
aigc models canary <model_name> --prompt-file prompts/canary_video.jsonl
```

This command reuses the normal run flow, but constrains it to a one-model validation path with the appropriate canary prompt file by default.

### Step 4. Real canary

Use the smallest realistic prompt file first.

### Step 5. Long-run validation

Only after canary success:

- persistent worker reuse
- batch throughput
- multi-replica behavior
- exports
- retry/resume

## Common Failure Modes

Watch for:

- wrong local path type:
  - repo path vs weights path confusion
- missing Conda env
- missing validation dependency
- output shape mismatch in batched inference
- artifact truthiness / numpy-like output assumptions
- model-library stdout corrupting worker control flow
- repeated model loading despite persistent-worker intent
- duplicated config values between registry and local overrides

## Capability and Benchmark Tracking

Keep onboarding artifacts in sync with the actual model state.

Generate the current capability matrix from the registry:

```bash
aigc models matrix --write-docs
```

This refreshes:

- [docs/model_capability_matrix.md](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.md)
- [docs/model_capability_matrix.json](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.json)

Record real cluster tuning and recommended defaults in:

- [configs/model_benchmarks.yaml](/Users/morinop/coding/whitzardgen/configs/model_benchmarks.yaml)
- [docs/model_benchmarks.md](/Users/morinop/coding/whitzardgen/docs/model_benchmarks.md)

## Definition of Done for a Model Integration

A model integration should not be considered complete until all of the following are true:

- registry entry exists and is valid
- local deployment fields are clearly defined
- `aigc models inspect` is useful
- `aigc doctor --model ...` is useful
- canary validation works
- real mode works on the target cluster
- outputs are traceable in:
  - `samples.jsonl`
  - `run_manifest.json`
  - `exports/dataset.jsonl`
- export bundles include valid artifact-bearing records
- focused tests exist

## Related Documents

- [spec.md](/Users/morinop/coding/whitzardgen/docs/spec.md)
- [model_adapter.md](/Users/morinop/coding/whitzardgen/docs/model_adapter.md)
- [model_registry_spec.md](/Users/morinop/coding/whitzardgen/docs/model_registry_spec.md)
- [runtime_spec.md](/Users/morinop/coding/whitzardgen/docs/runtime_spec.md)
- [dataset_schema.md](/Users/morinop/coding/whitzardgen/docs/dataset_schema.md)
