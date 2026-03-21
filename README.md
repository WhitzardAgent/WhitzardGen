# WhitzardGen

Multimodal synthetic data generation framework for image and video collection.

This repository is designed for:

- running many open-source image/video models behind one CLI
- long-running multi-model collection jobs
- persistent-worker and multi-replica execution
- traceable run artifacts, ledgers, recovery, and exports
- cluster deployment with machine-local model paths and manually prepared Conda envs

Chinese documentation: [README.zh-CN.md](/Users/morinop/coding/whitzardgen/README.zh-CN.md)

## What This Project Does

WhitzardGen is a dataset-generation framework, not a model-serving platform.

The framework turns:

- prompt files
- selected models
- cluster-local model/repo/env configuration

into:

- generated image/video artifacts
- per-run manifests and logs
- prompt-level ledgers
- recovery-capable run metadata
- organized dataset export bundles

Current focus:

- image and video generation
- prompt generation on top of a local T2T backend
- single-machine multi-GPU execution
- persistent workers for heavy models
- dataset-oriented export and organization

## Current Capabilities

- Prompt input:
  - `.txt`
  - `.csv`
  - `.jsonl`
- Rich prompt fields:
  - `prompt_id`
  - `prompt`
  - `negative_prompt`
  - `parameters`
  - `metadata`
- Profile-based runs:
  - `aigc run --profile ...`
- Prompt-generation workflow:
  - theme-tree planning
  - prompt bundle generation
  - prompt template switching
  - prompt writing style families
  - style-family few-shot examples
- Run-time features:
  - persistent workers
  - sequential replica warmup
  - multi-replica scheduling
  - live throughput / ETA monitoring
  - prompt-level sample ledger
  - retry / resume
  - failure-policy control
- Export features:
  - per-run dataset JSONL
  - organized export bundles
  - multi-run merged export
  - split-aware export layout
  - model-filtered export
  - `link` / `copy` artifact materialization

## Repository Layout

Key directories:

- [src/aigc](/Users/morinop/coding/whitzardgen/src/aigc): framework source code
- [configs/models](/Users/morinop/coding/whitzardgen/configs/models): canonical model registry split by task type (`t2i` / `t2v` / `t2t` / `t2a`)
- [configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models): machine-local model overrides split by task type
- [configs/local_runtime.yaml](/Users/morinop/coding/whitzardgen/configs/local_runtime.yaml): machine-local runtime/output defaults
- [configs/run_profiles](/Users/morinop/coding/whitzardgen/configs/run_profiles): reusable run profiles
- [prompts](/Users/morinop/coding/whitzardgen/prompts): sample and canary prompt files
- [envs](/Users/morinop/coding/whitzardgen/envs): per-model env specs and validation metadata
- [docs](/Users/morinop/coding/whitzardgen/docs): architecture and subsystem specs
- [tests](/Users/morinop/coding/whitzardgen/tests): lightweight regression coverage

## Install

Recommended:

```bash
pip install -r requirements.txt
```

Editable install:

```bash
pip install -e .
```

Check the CLI:

```bash
aigc version
```

If you previously installed an older build:

```bash
pip uninstall -y aigc
pip install -e .
```

If the console script still looks stale:

```bash
pip uninstall -y aigc
rm -rf *.egg-info
pip install --no-build-isolation -e .
```

## Environment Model

The framework no longer auto-creates Conda envs during `aigc run`.

Current policy:

- users prepare Conda envs manually
- the framework resolves which env a model should use
- the framework checks existence / light readiness
- subprocesses run under `conda run -n <env_name> ...`

Use this to inspect readiness:

```bash
aigc doctor
aigc doctor --model Z-Image
```

Use this to inspect effective config:

```bash
aigc models inspect Z-Image
```

## Configuration Model

There are three main config layers.

### 1. Model Registry

[configs/models](/Users/morinop/coding/whitzardgen/configs/models)

This directory defines what a model is. Registry fragments are grouped by task type:

- `configs/models/t2i.yaml`
- `configs/models/t2v.yaml`
- `configs/models/t2t.yaml`
- `configs/models/t2a.yaml`

Each fragment defines:

- adapter class
- modality
- task type
- capabilities
- runtime defaults
- Hugging Face repo hints

This file should remain mostly machine-independent.

### 2. Local Model Overrides

[configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models)

This directory defines how the current machine finds and runs a model. Overrides are also grouped by task type:

- `configs/local_models/t2i.yaml`
- `configs/local_models/t2v.yaml`
- `configs/local_models/t2t.yaml`
- `configs/local_models/t2a.yaml`

Use these files for:

- `conda_env_name`
- `local_path`
- `weights_path`
- `repo_path`
- `script_root`
- `hf_cache_dir`
- `max_gpus`

Recommended rule:

- `configs/models/` defines the model
- `configs/local_models/` defines machine-local deployment details

Example:

```yaml
Z-Image:
  local_path: /models/Z-Image

Wan2.2-T2V-A14B-Diffusers:
  repo_path: /repos/Wan2.2
  weights_path: /models/Wan2.2-T2V-A14B-Diffusers
  max_gpus: 4

CogVideoX-5B:
  conda_env_name: cogvideo
  weights_path: /models/CogVideoX-5B
```

### 3. Local Runtime Defaults

[configs/local_runtime.yaml](/Users/morinop/coding/whitzardgen/configs/local_runtime.yaml)

This controls machine-local runtime defaults such as:

- default run output root
- optional global default seed

Example:

```yaml
paths:
  runs_root: /shared/aigc_runs

generation:
  default_seed: 12345
```

If `generation.default_seed` is omitted, generation stays random by default unless a prompt or profile provides a seed.

### 4. Prompt Generation Config

[configs/prompt_generation](/Users/morinop/coding/whitzardgen/configs/prompt_generation)

Prompt generation now has its own explicit config layer:

- [profiles.yaml](/Users/morinop/coding/whitzardgen/configs/prompt_generation/profiles.yaml)
  - controls content-distribution pools such as scene, lighting, weather, camera, realism anchors
  - can also set the default text model for prompt synthesis via `default_llm_model`
- [templates](/Users/morinop/coding/whitzardgen/configs/prompt_generation/templates)
  - controls how the LLM is instructed to perform prompt synthesis
- [style_families](/Users/morinop/coding/whitzardgen/configs/prompt_generation/style_families)
  - controls the final prompt writing style and few-shot examples
- [target_style_mappings.yaml](/Users/morinop/coding/whitzardgen/configs/prompt_generation/target_style_mappings.yaml)
  - maps downstream AIGC model names to a default prompt writing style family

Current first-class prompt writing style families:

- `detailed_sentence`
- `keyword_list`
- `short_sentence`

Current default template and style:

- template: `photorealistic_base`
- style family: `detailed_sentence`
- generation profile: `photorealistic`
- default prompt-synthesis LLM in the stock profile: `Qwen3-32B`

Theme-tree defaults may also set:

```yaml
defaults:
  generation_profile: photorealistic
  prompt_template: photorealistic_base
  prompt_style_family: detailed_sentence
```

Resolution precedence is:

- template:
  - CLI `--template`
  - `tree.defaults.prompt_template`
  - system default template
- style family:
  - CLI `--style-family`
  - `tree.defaults.prompt_style_family`
  - `target_style_mappings[target_model]`
  - template default
- LLM model:
  - CLI `--llm-model`
  - `profiles.yaml -> profiles.<generation_profile>.default_llm_model`
  - built-in fallback

## Prompt Formats

### TXT

Minimal format, one prompt per line:

```text
a futuristic city at night
a cat sitting on a chair
一只可爱的猫
```

### CSV

Typical forms:

```csv
prompt
a futuristic city at night
```

or:

```csv
prompt_id,prompt,language,negative_prompt,parameters
p001,a futuristic city,en,"blurry","{""width"":1024}"
```

### JSONL

Recommended rich format for real collection work:

```json
{"prompt_id":"p001","prompt":"a cinematic cat in warm morning light","negative_prompt":"blurry, low quality","parameters":{"width":1024,"height":1024,"guidance_scale":4.0},"metadata":{"split":"train","topic":"animals"}}
```

Useful examples:

- [prompts/example_image_rich.jsonl](/Users/morinop/coding/whitzardgen/prompts/example_image_rich.jsonl)
- [prompts/example_video_rich.jsonl](/Users/morinop/coding/whitzardgen/prompts/example_video_rich.jsonl)
- [prompts/canary_image.jsonl](/Users/morinop/coding/whitzardgen/prompts/canary_image.jsonl)
- [prompts/canary_video.jsonl](/Users/morinop/coding/whitzardgen/prompts/canary_video.jsonl)

## Generation Parameter Precedence

Current precedence is:

```text
model defaults
< profile generation_defaults
< prompt-level parameters
```

That means:

- profile defaults can set a common run-wide baseline
- each prompt can override per-sample settings
- prompt-level values always win over profile defaults

Example profile:

```yaml
generation_defaults:
  width: 1024
  height: 1024
  guidance_scale: 4.0
  num_inference_steps: 40
```

Example prompt override:

```json
{"prompt_id":"p002","prompt":"a cat","parameters":{"width":1280}}
```

Effective `width` for `p002` becomes `1280`.

## CLI Overview

### Model Discovery

List models:

```bash
aigc models list
aigc models list --modality image
aigc models list --task-type t2v
```

Inspect one model:

```bash
aigc models inspect Z-Image
```

Run a one-model canary:

```bash
aigc models canary Z-Image --mock
aigc models canary Wan2.2-T2V-A14B-Diffusers
```

Generate the capability matrix from the current registry:

```bash
aigc models matrix --write-docs
```

### Environment Diagnostics

```bash
aigc doctor
aigc doctor --model Wan2.2-T2V-A14B-Diffusers
```

### Run Jobs

Single-model:

```bash
aigc run --models Z-Image --prompts prompts/canary_image.txt --execution-mode mock
```

Multi-model:

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts/canary_image.txt --execution-mode mock
```

Real video example:

```bash
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/canary_video.txt --execution-mode real
```

Failure policy:

```bash
aigc run \
  --models Z-Image \
  --prompts prompts/test_image_100.txt \
  --execution-mode real \
  --continue-on-error \
  --max-failures 20 \
  --max-failure-rate 0.10
```

### Run Profiles

Profiles make multi-model collection easier:

```bash
aigc run --profile configs/run_profiles/image_real.yaml
aigc run --profile configs/run_profiles/video_real.yaml
```

Examples:

- [configs/run_profiles/image_real.yaml](/Users/morinop/coding/whitzardgen/configs/run_profiles/image_real.yaml)
- [configs/run_profiles/video_real.yaml](/Users/morinop/coding/whitzardgen/configs/run_profiles/video_real.yaml)

CLI flags override profile values when both are present.

### Prompt Generation

Theme-tree planning:

```bash
aigc prompts plan --tree prompts/theme_tree_example.yaml --output json
```

Generate a prompt bundle with the default template/style-family stack:

```bash
aigc prompts generate \
  --tree prompts/theme_tree_example.yaml \
  --execution-mode mock
```

Switch prompt template explicitly:

```bash
aigc prompts generate \
  --tree prompts/theme_tree_example.yaml \
  --template documentary_scene
```

Switch prompt writing style family explicitly:

```bash
aigc prompts generate \
  --tree prompts/theme_tree_example.yaml \
  --style-family keyword_list
```

Resolve the default style family from a downstream target model:

```bash
aigc prompts generate \
  --tree prompts/theme_tree_example.yaml \
  --target-model Z-Image
```

Use real T2T synthesis with the current text backend:

```bash
aigc prompts generate \
  --tree prompts/theme_tree_example.yaml \
  --execution-mode real \
  --llm-model Qwen3-32B \
  --template photorealistic_base \
  --style-family detailed_sentence
```

Inspect a prompt bundle:

```bash
aigc prompts inspect <prompt_bundle_dir>
aigc prompts inspect <prompt_bundle_dir> --output json
```

Useful flags on `aigc prompts generate`:

- `--tree`
- `--out`
- `--count-config`
- `--llm-model`
- `--execution-mode [mock|real]`
- `--seed`
- `--profile`
- `--template`
- `--style-family`
- `--target-model`
- `--output [text|json]`

Recommended flow:

- use `aigc prompts plan` to verify quota allocation and resampling first
- use `--template` to change top-level synthesis instructions
- use `--style-family` to control final prompt writing style
- use `--target-model` only when you want automatic style-family defaults for a downstream generator
- use `--out` when you want the bundle written to an explicit local path

### Run Inspection

```bash
aigc runs list
aigc runs inspect <run_id>
aigc runs failures <run_id>
```

### Retry / Resume

Retry failed prompt outputs:

```bash
aigc runs retry <run_id>
```

Resume missing prompt outputs from an interrupted run:

```bash
aigc runs resume <run_id>
```

### Export Dataset Bundles

Single run:

```bash
aigc export dataset <run_id>
aigc export dataset <run_id> --mode link
aigc export dataset <run_id> --mode copy
```

Multi-run merged export:

```bash
aigc export dataset run_001 run_002 run_003
```

Model-filtered export:

```bash
aigc export dataset run_001 run_002 --model Z-Image --model FLUX.1-dev
```

Custom bundle output:

```bash
aigc export dataset run_001 run_002 --out /data/exports/my_bundle
```

## Run Outputs

Every run writes structured artifacts under the configured run root.

Typical files:

- `run_manifest.json`
- `failures.json`
- `samples.jsonl`
- `running.log`
- `runtime_status.json`
- `exports/dataset.jsonl`

What they mean:

- `run_manifest.json`: authoritative run summary and lineage
- `failures.json`: task-level failures
- `samples.jsonl`: append-only prompt-level success/failure ledger
- `running.log`: detailed timestamped run log
- `runtime_status.json`: live supervisor-owned telemetry snapshot
- `exports/dataset.jsonl`: per-run artifact-level export

## Prompt Bundles

`aigc prompts generate` writes a prompt bundle instead of a loose JSONL file.

Typical structure:

```text
prompt_bundle/
  prompts.jsonl
  prompt_manifest.json
  sampling_plan.json
  generation_log.jsonl
  stats.json
```

What they mean:

- `prompts.jsonl`: final prompt records used by downstream runs
- `prompt_manifest.json`: bundle-level metadata, including template/style-family/target-model resolution
- `sampling_plan.json`: quota-driven theme sampling result
- `generation_log.jsonl`: synthesis-time decision log, including few-shot selection
- `stats.json`: counts by category/subcategory/theme

Prompt records and bundle metadata now preserve prompt-generation traceability such as:

- `prompt_template`
- `prompt_template_version`
- `prompt_style_family`
- `prompt_style_family_version`
- `target_model_name`
- `few_shot_example_ids`
- `instruction_render_version`
- `resolved_style_source`

## Export Bundles

The user-facing dataset export layer now creates organized bundles.

Typical structure:

```text
dataset_bundle/
  dataset.jsonl
  export_manifest.json
  README.md
  media/
    train/
      Z-Image/
        image/
    val/
      Wan2.2-T2V-A14B-Diffusers/
        video/
    unspecified/
      ...
```

Current behavior:

- successful records only
- supports merged multi-run export
- supports model filtering
- supports `link` / `copy`
- preserves source lineage and original artifact path

## Supported Model Patterns

The framework supports several execution patterns:

- in-process image diffusers models
- in-process video diffusers models
- repo-based Python pipelines
- external-process/script-based fallback paths

Examples currently represented in the registry include:

- `Z-Image`
- `Z-Image-Turbo`
- `FLUX.1-dev`
- `stable-diffusion-xl-base-1.0`
- `Qwen-Image-2512`
- `HunyuanImage-3.0`
- `Wan2.2-T2V-A14B-Diffusers`
- `CogVideoX-5B`
- `LongCat-Video`
- `Wan2.2-TI2V-5B`
- `MOVA-720p`
- `HunyuanVideo-1.5`

Always use:

```bash
aigc models list
```

for the current authoritative set.

## Adapter Layout

Adapters are now organized by modality package instead of growing one giant image/video file.

Current structure:

- [src/aigc/adapters/images](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images)
- [src/aigc/adapters/videos](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos)

Examples:

- [src/aigc/adapters/images/zimage.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/zimage.py)
- [src/aigc/adapters/images/flux.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/flux.py)
- [src/aigc/adapters/videos/wan_t2v.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/wan_t2v.py)
- [src/aigc/adapters/videos/cogvideox.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/cogvideox.py)
- [src/aigc/adapters/videos/longcat.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/longcat.py)
- [src/aigc/adapters/videos/helios.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/helios.py)

This keeps shared logic in small base/common modules and makes new-model onboarding much easier to maintain.

## Adding a New Model

The clean path for integrating a new model is:

Before starting, also read:

- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)

### 1. Add a Registry Entry

Edit the matching registry fragment under [configs/models](/Users/morinop/coding/whitzardgen/configs/models):

- `t2i.yaml`
- `t2v.yaml`
- `t2t.yaml`
- `t2a.yaml`

- name
- version
- adapter
- modality
- task type
- capabilities
- runtime defaults
- weight hints

### 2. Add Local Deployment Overrides

Edit the matching local override fragment under [configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models) only if needed:

- `conda_env_name`
- `local_path`
- `weights_path`
- `repo_path`
- `script_root`
- `hf_cache_dir`
- `max_gpus`

### 3. Prepare the Conda Environment

Create the Conda env manually before real runs.

Then verify:

```bash
aigc doctor --model <model_name>
```

### 4. Implement or Reuse an Adapter

Typical places:

- [src/aigc/adapters/images](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images)
- [src/aigc/adapters/videos](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos)

Choose the appropriate pattern:

- diffusers in-process
- repo-based Python in-process
- external-process script runner

### 5. Add Validation and Tests

Recommended targets:

- registry load test
- adapter unit test
- mock run-flow smoke test
- doctor / env readiness expectation

### 6. Canary Validation

Start with the dedicated canary command:

```bash
aigc models canary <model_name> --mock
```

Then move to real cluster validation:

```bash
aigc models canary <model_name>
```

Useful onboarding artifacts:

- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)
- [docs/model_capability_matrix.md](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.md)
- [docs/model_capability_matrix.json](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.json)
- [configs/model_benchmarks.yaml](/Users/morinop/coding/whitzardgen/configs/model_benchmarks.yaml)
- [docs/model_benchmarks.md](/Users/morinop/coding/whitzardgen/docs/model_benchmarks.md)

## Notes for Specific Models

### Wan2.2-T2V-A14B-Diffusers

This model uses two path concepts:

- `repo_path`: local checkout of the `Wan2.2` repository
- `weights_path`: local Diffusers weights directory for `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

Do not point `weights_path` at a raw non-Diffusers Wan checkpoint directory.

### LongCat-Video

Current integration is Python in-process, intended to support:

- persistent worker reuse
- batch execution through one loaded pipeline

### CogVideoX-5B

Current integration is in-process and replica-aware. Use doctor + local model overrides to confirm the effective weights path and env name before running real jobs.

## Operational Tips

- Prefer `.jsonl` prompts for real collection jobs.
- Use profiles for repeatable image/video collection recipes.
- Use `aigc doctor` before starting a real cluster run.
- Watch `running.log` and `runtime_status.json` during long runs.
- Use `samples.jsonl` for prompt-level recovery visibility.
- Use `aigc runs retry` and `aigc runs resume` rather than manually editing old run directories.
- Use export bundles, not only per-run `exports/dataset.jsonl`, when preparing downstream datasets.

## Roadmap

Near-term roadmap:

- richer dataset-card style export summaries
- export-level dedupe and collision reporting
- stronger train/val/test export workflows
- improved quality review and filtering hooks
- better cluster-side real-run validation coverage

Mid-term roadmap:

- annotation / review pipeline integration
- data curation helpers on top of export bundles
- stronger artifact-level analytics and reporting
- broader model coverage and adapter hardening

Longer-term roadmap:

- audio / text generation support
- larger-scale cluster orchestration
- downstream evaluation / labeling integration
- more complete dataset lifecycle tooling

## More Detailed Specs

For architecture and subsystem details, read:

- [docs/spec.md](/Users/morinop/coding/whitzardgen/docs/spec.md)
- [docs/runtime_spec.md](/Users/morinop/coding/whitzardgen/docs/runtime_spec.md)
- [docs/cli_spec.md](/Users/morinop/coding/whitzardgen/docs/cli_spec.md)
- [docs/dataset_schema.md](/Users/morinop/coding/whitzardgen/docs/dataset_schema.md)
- [docs/model_registry_spec.md](/Users/morinop/coding/whitzardgen/docs/model_registry_spec.md)
- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)
- [docs/model_capability_matrix.md](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.md)
- [docs/model_benchmarks.md](/Users/morinop/coding/whitzardgen/docs/model_benchmarks.md)
- [docs/prompt_spec.md](/Users/morinop/coding/whitzardgen/docs/prompt_spec.md)
- [docs/codex_tasks.md](/Users/morinop/coding/whitzardgen/docs/codex_tasks.md)
