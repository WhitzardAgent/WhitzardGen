# AIGC Framework MVP

Multimodal AIGC synthetic data generation framework for image and video dataset generation.

The current repository is optimized for:

- lightweight local development in `mock` mode
- structured run/export/diagnostic flow
- later real execution on a GPU cluster with local model-path overrides

## Features

- prompt loading from `.txt`, `.csv`, and `.jsonl`
- model registry for all current MVP image/video targets
- local override config via `configs/local_models.yaml`
- local env-install override config via `configs/local_envs.yaml`
- mock-capable image and video adapters for local testing
- JSONL dataset export
- per-run manifests and failure summaries
- basic run inspection and environment diagnostics

## Install

### Option 1: recommended

```bash
pip install -r requirements.txt
```

### Option 2: editable install

```bash
pip install -e .
```

### Option 3: traditional install

```bash
python setup.py develop
```

If you previously installed an older broken build, reinstall after pulling the latest changes:

```bash
pip uninstall -y aigc
pip install -e .
```

If a cluster still reports a console-script import error after reinstall, prefer this clean sequence:

```bash
pip uninstall -y aigc
rm -rf *.egg-info
pip install --no-build-isolation -e .
```

After installation:

```bash
aigc version
```

## Quick Start

List models:

```bash
aigc models list
```

Inspect a model:

```bash
aigc models inspect Z-Image
```

Run a local mock image job:

```bash
aigc run --models Z-Image --prompts prompts/example.txt --execution-mode mock
```

Run a local mock video job:

```bash
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/video_example.txt --execution-mode mock
```

## Canary Prompts

For smoke validation, use:

- `prompts/canary_image.txt`
- `prompts/canary_image.csv`
- `prompts/canary_image.jsonl`
- `prompts/canary_video.txt`
- `prompts/canary_video.csv`
- `prompts/canary_video.jsonl`

Examples:

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts/canary_image.txt --execution-mode mock
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/canary_video.csv --execution-mode mock
```

## Cluster Configuration

Before real execution on a GPU cluster, edit:

- `configs/local_models.yaml`
- `configs/local_runtime.yaml` if you want all run outputs under a shared root or a global default seed

Example:

```yaml
Z-Image:
  local_path: /models/Z-Image
  hf_cache_dir: /models/hf-cache

FLUX.1-dev:
  local_path: /models/FLUX.1-dev

LongCat-Video:
  repo_path: /repos/LongCat-Video
  weights_path: /models/LongCat-Video
  script_root: /repos/LongCat-Video

Wan2.2-T2V-A14B-Diffusers:
  repo_path: /repos/Wan2.2
  weights_path: /models/Wan2.2-T2V-A14B-Diffusers
  hf_cache_dir: /models/hf-cache

HunyuanVideo-1.5:
  local_path: /models/HunyuanVideo-1.5
  repo_path: /repos/HunyuanVideo-1.5
  weights_path: /models/HunyuanVideo-1.5
```

For `Wan2.2-T2V-A14B-Diffusers`, the two path types are intentionally different:

- `repo_path`: local checkout of the `Wan2.2` GitHub repository
- `weights_path` or `local_path`: local Diffusers weights directory for `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

The Diffusers weights directory should contain files such as `model_index.json`
and `vae/config.json`. Pointing `weights_path` at the raw non-Diffusers Wan
checkpoint directory will fail during `from_pretrained(...)`.

Inspect effective config:

```bash
aigc models inspect Z-Image
```

Check readiness:

```bash
aigc doctor
aigc doctor --model Z-Image
```

## Global Output Root

If you do not want runs to be created under the repository's default `runs/`
directory, edit:

- `configs/local_runtime.yaml`

Example:

```yaml
paths:
  runs_root: /shared/aigc_runs

generation:
  default_seed: 12345
```

After that, the default output location for:

- run directories
- manifests
- failures summaries
- task files
- workdirs
- artifacts
- dataset exports

will all move under `/shared/aigc_runs/<run_id>/...`.

If `generation.default_seed` is set, runs will reuse that seed unless a prompt
or explicit run parameter overrides it. If it is omitted, the framework does
not inject a fixed seed, so diffusers-based generation stays random by default.

One-off runs can still override this with:

```bash
aigc run --models Z-Image --prompts prompts/example.txt --out /tmp/my_run
```

## Run Diagnostics

List runs:

```bash
aigc runs list
```

Inspect one run:

```bash
aigc runs inspect <run_id>
```

Show failures:

```bash
aigc runs failures <run_id>
```

Locate or copy exported dataset records:

```bash
aigc export dataset <run_id>
aigc export dataset <run_id> --out /tmp/dataset.jsonl
```

Each run writes:

- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/failures.json`
- `runs/<run_id>/exports/dataset.jsonl`

## Dependency Model

The repository-level install is intentionally lightweight.

Heavy runtime dependencies such as:

- `torch`
- `diffusers`
- `transformers`
- Flash Attention variants
- model repositories and checkpoints

are expected to be handled through the per-model Conda specs under `envs/` and cluster-local model paths configured in `configs/local_models.yaml`.

Each env spec now follows a simple cluster-friendly pattern:

- `envs/<spec>/python_version.txt`
- `envs/<spec>/requirements.txt`
- optional `envs/<spec>/post_install.sh`
- optional `envs/<spec>/validation.json`

At runtime, `aigc run` / `aigc doctor` will create the environment with:

```bash
conda create --prefix <env_path> python=<version> pip
```

and then install the corresponding `requirements.txt`.

If a cluster machine cannot install packages directly from GitHub or the public
internet, use `configs/local_envs.yaml` to override env-install behavior.

Supported patterns:

- replace a specific pip requirement such as `diffusers`
- replace an exact requirement line such as `git+https://github.com/huggingface/diffusers`
- provide extra pip install args such as `--no-index` / `--find-links`
- replace the entire requirements file for one env spec
- reuse an already-built Conda prefix and skip environment creation entirely

Example:

```yaml
envs:
  zimage:
    pip_install_args:
      - --no-index
      - --find-links
      - /shared/wheelhouse
    pip_requirement_overrides:
      diffusers: /shared/wheelhouse/diffusers-0.35.0-py3-none-any.whl
      git+https://github.com/huggingface/diffusers: /shared/wheelhouse/diffusers-0.35.0-py3-none-any.whl
  wan_t2v_diffusers:
    reuse_prefix: /shared/conda_envs/wan_diffusers_ready
```

## Test

Lightweight tests only:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

These do not require local GPU execution.
