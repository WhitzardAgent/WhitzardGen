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

HunyuanVideo-1.5:
  local_path: /models/HunyuanVideo-1.5
  repo_path: /repos/HunyuanVideo-1.5
  weights_path: /models/HunyuanVideo-1.5
```

Inspect effective config:

```bash
aigc models inspect Z-Image
```

Check readiness:

```bash
aigc doctor
aigc doctor --model Z-Image
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

## Test

Lightweight tests only:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

These do not require local GPU execution.
