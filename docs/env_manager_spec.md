# Environment Manager Specification

## 1. Purpose

The **Environment Manager** is responsible for ensuring that each model can run in the correct execution environment without requiring manual environment creation by the user.

The framework uses **Conda-based environment isolation** because different models may require different versions of:

- Python
- PyTorch
- CUDA-compatible packages
- diffusers / transformers
- custom dependencies from model repositories

The Environment Manager must support:

- automatic environment creation
- environment reuse and caching
- environment resolution by model
- environment activation for execution
- environment validation
- environment health reporting

The Environment Manager is a **runtime subsystem**. It is not user-facing in MVP except through indirect CLI behavior such as `aigc run` and `aigc doctor`.

---

# 2. Design Principles

The Environment Manager must follow these principles.

## 2.1 Automatic

Users should not need to manually run:

```bash
conda create ...
conda env create ...
conda activate ...
````

The framework must manage environments automatically.

---

## 2.2 Model-scoped Isolation

Different models may use different environments.

Examples:

* `Z-Image` may share an image-generation diffusers environment
* `LongCat-Video` may require a dedicated environment
* `HunyuanImage-3.0` may require a transformers-specific environment

Environment sharing is allowed only when explicitly configured and proven compatible.

---

## 2.3 Deterministic

Environment resolution must be deterministic.

The same model version and environment definition must always map to the same environment identity.

---

## 2.4 Cached

Once an environment is created successfully, it should be reused for future runs whenever possible.

The framework should not recreate environments unnecessarily.

---

## 2.5 Observable

The system must be able to answer:

* does environment exist?
* is it healthy?
* which models use it?
* why did environment creation fail?

---

# 3. Scope

The Environment Manager is responsible for:

* locating environment specs
* computing environment identities
* checking environment existence
* creating missing environments
* validating environment readiness
* providing execution wrappers for subprocess tasks

The Environment Manager is **not responsible for**:

* selecting which model to run
* scheduling tasks
* parsing model outputs
* storing dataset records

Those responsibilities belong to other subsystems.

---

# 4. Core Concepts

## 4.1 Environment Spec

Each model must be associated with an **environment specification**.

The environment spec defines:

* conda environment name or derived identity
* Python version
* required Conda dependencies
* required pip dependencies
* optional post-install hooks

Environment specs may be shared by multiple models.

---

## 4.2 Environment Identity

Every environment must have a stable identity.

Recommended identity inputs:

* environment spec content hash
* model group or environment family name
* framework version if needed

Example:

```text
env_diffusers_image_4f28c9
```

This allows deterministic reuse and avoids collisions.

---

## 4.3 Environment Family

An environment family is a logical grouping.

Examples:

* `diffusers_image`
* `video_script_runtime`
* `transformers_image`
* `audio_generation`

A family does not guarantee shared environment usage by itself, but it helps organize specs.

---

## 4.4 Environment State

Each managed environment should have one of the following states:

```text
missing
creating
ready
invalid
failed
```

Definitions:

* `missing`: no environment currently exists
* `creating`: environment creation in progress
* `ready`: environment exists and passes validation
* `invalid`: environment exists but health check failed
* `failed`: previous creation attempt failed

---

# 5. Environment Spec Format

Environment specs should be stored as files in the repository.

Recommended layout:

```text
envs/
  diffusers_image/
    python_version.txt
    requirements.txt
    post_install.sh
  longcat_video/
    python_version.txt
    requirements.txt
  hunyuan_image/
    python_version.txt
    requirements.txt
```

---

## 5.1 python_version.txt

This is the primary Conda environment definition for the simplified MVP strategy.

The framework should create environments with a command of the form:

```bash
conda create --prefix <env_path> python=<version>
```

Example:

```text
3.10
```

---

## 5.2 requirements.txt

Model-family dependencies are installed after environment creation from a per-spec
`requirements.txt` file.

Example:

```text
torch
torchvision
xformers
sentencepiece
einops
```

---

## 5.3 post_install.sh

Optional post-install commands.

Examples:

* cloning model repo helper code
* installing local editable packages
* validating custom kernels
* performing sanity checks

This file should be used sparingly.

---

# 6. Model-to-Environment Mapping

The Model Registry must define which environment spec is associated with each model.

Example:

```yaml
Z-Image:
  env_spec: diffusers_image

FLUX.1-dev:
  env_spec: diffusers_image

LongCat-Video:
  env_spec: longcat_video

HunyuanImage-3.0:
  env_spec: hunyuan_image
```

The Environment Manager uses this mapping to resolve environment requirements at runtime.

---

# 7. Environment Resolution Flow

For each task, the framework resolves the required environment as follows:

```text
Task
  ↓
Model
  ↓
Model Registry
  ↓
env_spec
  ↓
Environment Manager
  ↓
resolved environment instance
```

Resolution steps:

1. look up `env_spec` for model
2. compute stable environment identity
3. check whether environment exists
4. if missing, create it
5. validate readiness
6. return execution context

---

# 8. Environment Creation

## 8.1 Creation Trigger

Environment creation should be triggered automatically when:

* a run includes a model whose environment is missing
* a requested environment exists but is invalid and needs rebuild

The user should not need a separate explicit command for MVP.

---

## 8.2 Creation Steps

Recommended creation pipeline:

```text
resolve spec
  ↓
conda create --prefix ... python=<version>
  ↓
install requirements.txt
  ↓
run optional post_install.sh
  ↓
run validation checks
  ↓
mark environment ready
```

---

## 8.3 Locking

Environment creation must be protected by a lock to avoid concurrent duplicate creation.

Example scenario:

* two tasks request `diffusers_image`
* only one creation process should run
* the second waiter should block or poll until creation finishes

Recommended lock granularity:

```text
one lock per environment identity
```

---

# 9. Environment Reuse

If an environment is already in `ready` state, it should be reused.

Reuse conditions:

* same environment identity
* validation still passes
* environment path still exists

The system should avoid unnecessary rebuilds.

---

# 10. Environment Validation

Validation is required both after creation and during health checks.

Validation should verify at minimum:

* conda environment exists
* Python executable is available
* critical dependencies import successfully
* expected package versions are acceptable

Examples:

### Diffusers image environment validation

* import `torch`
* import `diffusers`
* import `transformers`

### LongCat video environment validation

* import `torch`
* import any model-required package
* verify helper script exists if required

Validation should be implemented as lightweight smoke tests.

---

# 11. Execution Integration

The Environment Manager must provide execution wrappers for both execution modes.

## 11.1 External-process Execution

For script-driven models, the executor needs a subprocess command wrapped in the correct conda environment.

Recommended pattern:

```text
conda run -n <env_name> python ...
```

or an equivalent robust wrapper.

The Environment Manager should provide a function that transforms a raw command into an environment-aware command.

Example logical API:

```python
wrap_command(env_id, command) -> wrapped_command
```

---

## 11.2 In-process Execution

For in-process adapters, the framework process itself cannot dynamically switch Python environments safely.

Therefore MVP should follow this rule:

### Rule

If a model requires in-process execution, it must run inside a worker process launched under the correct Conda environment.

That means even in-process adapters should execute inside an environment-specific worker boundary.

Recommended MVP approach:

* each task launches a worker command under `conda run`
* the worker process loads the in-process adapter and executes it
* results are written to workdir
* parent framework process remains environment-agnostic

This is important because otherwise one Python process cannot reliably host multiple incompatible model environments.

---

# 12. Environment-Aware Worker Boundary

To unify execution, the framework should treat **environment-specific worker execution** as the standard runtime model.

Recommended execution pattern:

```text
Main Scheduler Process
  ↓
spawn env-scoped worker under conda run
  ↓
worker loads adapter
  ↓
worker executes task
  ↓
worker writes outputs + logs
  ↓
main process collects results
```

This approach has several advantages:

* works for script-based models
* works for in-process diffusers / transformers models
* avoids cross-environment contamination
* simplifies dependency isolation

This should be the recommended MVP execution architecture.

---

# 13. Environment Metadata Store

The framework should maintain lightweight metadata for environments.

Recommended metadata fields:

```json
{
  "env_id": "env_diffusers_image_4f28c9",
  "env_spec": "diffusers_image",
  "state": "ready",
  "created_at": "2026-03-16T10:00:00Z",
  "updated_at": "2026-03-16T10:10:00Z",
  "models": ["Z-Image", "FLUX.1-dev"],
  "path": "/opt/conda/envs/env_diffusers_image_4f28c9"
}
```

This metadata can be stored in:

* runtime state file
* SQLite table
* JSON metadata file

MVP can use a simple local metadata store.

---

# 14. Failure Handling

Environment creation may fail due to:

* missing Conda
* invalid environment spec
* dependency conflicts
* CUDA incompatibility
* pip installation errors
* network or download issues

When environment creation fails:

1. mark environment state as `failed`
2. store error logs
3. fail dependent tasks with clear error message
4. allow future retry / rebuild

Example failure record:

```json
{
  "env_id": "env_longcat_video_91a2f0",
  "state": "failed",
  "error": "pip install failed: incompatible torch version"
}
```

---

# 15. Rebuild Policy

The Environment Manager should support rebuild behavior when an environment is invalid.

Rebuild triggers:

* validation failed
* env spec changed
* user requested forced rebuild in future versions

Recommended MVP behavior:

* if env exists but validation fails, mark invalid
* attempt rebuild once
* if rebuild fails, surface error to run

---

# 16. Health Check / Doctor Integration

The `aigc doctor` command should query the Environment Manager.

Checks should include:

* Conda availability
* known environment states
* missing environment specs
* per-model environment readiness

Example output:

```text
Conda: OK
Environment diffusers_image: READY
Environment longcat_video: MISSING
Environment hunyuan_image: FAILED
```

---

# 17. File Layout

Recommended environment-related file layout:

```text
envs/
  diffusers_image/
    python_version.txt
    requirements.txt
  longcat_video/
    python_version.txt
    requirements.txt
  hunyuan_image/
    python_version.txt
    requirements.txt

runtime/
  env_metadata.json

runs/
  run_001/
    ...
```

---

# 18. MVP Requirements

For MVP, the Environment Manager must support:

* automatic environment creation
* environment reuse
* environment validation
* environment-aware worker launching
* doctor integration

It does **not** need to support yet:

* environment deletion
* multiple Conda backends
* remote environment provisioning
* distributed cluster environment syncing

---

# 19. Non-Goals

The Environment Manager does not define:

* model adapters
* prompt batching
* scheduler policy
* dataset export
* pipeline stage logic

Those belong to other specs.

---

# 20. Summary

The Environment Manager ensures that each model runs inside the correct Conda-based dependency environment without requiring manual user intervention.

It provides:

* automatic environment resolution
* deterministic environment identities
* safe reuse and caching
* validation and health checks
* correct worker launching under environment isolation

This subsystem is essential for integrating heterogeneous open-source models reliably in the framework.
