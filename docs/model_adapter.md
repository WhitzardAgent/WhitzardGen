# Model Adapter Specification

## 1. Purpose

The **Model Adapter** layer provides a standardized interface between the framework and heterogeneous generative models.

Models in the ecosystem may differ significantly in how they are invoked:

Examples:

- Diffusers pipelines  
- Transformers-based APIs  
- Python demo scripts  
- torchrun-based distributed scripts  

Therefore the adapter system must normalize:

- input prompts
- execution mechanism
- batching behavior
- artifact outputs
- metadata extraction

Adapters must convert model-specific behaviors into a **common execution and artifact interface** used by the framework.

---

# 2. Adapter Responsibilities

A Model Adapter is responsible for:

1. Translating framework prompts into model inputs
2. Preparing execution configuration
3. Executing model inference (if using in-process mode)
4. Collecting generated artifacts
5. Mapping outputs back to prompt IDs
6. Extracting metadata (resolution, fps, seed, etc.)

Adapters must **not implement scheduling, retry, or resource management**. Those belong to the framework.

---

# 3. Execution Modes

Adapters support two execution modes.

## 3.1 External Process Mode

Used when models are invoked via scripts.

Examples:

- `torchrun run_demo_text_to_video.py`
- `python generate.py`
- repository demo scripts

Execution ownership:

```

Framework Executor → runs subprocess
Adapter → prepares command and parses outputs

```

## 3.2 In-Process Mode

Used when models expose Python APIs.

Examples:

- Diffusers pipelines
- Transformers model.generate()
- internal Python SDKs

Execution ownership:

```

Framework Executor → calls adapter.execute()
Adapter → loads model and runs inference

````

---

# 4. Adapter Interface

## BaseAdapter

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel

ExecutionMode = Literal["external_process", "in_process"]

class ExecutionPlan(BaseModel):
    mode: ExecutionMode
    command: Optional[List[str]] = None
    env: Dict[str, str] = {}
    cwd: Optional[str] = None
    timeout_sec: Optional[int] = None
    inputs: Dict[str, Any] = {}

class ArtifactRecord(BaseModel):
    type: Literal["image", "video", "audio", "text", "json"]
    path: str
    metadata: Dict[str, Any]

class ExecutionResult(BaseModel):
    exit_code: int
    logs: str
    outputs: Dict[str, Any] = {}

class BatchItemResult(BaseModel):
    prompt_id: str
    artifacts: List[ArtifactRecord]
    status: Literal["success", "failed"]
    metadata: Dict[str, Any] = {}
    error_message: Optional[str] = None

class ModelResult(BaseModel):
    status: Literal["success", "partial_success", "failed"]
    batch_items: List[BatchItemResult]
    logs: str = ""
    metadata: Dict[str, Any] = {}
````

---

## Adapter Methods

### prepare()

Creates the execution plan.

```python
def prepare(
    self,
    prompts: List[str],
    prompt_ids: List[str],
    params: Dict[str, Any],
    workdir: str
) -> ExecutionPlan
```

Responsibilities:

* validate parameters
* prepare input files if necessary
* construct command or inputs

---

### execute()

Used **only for in-process adapters**.

```python
def execute(
    self,
    plan: ExecutionPlan,
    prompts: List[str],
    params: Dict[str, Any],
    workdir: str
) -> ExecutionResult
```

Responsibilities:

* load model
* run inference
* write artifacts to workdir

External adapters do **not** implement this.

---

### collect()

Parses artifacts and returns normalized results.

```python
def collect(
    self,
    plan: ExecutionPlan,
    exec_result: ExecutionResult,
    prompts: List[str],
    prompt_ids: List[str],
    workdir: str
) -> ModelResult
```

Responsibilities:

* locate generated files
* map outputs to prompts
* extract metadata

---

# 5. Adapter Capabilities

Each adapter must declare capabilities.

```python
class AdapterCapabilities(BaseModel):

    supports_batch_prompts: bool = False

    max_batch_size: int = 1

    preferred_batch_size: int = 1

    supports_negative_prompt: bool = False

    supports_seed: bool = True

    output_types: List[str] = []
```

Example:

```python
class ZImageAdapter(BaseAdapter):

    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=8,
        preferred_batch_size=4,
        supports_negative_prompt=True,
        output_types=["image"]
    )
```

---

# 6. Batch Prompt Handling

Some models support batch inference.

Examples:

* Diffusers pipelines
* Transformer generation APIs

Example:

```
pipe(prompt=[p1, p2, p3])
```

The framework scheduler should group prompts based on adapter capabilities.

Task definition:

```
task = (model, prompt_batch, params)
```

Batch size rules:

```
if adapter.supports_batch_prompts:
    batch_size = min(preferred_batch_size, scheduler_limit)
else:
    batch_size = 1
```

---

# 7. Execution Flow

```
Scheduler
   ↓
Create Task(model, prompt_batch)
   ↓
Adapter.prepare()
   ↓
Executor
   ↓
if mode == external_process:
       run subprocess
else:
       adapter.execute()
   ↓
adapter.collect()
   ↓
store artifacts
```

---

# 8. Artifact Storage

Artifacts must be stored in a structured directory.

Example:

```
runs/
  run_001/
      Z-Image/
          prompt_001.png
          prompt_002.png
      LongCatVideo/
          prompt_001.mp4
```

Artifact record example:

```
{
  "type": "image",
  "path": "runs/run_001/Z-Image/prompt_001.png",
  "metadata": {
    "resolution": "1024x1024",
    "model": "Z-Image"
  }
}
```

---

# 9. Metadata Extraction

Adapters should extract metadata when possible.

Common metadata fields:

Images

* resolution
* seed
* inference_steps
* guidance_scale

Videos

* fps
* duration
* resolution
* frame_count

Example:

```
{
  "resolution": "720p",
  "fps": 30,
  "duration_sec": 8
}
```

---

# 10. Adapter Types

## Diffusers Adapter

Used for:

* SDXL
* FLUX.1
* Z-Image

Execution mode:

```
in_process
```

Inference example:

```
pipe(prompt=[...])
```

---

## Transformers Adapter

Used for:

* HunyuanImage-3.0

Execution mode:

```
in_process
```

Example:

```
model.generate_image(...)
```

---

## Script Adapter

Used for:

* LongCat-Video
* repository demo scripts

Execution mode:

```
external_process
```

Example:

```
torchrun run_demo_text_to_video.py
```

---

# 11. Error Handling

Failure conditions:

* subprocess exit_code ≠ 0
* expected artifact missing
* runtime exception

Task status mapping:

```
if no artifacts produced:
    failed
elif partial batch failure:
    partial_success
else:
    success
```

---

# 12. Extensibility

New models can be added by implementing a new adapter class.

Steps:

1. Implement BaseAdapter
2. Declare capabilities
3. Register adapter in model registry

Example:

```
model_registry.register(
    model_name="Z-Image",
    adapter=ZImageAdapter
)
```

---

# 13. Non-Goals

Adapters should **not implement**:

* scheduling
* retries
* resource allocation
* GPU management
* dataset export

These are handled by other system components.

---

# 14. Summary

The Model Adapter layer standardizes interaction with heterogeneous models by providing:

* unified execution interface
* batch prompt support
* artifact normalization
* metadata extraction
* execution abstraction

This architecture allows seamless integration of:

* Diffusers models
* Transformers models
* script-based repositories
* future API-based services.
