# Model Registry Specification

## 1. Purpose

The **Model Registry** is responsible for managing all models available to the framework.  
It acts as the **central source of truth** for:

- which models exist
- which adapter each model uses
- model capabilities
- runtime requirements (GPU, batch limits, environments)
- versioning and configuration

The registry enables the system to dynamically discover models and allows the scheduler and CLI to operate without hard-coding model implementations.

---

# 2. Responsibilities

The Model Registry must provide the following functionality:

1. **Model discovery**
2. **Adapter binding**
3. **Capability exposure**
4. **Environment configuration**
5. **Version tracking**
6. **Runtime constraints**
7. **Model filtering by modality/task**

The registry must **not execute models**.  
Execution is handled by the **executor + adapters**.

---

# 3. Core Concepts

## 3.1 Model

A **Model** is a registered generative model supported by the framework.

Examples:

- Z-Image
- FLUX.1-dev
- SDXL
- HunyuanImage-3.0
- LongCat-Video
- Wan2.2-T2V

---

## 3.2 Adapter

Each model is associated with exactly **one adapter class**.

Adapters define how to:

- prepare inputs
- run inference
- collect outputs

Example:

```

Z-Image → DiffusersImageAdapter
LongCat-Video → ScriptVideoAdapter
HunyuanImage → TransformersImageAdapter

````

---

## 3.3 Model Capability

Capabilities describe what the model can do.

Examples:

- supports_batch_prompts
- max_batch_size
- output_types
- supports_negative_prompt
- supported_modalities

Capabilities help the **scheduler** decide how to group tasks.

---

# 4. Registry Data Model

## ModelInfo

```python
from pydantic import BaseModel
from typing import List, Dict, Optional

class ModelInfo(BaseModel):

    name: str

    version: str

    adapter: str

    modality: str

    task_type: str

    capabilities: Dict

    runtime: Dict

    weights: Dict
````

---

## Field Description

### name

Unique model identifier.

Example:

```
Z-Image
FLUX.1-dev
LongCat-Video
```

---

### version

Model version string.

Example:

```
1.0
3.0
A14B
```

---

### adapter

Adapter class used by the model.

Example:

```
DiffusersImageAdapter
ScriptVideoAdapter
TransformersImageAdapter
```

---

### modality

Primary output modality.

Allowed values:

```
image
video
audio
text
```

---

### task_type

Defines model generation task.

Examples:

```
t2i (text to image)
t2v (text to video)
i2v (image to video)
t2a (text to audio)
t2t (text to text)
```

---

### capabilities

Capabilities declared by the adapter.

Example:

```json
{
  "supports_batch_prompts": true,
  "max_batch_size": 8,
  "preferred_batch_size": 4,
  "supports_negative_prompt": true
}
```

---

### runtime

Runtime requirements.

Example:

```json
{
  "gpu_required": true,
  "min_vram_gb": 16,
  "recommended_vram_gb": 24,
  "execution_mode": "in_process"
}
```

---

### weights

Model weight location.

Example:

```json
{
  "hf_repo": "Tongyi-MAI/Z-Image",
  "local_path": null
}
```

---

# 5. Registry Storage

The registry can be implemented in two ways.

## Option A: YAML Registry (recommended for MVP)

A static configuration file.

Example:

```yaml
models:

  Z-Image:
    version: "1.0"
    adapter: "ZImageAdapter"
    modality: "image"
    task_type: "t2i"

    capabilities:
      supports_batch_prompts: true
      max_batch_size: 8
      preferred_batch_size: 4

    runtime:
      execution_mode: "in_process"
      gpu_required: true
      min_vram_gb: 16

    weights:
      hf_repo: "Tongyi-MAI/Z-Image"

  LongCat-Video:
    version: "1.0"
    adapter: "LongCatVideoAdapter"
    modality: "video"
    task_type: "t2v"

    capabilities:
      supports_batch_prompts: false

    runtime:
      execution_mode: "external_process"
      gpu_required: true
      min_vram_gb: 24

    weights:
      hf_repo: "meituan-longcat/LongCat-Video"
```

---

## Option B: Python Registry

Registry defined in code.

Example:

```python
MODEL_REGISTRY = {

    "Z-Image": ModelInfo(
        name="Z-Image",
        version="1.0",
        adapter="ZImageAdapter",
        modality="image",
        task_type="t2i",
        capabilities={
            "supports_batch_prompts": True,
            "max_batch_size": 8
        }
    )

}
```

---

# 6. Registry API

The registry should expose a minimal interface.

## load_registry()

Loads model definitions.

```python
registry = load_registry("models.yaml")
```

---

## get_model()

Returns model configuration.

```python
model = registry.get_model("Z-Image")
```

---

## list_models()

Lists all registered models.

```python
registry.list_models()
```

Example output:

```
Z-Image
FLUX.1-dev
SDXL
HunyuanImage-3.0
LongCat-Video
```

---

## get_models_by_modality()

Filter models by modality.

```python
registry.get_models_by_modality("video")
```

Example:

```
LongCat-Video
Wan2.2-T2V
MOVA-720p
```

---

## get_models_by_task()

Filter models by task type.

```python
registry.get_models_by_task("t2i")
```

Example:

```
Z-Image
FLUX.1-dev
SDXL
HunyuanImage
```

---

# 7. Adapter Binding

When a model is loaded, the registry must bind it to its adapter class.

Example:

```python
adapter_class = ADAPTER_REGISTRY[model.adapter]
adapter = adapter_class(model_config=model)
```

---

# 8. CLI Integration

The CLI should query the registry.

Example:

### list models

```
aigc models list
```

Output:

```
MODEL              TYPE
---------------------------
Z-Image            text-to-image
FLUX.1-dev         text-to-image
LongCat-Video      text-to-video
HunyuanImage-3.0   text-to-image
```

---

### inspect model

```
aigc models inspect Z-Image
```

Output:

```
Model: Z-Image
Task: text-to-image
Batch support: yes
Max batch size: 8
Execution mode: in_process
HF repo: Tongyi-MAI/Z-Image
```

---

# 9. Versioning

Multiple versions of a model may exist.

Example:

```
FLUX.1-dev
FLUX.1-schnell
FLUX.2
```

Registry must support versioned keys:

```
model_name:version
```

Example:

```
FLUX.1-dev:1.0
```

---

# 10. Future Extensions

Future registry features may include:

* automatic HuggingFace metadata syncing
* dynamic model loading
* remote registry services
* runtime benchmarking metadata
* automatic capability detection

---

# 11. Summary

The Model Registry provides a **central configuration system** for models in the framework.

It enables:

* clean separation between models and adapters
* dynamic discovery of models
* capability-aware scheduling
* CLI introspection
* scalable integration of new models

The registry is critical for building a **modular and extensible AIGC generation framework**.
