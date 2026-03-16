# Prompt Specification

## 1. Purpose

The **Prompt Specification** defines the structure, lifecycle, and management of prompts used by the AIGC generation framework.

Prompts are the **primary input unit** for content generation tasks across multiple models and modalities. This specification ensures that prompts are:

- uniquely identifiable
- reproducible
- compatible across models
- batchable for efficient inference
- traceable to generated artifacts

The specification supports multimodal generation including:

- text-to-image (T2I)
- text-to-video (T2V)
- text-to-audio (T2A)
- text-to-text (T2T)

---

# 2. Prompt Record Schema

Each prompt must be represented as a structured record.

```json
{
  "prompt_id": "prompt_000001",
  "prompt": "A futuristic cyberpunk city at night with neon lights",
  "negative_prompt": "blurry, low quality",
  "language": "en",
  "parameters": {
    "style": "cinematic",
    "resolution": "1024x1024"
  },
  "metadata": {
    "category": "cityscape",
    "difficulty": "medium"
  },
  "version": "v1"
}
````

---

## 2.1 Field Definitions

### prompt_id

A globally unique identifier for the prompt.

Required for:

* dataset traceability
* retry logic
* artifact mapping
* deduplication

Example:

```
prompt_000123
```

---

### prompt

The main textual instruction given to the model.

Example:

```
"A futuristic cyberpunk city at night"
```

---

### negative_prompt

Optional text specifying undesired attributes.

Common for diffusion models.

Example:

```
"blurry, distorted, low quality"
```

---

### language

Language of the prompt.

Allowed values:

```
en
zh
```

Future expansion may include additional languages.

---

### parameters

Optional structured parameters that guide generation.

Examples:

| Parameter  | Example   |
| ---------- | --------- |
| style      | cinematic |
| resolution | 1024x1024 |
| fps        | 30        |
| duration   | 8s        |

---

### metadata

Optional metadata used for dataset organization and analytics.

Examples:

```
category
style
scene
difficulty
dataset_split
```

---

### version

Prompt template version used to generate this prompt.

Example:

```
t2i_v2
```

---

# 3. Prompt Dataset Format

Prompt datasets are stored as **JSONL files**.

Example:

```
prompts.jsonl
```

Example content:

```json
{"prompt_id":"p1","prompt":"a futuristic city","language":"en"}
{"prompt_id":"p2","prompt":"一只可爱的猫","language":"zh"}
```

Each line represents a single prompt record.

---

## 3.1 Dataset Requirements

Datasets must satisfy:

* unique `prompt_id`
* valid JSON record
* supported language
* non-empty prompt text

Validation should occur during dataset loading.

---

# 4. Batch Prompt Specification

For models supporting batch inference, prompts may be grouped into batches.

Batch structure:

```json
{
  "batch_id": "batch_001",
  "prompt_ids": ["p1","p2","p3"],
  "prompts": [
    "a futuristic city",
    "a cyberpunk street",
    "a flying car"
  ]
}
```

Batching improves GPU utilization for models such as:

* Diffusers pipelines
* transformer generation models

Batch size is determined by:

```
adapter.max_batch_size
scheduler_limits
```

---

# 5. Language Handling

The system supports multilingual prompts.

Current supported languages:

```
English (en)
Chinese (zh)
```

---

## 5.1 Language Compatibility

Each model declares supported languages.

Example:

```
Z-Image → en, zh
SDXL → en
```

Processing logic:

```
if prompt_language supported by model:
    use prompt as-is
else:
    translate to supported language
```

---

## 5.2 Automatic Translation

If required, prompts are automatically translated.

Example:

Input:

```
生成一张未来城市的图片
```

Translated:

```
Generate an image of a futuristic city
```

Translation should occur during **preprocessing stage**.

---

# 6. Prompt Templates

Prompt templates allow structured prompt generation.

---

## 6.1 Text-to-Image Template

```json
{
  "template": "An image of {scene} in {style} style with {lighting}"
}
```

Example output:

```
An image of a futuristic city in cinematic style with neon lighting
```

---

## 6.2 Text-to-Video Template

```json
{
  "template": "A {duration} cinematic video of {scene}"
}
```

Example:

```
An 8 second cinematic video of a flying car in a cyberpunk city
```

---

## 6.3 Text-to-Audio Template

```json
{
  "template": "Ambient {sound_type} audio with {mood} atmosphere"
}
```

Example:

```
Ambient forest audio with calm atmosphere
```

---

# 7. Prompt Augmentation

Prompts may be augmented automatically to increase dataset diversity.

Augmentation strategies include:

* style modifiers
* camera descriptors
* lighting conditions
* environment descriptors

Example:

Base prompt:

```
a cat
```

Augmented prompt:

```
a photorealistic portrait of a cat with dramatic studio lighting
```

---

# 8. Prompt Constraints

Prompts must satisfy model-specific limits.

Examples:

| Model        | Prompt Token Limit |
| ------------ | ------------------ |
| SDXL         | ~77 tokens         |
| FLUX         | ~256 tokens        |
| Transformers | model dependent    |

If prompt exceeds limits:

```
truncate
summarize
reject
```

---

# 9. Prompt Normalization

Before execution prompts must be normalized.

Normalization rules:

* remove leading/trailing whitespace
* collapse repeated spaces
* normalize unicode
* strip unsupported characters

Example:

```
"  a   futuristic city  "
```

Normalized:

```
"a futuristic city"
```

---

# 10. Prompt → Artifact Mapping

Each generated artifact must be linked to its originating prompt.

Example record:

```json
{
  "prompt_id": "p001",
  "model": "Z-Image",
  "artifact": "runs/run1/zimage/p001.png"
}
```

This mapping enables:

* retry of failed prompts
* dataset export
* evaluation metrics

---

# 11. Prompt Execution Metadata

When prompts are executed, additional metadata should be recorded.

Example:

```json
{
  "prompt_id": "p001",
  "model": "Z-Image",
  "seed": 12345,
  "guidance_scale": 7.5,
  "steps": 50
}
```

---

# 12. Failure Handling

Prompt execution may fail due to:

* runtime errors
* model crashes
* invalid prompts

Failure record:

```json
{
  "prompt_id": "p001",
  "status": "failed",
  "error": "CUDA out of memory"
}
```

Failed prompts may be retried.

---

# 13. Dataset Export

Final generated dataset entries should include:

```json
{
  "prompt_id": "p001",
  "prompt": "a futuristic city",
  "model": "Z-Image",
  "artifact": "image_001.png",
  "metadata": {
    "resolution": "1024x1024"
  }
}
```

Supported dataset formats:

```
JSONL
Parquet
CSV
```

---

# 14. Future Extensions

Future prompt system capabilities may include:

* automated prompt generation using LLMs
* prompt scoring
* semantic diversity enforcement
* prompt clustering
* multilingual prompt synthesis

---

# 15. Summary

The Prompt Specification defines the standardized structure and lifecycle of prompts within the generation framework.

It ensures:

* prompt reproducibility
* dataset traceability
* compatibility with heterogeneous models
* efficient batch execution
* consistent dataset generation

This specification enables scalable generation of **large multimodal synthetic datasets** across many generative models.
