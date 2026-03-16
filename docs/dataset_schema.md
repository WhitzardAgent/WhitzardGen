# Dataset Schema Specification

## 1. Purpose

The **Dataset Schema** defines the canonical structure for all records produced by the framework.

It specifies how the system stores and exports:

- prompts
- run metadata
- model metadata
- task metadata
- generated artifacts
- execution metadata
- optional labels (future)

The schema must support:

- large-scale synthetic dataset generation
- multimodal artifacts
- traceability from prompt to artifact
- reproducibility of generation settings
- compatibility with JSONL and Parquet export

This document defines the **logical schema** of the dataset, not the database schema for internal runtime state.

---

# 2. Design Principles

The dataset schema must be:

1. **Artifact-centered**
   - every record should clearly identify the generated artifact

2. **Prompt-traceable**
   - every artifact must map back to a unique `prompt_id`

3. **Model-aware**
   - every record must specify which model generated it

4. **Modality-agnostic**
   - schema must support image, video, audio, and text

5. **Batch-compatible**
   - records must remain correct regardless of whether inference was single-prompt or batched

6. **Export-friendly**
   - easy to serialize into JSONL and Parquet

7. **Forward-compatible**
   - future labeling and evaluation fields can be added without breaking existing exports

---

# 3. Dataset Granularity

The framework should export the dataset at the **artifact record level**.

That means:

- one generated image = one dataset record
- one generated video = one dataset record
- one generated audio file = one dataset record
- one generated text output = one dataset record

This is the canonical export granularity.

Even if a task executes a batch of prompts, the final export must produce **one record per artifact per prompt**.

---

# 4. Core Record Schema

Each exported dataset record should follow this structure.

```json
{
  "record_id": "rec_00000001",
  "run_id": "run_001",
  "task_id": "task_000123",
  "prompt_id": "p001",
  "prompt": "a futuristic city at night",
  "negative_prompt": "blurry, low quality",
  "language": "en",
  "model_name": "Z-Image",
  "model_version": "1.0",
  "adapter_name": "ZImageAdapter",
  "modality": "image",
  "task_type": "t2i",
  "artifact_type": "image",
  "artifact_path": "runs/run_001/Z-Image/p001.png",
  "artifact_metadata": {
    "width": 1024,
    "height": 1024,
    "format": "png"
  },
  "generation_params": {
    "seed": 12345,
    "guidance_scale": 4.0,
    "num_inference_steps": 50
  },
  "prompt_metadata": {
    "category": "cityscape",
    "difficulty": "medium"
  },
  "execution_metadata": {
    "status": "success",
    "batch_id": "batch_001",
    "batch_index": 0,
    "started_at": "2026-03-16T10:00:00Z",
    "finished_at": "2026-03-16T10:00:12Z",
    "duration_sec": 12.4
  }
}
````

---

# 5. Required Fields

The following fields are required for every exported record.

| Field                       | Description                        |
| --------------------------- | ---------------------------------- |
| `record_id`                 | unique record identifier           |
| `run_id`                    | run identifier                     |
| `task_id`                   | task identifier                    |
| `prompt_id`                 | unique prompt identifier           |
| `prompt`                    | final prompt used for generation   |
| `model_name`                | model used to generate artifact    |
| `modality`                  | image / video / audio / text       |
| `task_type`                 | t2i / t2v / t2a / t2t              |
| `artifact_type`             | image / video / audio / text       |
| `artifact_path`             | path to generated artifact         |
| `execution_metadata.status` | success / failed / partial_success |

---

# 6. Field Groups

To keep the schema organized, fields are grouped conceptually.

## 6.1 Identity Fields

Fields that uniquely identify the record and its origin.

```json
{
  "record_id": "rec_00000001",
  "run_id": "run_001",
  "task_id": "task_000123",
  "prompt_id": "p001"
}
```

### Definitions

* `record_id`: globally unique dataset record ID
* `run_id`: identifies the overall generation run
* `task_id`: identifies the execution task
* `prompt_id`: identifies the source prompt

---

## 6.2 Prompt Fields

These fields preserve prompt traceability.

```json
{
  "prompt": "a futuristic city at night",
  "negative_prompt": "blurry, low quality",
  "language": "en",
  "prompt_metadata": {
    "category": "cityscape"
  }
}
```

### Definitions

* `prompt`: final prompt after preprocessing
* `negative_prompt`: optional negative prompt used in generation
* `language`: language of the final prompt
* `prompt_metadata`: additional metadata from prompt dataset

---

## 6.3 Model Fields

These fields describe the generating model.

```json
{
  "model_name": "Z-Image",
  "model_version": "1.0",
  "adapter_name": "ZImageAdapter",
  "modality": "image",
  "task_type": "t2i"
}
```

### Definitions

* `model_name`: registry name of the model
* `model_version`: version string
* `adapter_name`: adapter used
* `modality`: image / video / audio / text
* `task_type`: generation task type

---

## 6.4 Artifact Fields

These fields describe the generated output.

```json
{
  "artifact_type": "image",
  "artifact_path": "runs/run_001/Z-Image/p001.png",
  "artifact_metadata": {
    "width": 1024,
    "height": 1024,
    "format": "png"
  }
}
```

### Definitions

* `artifact_type`: output type
* `artifact_path`: path to generated file
* `artifact_metadata`: media-specific metadata

---

## 6.5 Generation Parameter Fields

These fields capture the generation configuration used.

```json
{
  "generation_params": {
    "seed": 12345,
    "guidance_scale": 4.0,
    "num_inference_steps": 50
  }
}
```

These fields are critical for reproducibility.

Examples include:

* `seed`
* `guidance_scale`
* `num_inference_steps`
* `height`
* `width`
* `fps`
* `duration`
* `temperature`
* `top_p`

Not all fields apply to every model.

---

## 6.6 Execution Metadata Fields

These fields capture runtime execution details.

```json
{
  "execution_metadata": {
    "status": "success",
    "batch_id": "batch_001",
    "batch_index": 0,
    "started_at": "2026-03-16T10:00:00Z",
    "finished_at": "2026-03-16T10:00:12Z",
    "duration_sec": 12.4
  }
}
```

### Definitions

* `status`: success / failed / partial_success
* `batch_id`: prompt batch identifier if batch inference used
* `batch_index`: index of this prompt inside batch
* `started_at`: task start timestamp
* `finished_at`: task end timestamp
* `duration_sec`: execution duration

---

# 7. Artifact Metadata by Modality

Different modalities require different metadata.

## 7.1 Image Metadata

```json
{
  "artifact_metadata": {
    "width": 1024,
    "height": 1024,
    "format": "png",
    "channels": 3
  }
}
```

Recommended image metadata fields:

* `width`
* `height`
* `format`
* `channels`

---

## 7.2 Video Metadata

```json
{
  "artifact_metadata": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "duration_sec": 8.0,
    "frame_count": 240,
    "format": "mp4"
  }
}
```

Recommended video metadata fields:

* `width`
* `height`
* `fps`
* `duration_sec`
* `frame_count`
* `format`

---

## 7.3 Audio Metadata

```json
{
  "artifact_metadata": {
    "duration_sec": 12.0,
    "sample_rate": 44100,
    "channels": 2,
    "format": "wav"
  }
}
```

Recommended audio metadata fields:

* `duration_sec`
* `sample_rate`
* `channels`
* `format`

---

## 7.4 Text Metadata

```json
{
  "artifact_metadata": {
    "char_count": 320,
    "token_count": 78,
    "format": "txt"
  }
}
```

Recommended text metadata fields:

* `char_count`
* `token_count`
* `format`

---

# 8. Record Status Rules

A dataset record may be generated only for successful artifacts.

Recommended export behavior for MVP:

* export only `success` records
* keep failed task metadata separately in runtime logs/state store

Optional future behavior:

* export failure records in a separate failure report

Example success record:

```json
{
  "record_id": "rec_1",
  "execution_metadata": {
    "status": "success"
  }
}
```

---

# 9. Batch Execution Semantics

Some models support batch prompt inference.

Batch execution must still export **one record per prompt artifact**.

Example:

Batch input:

```json
{
  "batch_id": "batch_001",
  "prompt_ids": ["p001", "p002", "p003"]
}
```

Model output:

* `p001` → `img_001.png`
* `p002` → `img_002.png`
* `p003` → `img_003.png`

Final export:

```json
{"record_id":"r1","prompt_id":"p001","artifact_path":"img_001.png"}
{"record_id":"r2","prompt_id":"p002","artifact_path":"img_002.png"}
{"record_id":"r3","prompt_id":"p003","artifact_path":"img_003.png"}
```

This mapping is mandatory.

---

# 10. Dataset Export Formats

The schema must support at least:

* JSONL
* Parquet

## 10.1 JSONL Export

Each line is a complete dataset record.

Example:

```json
{"record_id":"rec_1","prompt_id":"p001","model_name":"Z-Image","artifact_path":"runs/run_001/Z-Image/p001.png"}
```

JSONL is the primary MVP format.

---

## 10.2 Parquet Export

Parquet is recommended for large-scale datasets.

Nested objects should remain structured if supported, or be flattened if necessary.

Recommended flattening examples:

```text
artifact_metadata.width
artifact_metadata.height
generation_params.seed
execution_metadata.status
```

---

# 11. Derived Dataset Views

The framework may export alternative dataset views.

## 11.1 Minimal View

A lightweight schema for training and indexing.

```json
{
  "prompt_id": "p001",
  "prompt": "a futuristic city",
  "model_name": "Z-Image",
  "artifact_path": "runs/run_001/Z-Image/p001.png"
}
```

---

## 11.2 Full View

The full schema including metadata.

Used for:

* research
* reproducibility
* debugging
* dataset analysis

---

# 12. Optional Future Fields

These fields are not required for MVP, but the schema should remain compatible with them.

## 12.1 Label Fields

```json
{
  "labels": {
    "safety": {
      "nudity": "safe",
      "violence": "none"
    },
    "semantic": {
      "scene": "cityscape"
    },
    "quality": {
      "clarity": "high"
    }
  }
}
```

---

## 12.2 Evaluation Fields

```json
{
  "evaluation": {
    "prompt_adherence": 0.92,
    "quality_score": 0.81
  }
}
```

---

## 12.3 Provenance Fields

```json
{
  "provenance": {
    "prompt_version": "v2",
    "template_version": "t2i_v3",
    "preprocessing_version": "pp_v1"
  }
}
```

---

# 13. Validation Rules

Before export, dataset records should be validated.

Minimum validation checks:

* `record_id` exists
* `prompt_id` exists
* `artifact_path` exists
* `model_name` exists
* `modality` valid
* `task_type` valid
* artifact file exists on disk
* artifact metadata can be extracted

Invalid records should either:

* be excluded
* or fail export depending on configuration

---

# 14. Recommended File Layout

Example export layout:

```text
runs/
  run_001/
    artifacts/
      Z-Image/
        p001.png
        p002.png
      LongCat-Video/
        p001.mp4
    exports/
      dataset.jsonl
      dataset.parquet
```

Recommended convention:

* artifacts and exports should be stored separately
* export files should reference artifact paths

---

# 15. Schema Evolution Policy

The dataset schema should evolve in a backward-compatible way.

Rules:

1. New optional fields may be added
2. Existing required fields should not be removed
3. Field names should remain stable once adopted
4. Export version should be recorded

Example:

```json
{
  "schema_version": "1.0"
}
```

This can be stored globally in dataset manifest files or per-export metadata.

---

# 16. Summary

The Dataset Schema defines the canonical exported structure for generated artifacts.

It ensures:

* prompt traceability
* model traceability
* reproducibility
* batch-safe export behavior
* support for multimodal outputs
* compatibility with large-scale dataset generation

This schema is the contract between the generation framework and all downstream consumers of the synthetic dataset.