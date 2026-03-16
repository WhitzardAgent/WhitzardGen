# Scheduler Specification

## 1. Purpose

The **Scheduler** is responsible for orchestrating execution of generation tasks within the framework.

It manages:

- task creation
- prompt batching
- concurrency
- retry logic
- run state tracking
- resource allocation

The scheduler sits between the **Pipeline DAG** and **Model Execution Layer**.

It ensures tasks are executed efficiently while respecting system resource constraints.

---

# 2. Scheduler Responsibilities

The scheduler is responsible for:

| Responsibility | Description |
|---|---|
Prompt batching | group prompts for batch-capable models |
Task creation | expand batches into executable tasks |
Task dispatch | assign tasks to workers |
Concurrency control | limit simultaneous tasks |
Retry handling | retry failed tasks |
Run state tracking | track progress of jobs |

The scheduler **does not perform model execution**.  
Model execution is handled by the **executor + adapters**.

---

# 3. Core Concepts

## 3.1 Run

A **Run** represents a single generation job.

A run contains:

- prompt dataset
- selected models
- generation parameters
- all generated tasks
- run metadata

Example:

```json
{
  "run_id": "run_001",
  "prompt_dataset": "prompts/video_prompts.jsonl",
  "models": ["LongCat-Video", "Wan2.2-T2V"]
}
````

---

## 3.2 Task

A **Task** is the smallest unit of execution.

A task represents one model executing on a batch of prompts.

Definition:

```
task = (model, prompt_batch, parameters)
```

Example:

```json
{
  "task_id": "task_001",
  "model": "Z-Image",
  "prompt_ids": ["p001", "p002"],
  "params": {
    "steps": 50
  }
}
```

---

## 3.3 Task States

Tasks progress through states.

Allowed states:

```
pending
running
success
partial_success
failed
skipped
```

State transitions:

```
pending → running → success
pending → running → failed
```

---

# 4. Scheduler Workflow

The scheduler executes the following workflow.

```
Load Prompts
 ↓
Group Prompts Into Batches
 ↓
Expand Tasks Per Model
 ↓
Add Tasks To Queue
 ↓
Dispatch Tasks To Workers
 ↓
Monitor Execution
 ↓
Retry Failed Tasks
 ↓
Mark Run Complete
```

---

# 5. Prompt Batching

Batching improves efficiency for models supporting multi-prompt inference.

Batching rules:

```
batch_size = min(
    adapter.max_batch_size,
    scheduler.batch_limit
)
```

Example:

```
model: Z-Image
max_batch_size: 8
scheduler_limit: 4

final_batch_size = 4
```

If the model does **not support batching**:

```
batch_size = 1
```

---

# 6. Task Expansion

Tasks are generated for each model and batch.

Example:

Prompt dataset:

```
p1
p2
p3
p4
```

Batch size:

```
2
```

Models:

```
Z-Image
FLUX
```

Generated tasks:

```
task_1 → Z-Image → [p1,p2]
task_2 → Z-Image → [p3,p4]

task_3 → FLUX → [p1,p2]
task_4 → FLUX → [p3,p4]
```

---

# 7. Task Queue

Tasks are stored in a **queue** until executed.

The scheduler maintains:

```
pending_tasks
running_tasks
completed_tasks
failed_tasks
```

Tasks are popped from the queue when resources are available.

---

# 8. Worker Model

Workers execute tasks.

Worker responsibilities:

* receive task
* prepare execution environment
* call model adapter
* collect artifacts
* report result

Worker does **not decide scheduling policy**.

---

# 9. Concurrency Control

The scheduler limits simultaneous tasks.

Two levels of concurrency exist.

## 9.1 Global Concurrency

Maximum number of tasks running simultaneously.

Example:

```
max_workers = 4
```

---

## 9.2 Per-Model Concurrency

Some models should not run multiple instances simultaneously.

Example:

```
LongCat-Video → max 1
Z-Image → max 2
```

---

# 10. Resource Awareness

The scheduler should consider GPU availability.

Example configuration:

```
gpu_count = 4
```

Tasks should be distributed across GPUs.

Example:

```
GPU0 → Z-Image
GPU1 → Z-Image
GPU2 → LongCatVideo
GPU3 → WanVideo
```

This may be handled via environment variables:

```
CUDA_VISIBLE_DEVICES
```

---

# 11. Retry Logic

Tasks may fail due to:

* CUDA OOM
* model crash
* timeout
* missing artifact

Retry policy:

```
max_retries = 3
```

Retry process:

```
task fails
↓
increment retry counter
↓
requeue task
```

If retry limit reached:

```
task marked failed
```

---

# 12. Resume Behavior

Runs must support resuming after interruption.

Resume logic:

1. load run state
2. detect completed tasks
3. skip completed tasks
4. resume pending tasks

Example:

```
run interrupted at task 200/1000
```

Resume continues from task 201.

---

# 13. Run Completion

A run completes when:

```
all tasks success
OR
all tasks success/failed
```

Completion report includes:

* total prompts
* total tasks
* success rate
* failed tasks

Example:

```json
{
  "run_id": "run_001",
  "tasks_total": 1200,
  "tasks_success": 1180,
  "tasks_failed": 20
}
```

---

# 14. Logging

The scheduler must produce structured logs.

Example log record:

```json
{
  "timestamp": "2026-03-16T10:00:00",
  "event": "task_started",
  "task_id": "task_042",
  "model": "Z-Image"
}
```

---

# 15. Metrics

Scheduler metrics should include:

* tasks completed
* tasks failed
* average task duration
* throughput (tasks/sec)

These metrics help monitor generation performance.

---

# 16. Non-Goals

The scheduler does not implement:

* pipeline DAG logic
* model adapters
* prompt schema
* CLI commands

These are defined in separate specifications.

---

# 17. Summary

The scheduler is responsible for managing the lifecycle of generation tasks.

It provides:

* efficient batching
* controlled concurrency
* robust retry logic
* resumable runs
* scalable task execution

It forms the central orchestration component of the generation framework.
