# runtime_spec.md

# Runtime Architecture Specification

## Overview

This document defines the runtime execution architecture for the AIGC generation framework.

The runtime system is responsible for:

- executing model inference workloads
- managing GPU resources
- scheduling prompt tasks
- maximizing hardware utilization
- ensuring deterministic artifact mapping
- supporting large-scale prompt datasets and heavy generative models

The runtime must support:

- large prompt workloads (10k–100k+ prompts)
- diffusion and video models requiring large GPU memory
- multi-GPU machines (8+ GPUs)
- persistent model execution
- scalable multi-replica inference

The guiding principle of the runtime system is:

> **Load heavy models once and reuse them for many inference tasks.**

---

# Runtime Architecture

The runtime system is composed of four logical layers:

```

Runtime
├── Run Manager
├── Persistent Model Workers
├── Multi-Replica Execution
└── GPU Resource Scheduler

```

Execution structure:

```

Run
├── Model A
│    ├── Replica 0
│    ├── Replica 1
│    └── Replica 2
│
└── Model B
├── Replica 0
└── Replica 1

```

Each **replica** is a persistent worker process.

Each replica:

- loads the model once
- processes many tasks
- exits when the run finishes

---

# Design Goals

## 1. Eliminate repeated model loading

Large models must **not reload for every task**.

Instead:

```

worker start
→ load model
→ process many tasks
→ shutdown

```

This dramatically improves throughput.

---

## 2. Utilize multiple GPUs efficiently

Machines may contain:

```

8 GPUs
16 GPUs

```

The runtime must:

- launch multiple model replicas
- assign GPU subsets to each replica
- distribute tasks across replicas

---

## 3. Support extremely large prompt lists

Prompt sets may contain:

```

10,000 prompts
50,000 prompts
100,000 prompts

```

The runtime must:

- split workloads
- distribute tasks
- maintain deterministic artifact mapping

---

## 4. Maintain reproducibility

Each artifact must map deterministically to:

```

prompt_id
model
replica_id
task_id
artifact_path

```

This guarantees reproducibility.

---

# Core Concepts

## Run

A **run** corresponds to a CLI invocation.

Example:

```

aigc run --models Z-Image --prompts prompts.txt

```

A run contains:

- prompt dataset
- selected models
- runtime scheduling plan
- artifacts and metadata

Run directory:

```

runs/<run_id>/

```

Example structure:

```

runs/run_001/
artifacts/
exports/
run_manifest.json
failures.json

```

---

## Prompt

A prompt represents a generation request.

Example:

```

prompt_id: p000123
text: "a dragon flying over a cyberpunk city"

```

Supported input formats:

```

.txt
.csv
.jsonl

```

Prompts are normalized during loading.

---

## Task

A task is a unit of model inference.

A task may contain:

```

1 prompt
or
a batch of prompts

```

Example:

```

task_001
model: Z-Image
prompts: [p1, p2, p3, p4]

```

Tasks are produced after prompt batching.

---

## Replica

A **replica** is a persistent model worker.

Example:

```

Z-Image Replica 0
Z-Image Replica 1
Z-Image Replica 2

```

Each replica:

- loads the model once
- processes multiple tasks
- exits when the run finishes

---

# Persistent Model Workers

## Motivation

Naive execution loads the model repeatedly:

```

task
→ spawn worker
→ load model
→ run inference
→ exit

```

For large models this is extremely inefficient.

Persistent workers solve this problem.

---

## Worker Lifecycle

Worker lifecycle:

```

start worker
load registry
instantiate adapter
load model

loop:
receive task
run inference
write artifacts

shutdown

```

---

## Worker Logging

Workers must produce clear logs.

Startup:

```

[worker][Z-Image] starting persistent worker
[worker][Z-Image] loading model...
[worker][Z-Image] model loaded in 18.42s
[worker][Z-Image] ready

```

Task execution:

```

[worker][Z-Image] running task task_004 batch_size=4
[worker][Z-Image] finished task task_004 artifacts=4

```

Shutdown:

```

[worker][Z-Image] shutting down

```

These logs are critical for debugging cluster runs.

---

# Worker Strategies

Two worker strategies exist.

## Per-Task Worker

Legacy execution model.

```

task
→ spawn worker
→ execute
→ exit

```

Used for:

- script-driven models
- unstable external pipelines

---

## Persistent Worker

Preferred execution model.

```

worker start
load model
run multiple tasks
worker exit

```

Used for:

- Diffusers models
- large in-process pipelines
- heavy GPU models

---

# GPU Resource Discovery

GPU resources are discovered from:

```

CUDA_VISIBLE_DEVICES

```

or CLI override:

```

--gpus 0,1,2,3

```

Example detected GPUs:

```

available_gpus = [0,1,2,3,4,5,6,7]

```

Total GPU count:

```

gpu_count = 8

````

---

# Model Runtime Requirements

Each model defines runtime requirements in the registry.

Example:

```yaml
runtime:
  worker_strategy: persistent
  gpus_per_replica: 2
  preferred_batch_size: 1
  supports_multi_replica: true
````

Parameter definitions:

| field                  | description                      |
| ---------------------- | -------------------------------- |
| worker_strategy        | persistent or per_task           |
| gpus_per_replica       | GPUs required per model instance |
| preferred_batch_size   | recommended batch size           |
| supports_multi_replica | whether replicas are allowed     |

---

# Replica Planning

Replica count is calculated as:

```
replica_count = floor(total_gpus / gpus_per_replica)
```

Example:

```
8 GPUs total
2 GPUs per replica
→ 4 replicas
```

---

# GPU Assignment

Each replica receives a fixed GPU subset.

Example:

```
Replica 0 → GPUs [0,1]
Replica 1 → GPUs [2,3]
Replica 2 → GPUs [4,5]
Replica 3 → GPUs [6,7]
```

Replica processes launch with:

```
CUDA_VISIBLE_DEVICES=<assigned GPUs>
```

---

# Workload Sharding

Large prompt datasets must be split across replicas.

Example:

```
20,000 prompts
4 replicas
```

Sharding:

```
Replica 0 → prompts 0-4999
Replica 1 → prompts 5000-9999
Replica 2 → prompts 10000-14999
Replica 3 → prompts 15000-19999
```

Alternative strategy:

```
round-robin distribution
```

---

# Task Distribution

Execution pipeline:

```
load prompts
validate prompts
expand tasks
group tasks by model

for each model:

  determine runtime requirements
  detect available GPUs
  compute replica count
  assign GPU sets

  start persistent workers

  distribute tasks across replicas

collect artifacts
export dataset
write run manifest
```

---

# Replica Worker Startup

Each replica receives:

```
model
replica_id
gpu_assignment
run_id
execution_mode
```

Example:

```
Replica 2
model = CogVideoX
GPUs = [4,5]
```

Startup log:

```
[worker][CogVideoX][replica=2] GPUs=[4,5]
loading model...
```

---

# Artifact Mapping

Artifacts must maintain deterministic mapping.

Each artifact record includes:

```
prompt_id
model
replica_id
task_id
artifact_path
metadata
```

Example:

```
{
  "prompt_id": "p001",
  "model": "Z-Image",
  "replica_id": 2,
  "artifact": "runs/run_001/artifacts/zimage_001.png"
}
```

---

# Failure Handling

Failures must be recorded.

Failure file:

```
runs/<run_id>/failures.json
```

Failure record format:

```
task_id
replica_id
error_message
```

Workers must continue processing remaining tasks.

---

# Execution Modes

Two execution modes exist.

## Mock Mode

Local development mode.

Artifacts are simulated.

## Real Mode

Real inference execution.

Persistent workers execute full model pipelines.

---

# Run Output Structure

Run directory structure:

```
runs/<run_id>/
  artifacts/
  exports/
  run_manifest.json
  failures.json
```

Dataset export remains unchanged.

---

# Future Extensions

The runtime architecture is designed for future expansion.

Possible extensions include:

* distributed multi-node scheduling
* GPU memory-aware scheduling
* priority queues
* retry / resume support
* elastic replica scaling
* worker health monitoring
* dynamic load balancing

---

# Summary

The runtime system introduces three key improvements:

1. **Persistent model workers**
   Models load once and process many tasks.

2. **Multi-replica execution**
   Multiple model instances run in parallel.

3. **GPU-aware scheduling**
   Workloads are automatically distributed across GPUs.

This architecture enables efficient execution of large workloads on multi-GPU systems while maintaining deterministic artifact mapping and reproducibility.
