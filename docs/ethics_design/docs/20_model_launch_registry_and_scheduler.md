# Model Launch Registry and Resource-Aware Scheduler

## Goal
Allow the user to register per-model vLLM startup scripts so the engine can launch models automatically based on host resources instead of requiring manual startup.

## Core idea
Separate:
- **model registry**: logical model identities used by experiments
- **launch registry**: how each model can actually be started on a machine
- **host inventory**: current machine resources
- **scheduler**: chooses a feasible launch plan

## Required files
- `configs/model_registry.yaml`
- `configs/model_launcher_registry.yaml`
- `configs/host_inventory.yaml` or a generated host inventory cache

## Launch registry contract
Each launcher entry should specify:
- launcher id
- model alias
- startup script path
- env file or environment variables
- working directory
- supported hardware backend
- minimum GPU count
- minimum total VRAM
- optional tensor-parallel choices
- optional pipeline-parallel choices
- default port template
- health check endpoint
- startup timeout
- shutdown mode
- whether the launcher can co-host with others on the same GPU set

## Scheduler policy
Given a run config, the scheduler should:
1. determine which model aliases are required
2. inspect current healthy endpoints
3. inspect host resources
4. pick launchers compatible with the host
5. allocate ports and GPU sets
6. start only missing services
7. wait for health checks
8. hand resolved endpoints to the request layer

## Resource-aware decisions
The scheduler must consider:
- GPU count
- per-GPU VRAM
- backend compatibility
- launcher exclusivity
- tensor/pipeline parallel options
- memory utilization targets
- current running services

## Required behavior
- reuse already healthy matching services when possible
- fail clearly when no feasible launch plan exists
- emit a machine-readable launch plan artifact
- optionally tear down services after a run
