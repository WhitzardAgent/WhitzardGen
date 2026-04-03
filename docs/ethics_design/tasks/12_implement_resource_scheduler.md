# Task 12 — Implement resource-aware model scheduler

## Goal
Inspect the host, choose feasible launchers, allocate ports/GPUs, and produce a launch plan.

## Context
Read:
- `MASTER_SPEC.md`
- `docs/20_model_launch_registry_and_scheduler.md`
- `examples/host_inventory.example.yaml`

## Constraints
- reuse healthy existing services when possible;
- fail clearly when no feasible launch plan exists;
- preserve a machine-readable launch plan artifact;
- support `--auto-launch` from CLI.

## Done when
- required model aliases can be resolved into a launch plan;
- one end-to-end startup and health-check flow works;
- tests cover infeasible plans and reuse of existing services.
