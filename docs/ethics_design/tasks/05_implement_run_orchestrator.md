# Task 05 — Implement the run orchestrator and artifact persistence

## Goal
Execute many prompt instances across many models with resumability and artifact persistence.

## Context
Read:
- `docs/07_request_orchestrator.md`
- `docs/09_storage_and_reproducibility.md`
- `examples/run_config.example.yaml`

## Constraints
- support partial reruns;
- do not reissue already completed requests unnecessarily;
- persist raw requests, raw responses, and run manifests;
- use async execution with per-model controls.

## Done when
- one run config can execute a multi-model batch;
- artifacts are written under a run directory;
- failed items can be resumed without rerunning successes.
