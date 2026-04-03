# Task 11 — Implement model launcher registry

## Goal
Add a launcher-registry subsystem that lets the user register per-model vLLM startup scripts.

## Context
Read:
- `MASTER_SPEC.md`
- `docs/20_model_launch_registry_and_scheduler.md`
- `examples/model_launcher_registry.example.yaml`

## Constraints
- keep model registry and launcher registry separate;
- launcher entries must be validated;
- health check behavior must be explicit;
- do not hardcode scripts in Python modules.

## Done when
- launcher registry loads and validates;
- one launcher entry can be turned into a concrete start command;
- tests cover invalid resource declarations and missing scripts.
