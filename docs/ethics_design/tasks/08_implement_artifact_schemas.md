# Task 08 — Implement artifact schemas and validation boundaries

## Goal
Turn the blueprint schemas in `schemas/` into real typed validation models in code.

## Context
Read:
- `schemas/*.yaml`
- `docs/04_data_contracts.md`
- `docs/09_storage_and_reproducibility.md`

## Constraints
- each artifact layer should have its own schema;
- schemas should be reusable by CLI commands and tests;
- keep nullable reasoning-trace handling explicit.

## Done when
- code-level schemas exist for all major artifact types;
- validation tests exist for at least one valid and one invalid record per schema.
