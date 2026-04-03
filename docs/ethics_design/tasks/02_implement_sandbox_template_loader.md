# Task 02 — Implement sandbox template loading and validation

## Goal
Load sandbox templates from YAML, validate them, and expose a typed in-memory representation.

## Context
Read:
- `docs/04_data_contracts.md`
- `docs/05_prompt_compiler_and_variant_generation.md`
- the files under `sandbox_templates/`

## Constraints
- preserve all template metadata;
- fail loudly on schema violations;
- support loading the manifest and the per-template files;
- add tests for valid and invalid templates.

## Done when
- templates can be loaded from disk;
- validation failures are actionable;
- one integration test loads the real package manifest plus at least one template.
