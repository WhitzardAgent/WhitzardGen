# Task 03 — Implement variant generation and prompt compilation

## Goal
Generate `ScenarioVariant` records from sandbox templates and compile them into naturalistic prompts.

## Context
Read:
- `docs/05_prompt_compiler_and_variant_generation.md`
- `docs/04_data_contracts.md`
- `examples/prompt_profile.example.yaml`

## Constraints
- structural slots must obey template invariants;
- prompts must not leak benchmark language;
- every prompt must store the exact slot values used;
- include naturalism checks and prompt snapshot tests.

## Done when
- one template can generate multiple variants;
- variants compile to prompts;
- prompt metadata is persisted in a typed structure;
- tests cover invariant preservation and leakage checks.
