# Task 17 — Implement situated decision prompt contract

## Goal
Make the prompt compiler produce non-roleplay, situated decision prompts with exactly two admissible actions.

## Context
Read:
- `MASTER_SPEC.md`
- `docs/27_situated_decision_prompt_spec.md`
- `docs/05_prompt_compiler_and_variant_generation.md`

## Constraints
- no benchmark language;
- no “pretend you are” framing;
- two explicit actions only in the main statistical prompt;
- option order randomization must be recorded.

## Done when
- compiled prompts follow the required structure;
- A/B mapping is persisted in prompt metadata;
- tests cover leakage bans and ordering control.
