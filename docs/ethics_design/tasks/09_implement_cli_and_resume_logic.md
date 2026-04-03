# Task 09 — Implement CLI entrypoints and resume logic

## Goal
Create the CLI surface and implement resume-safe orchestration keys.

## Context
Read:
- `docs/16_cli_entrypoints_and_config_contracts.md`
- `docs/17_failure_taxonomy_and_resume_strategy.md`

## Constraints
- every CLI command must take explicit inputs and outputs;
- resume must skip successful completed units;
- failure categories should be observable in artifacts.

## Done when
- the main CLI commands exist;
- one interrupted run can be resumed correctly;
- tests cover deduplication and partial rerun behavior.
