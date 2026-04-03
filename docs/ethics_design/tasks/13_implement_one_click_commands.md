# Task 13 — Implement one-click run and test workflows

## Goal
Implement the final user-facing one-click flows.

## Context
Read:
- `MASTER_SPEC.md`
- `docs/21_one_click_workflows.md`
- `docs/22_cli_master_spec.md`

## Constraints
- `run execute --auto-launch` must really perform the full pipeline;
- `test all --auto-launch` must run a small but real end-to-end test;
- artifacts must be written in all cases.

## Done when
- a user can run one command to start models, execute a run, analyze it, and build reports;
- a user can run one command to smoke-test the stack.
