# Task 10 — Implement analysis metrics and reasoning-trace policy

## Goal
Turn the metrics and reasoning-trace policy into code-level analysis functions and guards.

## Context
Read:
- `docs/15_analysis_dimensions_and_metrics.md`
- `docs/18_reasoning_trace_policy.md`
- `schemas/normalized_response.schema.yaml`
- `schemas/analysis_result.schema.yaml`

## Constraints
- do not treat justification text as hidden chain-of-thought;
- metrics must be explicit and versionable;
- analysis should branch cleanly on whether reasoning traces are actually present.

## Done when
- metric functions or judge tasks are implemented;
- reasoning-trace guards prevent invalid conflation;
- tests cover both trace-present and trace-absent cases.
