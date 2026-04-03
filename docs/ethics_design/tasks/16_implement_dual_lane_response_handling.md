# Task 16 — Implement dual-lane response handling

## Goal
Implement comparability-lane and introspection-lane storage and analysis boundaries.

## Context
Read:
- `MASTER_SPEC.md`
- `docs/26_reasoning_vs_nonreasoning_handling.md`
- `docs/18_reasoning_trace_policy.md`

## Constraints
- all models must support the comparability lane;
- introspection lane must be optional and capability-driven;
- do not conflate short justification with reasoning trace.

## Done when
- the response normalization layer stores both lanes correctly;
- model capability flags control which lane fields are expected;
- tests cover reasoning and non-reasoning model cases.
