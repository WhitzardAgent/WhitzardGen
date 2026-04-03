# Task 06 — Implement deterministic parsing and LLM-based analysis

## Goal
Normalize raw responses and run automated analysis over them.

## Context
Read:
- `docs/08_llm_analysis_pipeline.md`
- `docs/04_data_contracts.md`
- `examples/analysis_policy.example.yaml`

## Constraints
- deterministic parsing should run before judge-model analysis;
- judge outputs should be schema-constrained;
- do not assume hidden chain-of-thought access;
- all analysis outputs must be persisted with policy and judge metadata.

## Done when
- raw responses normalize into a stable schema;
- one analysis policy can be applied end-to-end;
- results are queryable by template, slot, and model.
