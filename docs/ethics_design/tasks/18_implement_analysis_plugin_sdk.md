# Task 18 — Implement the analysis plugin SDK and runtime

## Goal
Create a plugin-first analysis layer so new analysis methods can be added without modifying core orchestration.

## Context
Read:
- `MASTER_SPEC.md`
- `docs/28_analysis_plugin_architecture.md`
- `docs/29_plugin_governance_and_human_in_the_loop.md`
- `schemas/analysis_plugin.schema.yaml`
- `examples/analysis_plugin_manifest.example.yaml`

## Constraints
- plugin discovery must be explicit and testable;
- plugin outputs must be versioned and persisted independently;
- plugin provenance must be available to the UI;
- at least one deterministic plugin and one judge-model plugin must be implemented as examples.

## Done when
- plugins can be registered and run from config;
- dependency ordering works;
- plugin outputs show up in analysis artifacts and the UI.
