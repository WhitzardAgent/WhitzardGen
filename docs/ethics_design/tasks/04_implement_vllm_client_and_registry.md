# Task 04 — Implement the model registry and local vLLM client layer

## Goal
Add a model registry plus a transport client for local OpenAI-compatible vLLM endpoints.

## Context
Read:
- `docs/06_local_vllm_model_serving.md`
- `docs/04_data_contracts.md`
- `examples/model_registry.example.yaml`

## Constraints
- model routing must be registry-driven;
- support per-model timeouts and concurrency limits;
- keep transport and prompt generation separate;
- add one smoke-testable local request path with a mocked transport fallback.

## Done when
- the registry loads;
- requests can be built from a prompt instance;
- responses can be captured in raw artifact form;
- tests cover endpoint selection and request normalization.
