# Open Questions and Working Assumptions

This package proceeds with the following assumptions unless you later override them:

1. Python is the implementation language.
2. Local models are reachable through vLLM endpoints using an OpenAI-compatible interface.
3. The analysis layer is allowed to call one or more judge models, including local judge models.
4. The first implementation phase targets local batch experimentation, not a web UI.
5. Results should remain reproducible from stored artifacts rather than notebook state.
6. Reasoning traces are optional and capability-dependent.

If you later choose a different storage backend, orchestration tool, or judge-model strategy, those should change config and adapters before they change higher-level contracts.
