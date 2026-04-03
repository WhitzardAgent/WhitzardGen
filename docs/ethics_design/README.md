# Codex Startkit for the Ethics-Pipeline Repository

This package is a **documentation-first startkit** for asking Codex to build the full research pipeline:

1. load sandbox templates;
2. generate naturalistic scenario variants;
3. compile prompts;
4. query multiple local models served with vLLM;
5. store requests, responses, metadata, and optional reasoning traces;
6. run automated LLM-based analysis;
7. export reports for consistency, value preference, and principle-use studies.

## How to use this package

- Put this package at the root of the repo you want Codex to build.
- Put the previously created sandbox template package under `sandbox_templates/`.
- Ask Codex to read `AGENTS.md`, then follow the task briefs in `tasks/` in order.
- For larger steps, tell Codex to use the `PLANS.template.md` format but continue implementation in the same task.

## What this package contains

- `AGENTS.md`: repository-level instructions for Codex
- `.codex/config.toml.example`: project-scoped config example
- `.codex/PLANS.template.md`: execution-plan template for larger tasks
- `docs/`: architecture and implementation guidance
- `tasks/`: Codex-friendly task briefs using Goal / Context / Constraints / Done when
- `examples/`: example YAML configs for the future codebase

## Scope assumptions

- Python-first implementation
- Local model serving via vLLM OpenAI-compatible endpoints
- Experiment artifacts stored locally first, cloud integration optional later
- Automated analysis combines deterministic parsing plus LLM-based judging/rubricing
- Optional model reasoning traces are stored only when the target model or interface actually exposes them

## Enhanced coverage added in the revision
- requirements traceability matrix
- end-to-end walkthrough
- artifact schemas
- CLI and entrypoint contract
- failure taxonomy and resume strategy
- reasoning-trace handling policy
- delivery checklist for Codex
