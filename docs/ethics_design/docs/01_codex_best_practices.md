# Codex Best Practices to Apply in This Repository

This startkit is designed around current official Codex guidance.

## 1. Prompt each task with four parts
For each implementation task, include:
- Goal
- Context
- Constraints
- Done when

This repository’s `tasks/` files already use that structure.

## 2. Plan first for difficult work, but do not stop at planning
For larger changes, Codex should plan first. Use `.codex/PLANS.template.md` for multi-step work. However, planning should not become a substitute for implementation.

## 3. Put durable guidance in `AGENTS.md`
`AGENTS.md` should carry repo layout, commands, conventions, constraints, and what done means. Keep it short and practical; move details into docs.

## 4. Use project-scoped configuration
Use `.codex/config.toml` for project defaults. Use CLI overrides only for one-off situations.

## 5. Ask Codex to test, verify, and review
Every substantial change should include tests, relevant checks, and a self-review pass.

## 6. Use MCP only when it removes a real recurring loop
Do not wire in external tools prematurely. Add MCP only when live context really lives outside the repo and changes frequently.

## 7. Keep skills or task briefs focused
Each task brief should describe one implementation slice with explicit inputs and outputs.

## Repository-specific synthesis
For this project, the strongest Codex pattern is:
- one task brief per pipeline stage;
- one execution plan for any multi-file step;
- tests and review required before moving to the next stage;
- short, concrete outputs rather than verbose status preambles.
