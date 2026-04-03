# Repository Delivery Checklist

Use this before declaring the repository ready for broad Codex-driven implementation.

## Architecture
- [ ] repo layout matches `docs/02_target_repo_layout.md`
- [ ] module boundaries are clean
- [ ] CLI entrypoints are defined

## Data contracts
- [ ] template loader exists
- [ ] variant schema exists
- [ ] prompt schema exists
- [ ] request/response schemas exist
- [ ] normalized response schema exists
- [ ] analysis result schema exists
- [ ] run manifest schema exists

## Pipeline
- [ ] one template can run end-to-end
- [ ] multi-model local execution works
- [ ] partial resume works
- [ ] analysis outputs are persisted
- [ ] reports can be rebuilt from artifacts only

## Research readiness
- [ ] all slot values are stored
- [ ] benchmark leakage checks exist
- [ ] reasoning-trace policy is enforced
- [ ] principle / preference metrics are versioned
- [ ] run manifests capture config snapshots

## Codex readiness
- [ ] AGENTS.md is concise and accurate
- [ ] task briefs are up to date
- [ ] tests can be run with one documented command
- [ ] Codex can build the next slice without needing hidden context
