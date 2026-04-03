# CLI Entrypoints and Config Contracts

The future codebase should expose explicit CLI entrypoints so Codex has a concrete implementation target.

## Recommended CLI surface
- `ethics-pipeline templates validate`
- `ethics-pipeline variants generate`
- `ethics-pipeline prompts compile`
- `ethics-pipeline run execute`
- `ethics-pipeline run resume`
- `ethics-pipeline responses normalize`
- `ethics-pipeline analysis run`
- `ethics-pipeline reports build`
- `ethics-pipeline health models`

## Config layering
Recommended runtime config sources:
1. sandbox template package
2. model registry
3. prompt profile
4. analysis policy
5. run config

## Contract rule
Each CLI command should accept explicit input files and write explicit output artifacts rather than rely on hidden state.

## Suggested interface pattern
- every command accepts `--input` / `--output` or `--run-dir`
- every command can run in `--dry-run` mode where sensible
- every command emits a machine-readable manifest or summary
