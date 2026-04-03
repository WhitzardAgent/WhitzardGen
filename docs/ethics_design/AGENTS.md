# AGENTS.md

## Mission
Build a reproducible research pipeline for moral-conflict scenario generation, local multi-model inference, and automated downstream analysis.

## What the repository is for
This repository is not a product app. It is research infrastructure for generating scenario variants from sandbox templates, querying multiple local models, and analyzing outputs at scale.

## Source of truth
- `sandbox_templates/` is the source of truth for moral-conflict templates.
- `docs/` describes the target architecture and constraints.
- `tasks/` provides implementation slices. Follow them incrementally.

## Working style
For each task:
1. restate the goal briefly in implementation terms;
2. inspect the relevant docs and target files;
3. make the smallest end-to-end useful change;
4. add or update tests;
5. run the relevant checks;
6. review your own diff for edge cases and regressions.

For larger or ambiguous work:
- write or update a short execution plan using `.codex/PLANS.template.md`;
- continue implementation in the same task after planning;
- do not stop after producing only a plan.

## Prompt discipline
When responding to implementation tasks:
- stay concrete and action-oriented;
- do not write long preambles or status essays;
- prefer repository changes, tests, and concise notes;
- avoid speculative abstractions unless they clearly reduce future rework.

## Engineering constraints
- Prefer Python 3.11+.
- Prefer typed code and explicit data contracts.
- Prefer `pydantic` for schemas and runtime validation.
- Prefer `asyncio` for high-concurrency request orchestration.
- Prefer deterministic local artifacts (JSONL, Parquet, DuckDB, YAML, Markdown) over hidden state.
- Do not hardcode model-specific assumptions into generic orchestration paths.
- Design for resumability and partial reruns.

## Pipeline constraints
- Keep generation, transport, storage, and analysis as separate modules.
- Preserve template invariants during scenario generation.
- Never let prompt rendering leak benchmark or test language.
- Treat model responses and analyses as separate artifact layers.
- Store the full slot settings that produced each prompt.
- Support local vLLM endpoints through a model registry rather than one-off scripts.

## Done means
A task is done only when:
- the code path works end-to-end for the implemented slice;
- tests for that slice exist and pass;
- the public interfaces are documented or self-evident from types and examples;
- the implementation writes artifacts needed for reproducibility;
- the diff has been reviewed for obvious failure modes.

## Repository conventions
- Keep modules small and composable.
- Keep prompt templates and analysis rubrics in versioned files.
- Put executable entrypoints in `src/ethics_pipeline/cli/` or `scripts/`.
- Keep sample configs under `examples/` or `configs/`.
- Keep experimental notebooks out of core logic.

## Do-not rules
- Do not collapse all functionality into one monolithic script.
- Do not mix prompt generation with model transport code.
- Do not parse analysis labels directly from free-form text without a schema.
- Do not add external services unless a doc explicitly calls for them.
- Do not assume hidden chain-of-thought access.

## Fixed-interface policy
The repo should converge on the fixed module, CLI, schema, and UI contracts defined in `MASTER_SPEC.md` and the related docs. Do not rename public interfaces casually once they are implemented.
