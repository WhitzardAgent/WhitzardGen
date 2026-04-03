# Target Repository Layout

```text
repo/
  AGENTS.md
  README.md
  sandbox_templates/
    manifest.yaml
    templates/*.yaml
  configs/
    model_registry.yaml
    prompt_profiles.yaml
    analysis_policies.yaml
    run_presets.yaml
  src/
    ethics_pipeline/
      templates/
      generation/
      prompts/
      models/
      orchestration/
      storage/
      analysis/
      reporting/
      cli/
  tests/
    unit/
    integration/
    smoke/
  runs/
    <run_id>/
      manifest.json
      variants.parquet
      prompts.jsonl
      requests.jsonl
      responses.jsonl
      normalized_responses.parquet
      analyses.jsonl
      reports/
  scripts/
  docs/
```

## Module responsibilities
- `templates/`: load and validate sandbox templates
- `generation/`: slot sampling and variant specification
- `prompts/`: naturalistic prompt rendering and prompt-profile management
- `models/`: registry, clients, transport adapters, retries
- `orchestration/`: batching, concurrency, resumability
- `storage/`: artifact schemas and persistence
- `analysis/`: parsers, LLM judges, consistency checks
- `reporting/`: summary tables, exports, dashboards-ready files
