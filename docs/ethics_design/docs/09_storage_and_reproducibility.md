# Storage and Reproducibility

## Artifact principle
Every stage should write artifacts that allow the next stage to be rerun without recomputing previous steps.

## Recommended local artifact formats
- YAML for configs and templates
- JSONL for request/response streams
- Parquet for tabular normalized records
- DuckDB for local analytics and joins
- Markdown for run summaries

## Run manifest
Each run should store:
- run id
- git commit if available
- template manifest version
- prompt profile version
- model registry snapshot
- analysis policy version
- random seeds
- start/end timestamps

## Reproducibility rule
A published finding should be traceable back to:
- template id
- slot values
- prompt text
- model endpoint and served model name
- sampling parameters
- raw response
- analysis policy and judge model
