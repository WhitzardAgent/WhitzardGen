# One-Click Workflows

## Workflow A — Run the pipeline from scratch
Command:
- `ethics-pipeline run execute --config configs/run.yaml --auto-launch`

Expected behavior:
1. validate configs
2. inspect host
3. launch missing model services
4. validate templates
5. generate variants
6. compile prompts
7. execute requests
8. normalize responses
9. run automated analysis
10. build reports
11. print the run directory

## Workflow B — Full system smoke test
Command:
- `ethics-pipeline test all --auto-launch`

Expected behavior:
1. validate configs and schemas
2. launch a minimal local model set
3. run a tiny end-to-end template batch
4. verify artifacts and reports
5. exit nonzero on any failed stage

## Workflow C — Interactive analysis session
Command:
- `ethics-pipeline ui`

Expected behavior:
1. open or start the UI backend
2. let the user browse runs
3. compare models
4. inspect template families and slot settings
5. export filtered datasets and charts
