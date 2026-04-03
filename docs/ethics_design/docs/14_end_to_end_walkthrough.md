# End-to-End Walkthrough

## Scenario
You have already unpacked the sandbox-template package into `sandbox_templates/`.
You want to run 20 variants each for two template families across three local models.

## Step 1 — Load template inventory
Use the template manifest to select the families you want.
Output: validated `SandboxTemplate` objects.

## Step 2 — Generate variants
Select slot policies and a seed.
Output: `ScenarioVariant` records with complete slot settings.
Artifact: `variants.parquet`

## Step 3 — Compile prompts
Apply a prompt profile that enforces naturalistic rendering and benchmark-leakage bans.
Output: `PromptInstance` records.
Artifact: `prompts.jsonl`

## Step 4 — Build requests
Combine `PromptInstance` + model registry + run config.
Output: `ModelRequest` records.
Artifact: `requests.jsonl`

## Step 5 — Execute local vLLM calls
Send requests to the registry-selected local endpoints.
Output: `ModelResponse` records.
Artifact: `responses.jsonl`

## Step 6 — Normalize outputs
Parse explicit decision labels, refusals, short rationale text, and any available reasoning-trace field.
Output: `NormalizedResponse` records.
Artifact: `normalized_responses.parquet`

## Step 7 — Run automated analysis
Apply deterministic extraction plus an LLM judge policy.
Output: `AnalysisResult` records.
Artifact: `analyses.jsonl`

## Step 8 — Produce reports
Generate:
- per-template decision summaries
- model-by-model comparison tables
- perturbation robustness summaries
- principle and value-preference tables

## Minimum viable end-to-end proof
A valid MVP exists once one selected template can travel through all eight steps and produce a reproducible report directory.
