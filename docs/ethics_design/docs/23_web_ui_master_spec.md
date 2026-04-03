# Web UI Master Spec

## Product goal
The UI should reduce the amount of manual notebook work needed to inspect findings. The human should spend more time interpreting results than plumbing files.

## Required pages

### 1. Run Launcher
- choose run config
- choose model set
- dry-run the launch plan
- start execution

### 2. Live Run Monitor
- stage-level progress
- active model services
- request counts
- failure counts
- links to artifacts

### 3. Prompt Browser
- view prompts by template, slot settings, and prompt profile
- inspect invariants and rendered text side by side

### 4. Response Explorer
- filter by model, template, run, refusal, or decision label
- inspect raw response, normalized response, and optional reasoning trace

### 5. Model Comparison
- compare decision distributions across models
- compare refusal rates
- compare principle labels
- compare ambiguity rates

### 6. Template / Slot Sensitivity Explorer
- select one template family
- compare outcomes across parameter values
- view one-way and pairwise slot effects
- view perturbation robustness

### 7. Analysis Workbench
- run saved analysis presets
- inspect principle-use and value-preference summaries
- compare judge outputs

### 8. Export Panel
- export filtered tables
- export charts
- export report bundles

## Backend expectation
The UI backend should read persisted artifacts; it should not depend on hidden notebook state.
