# MASTER SPEC

This file is the **fixed build target** for Codex. It is more prescriptive than the other docs.

## 1. Final objective
Implement a one-click local research system that can:
1. load sandbox templates;
2. generate controlled scenario variants from a template;
3. render naturalistic prompts;
4. auto-launch required local models through registered vLLM startup scripts;
5. route prompt batches across multiple local models;
6. persist all artifacts needed for reproducibility;
7. run automated analysis on decisions, rationales, and reasoning traces if available;
8. expose both CLI and Web UI for experiment execution and analysis.

## 2. Mandatory public commands
The final system MUST expose these CLI commands:
- `ethics-pipeline doctor`
- `ethics-pipeline models status`
- `ethics-pipeline models up`
- `ethics-pipeline models down`
- `ethics-pipeline templates validate`
- `ethics-pipeline variants generate`
- `ethics-pipeline prompts compile`
- `ethics-pipeline run execute`
- `ethics-pipeline run resume`
- `ethics-pipeline responses normalize`
- `ethics-pipeline analysis run`
- `ethics-pipeline reports build`
- `ethics-pipeline ui`
- `ethics-pipeline test smoke`
- `ethics-pipeline test all`

## 3. Mandatory one-click workflows
The final system MUST support:
- `ethics-pipeline doctor` to inspect host resources and configuration
- `ethics-pipeline models up --plan <run_config>` to auto-launch required local model services
- `ethics-pipeline run execute --config <run_config> --auto-launch` for one-click execution
- `ethics-pipeline test all --auto-launch` for one-click test and smoke validation
- `ethics-pipeline ui` to open the analysis workbench

## 4. Mandatory analysis goals
The system MUST support at least these first-class analyses:
- cross-model comparison
- within-template parameter sensitivity
- perturbation robustness
- principle-use inference
- value-preference inference
- refusal and ambiguity analysis
- rationale-shape analysis
- reasoning-trace analysis when available

## 5. Mandatory architecture boundaries
Keep these subsystems separate:
- template ingestion
- variant generation
- prompt compilation
- model launch and health
- request execution
- artifact persistence
- normalization
- analysis
- reporting
- web UI

## 6. Mandatory artifacts
Every run MUST write:
- run manifest
- variant records
- prompt instances
- model requests
- raw responses
- normalized responses
- analysis results
- report outputs

## 7. Mandatory UI surfaces
The web UI MUST include:
- Run Launcher
- Live Run Monitor
- Prompt Browser
- Response Explorer
- Model Comparison view
- Template / Slot Sensitivity view
- Analysis Workbench
- Report Export panel

## 8. Mandatory response-handling policy
The system MUST implement two-lane response handling:
- a **comparability lane** for all models, centered on forced choice, short justification, and refusal handling;
- an **introspection lane** for extended rationale and reasoning traces when actually available.

## 9. Mandatory prompt framing policy
The system MUST use **situated decision framing**, not generic roleplay. Prompts should present a live decision context with exactly two admissible actions and no benchmark language.

## 10. Mandatory plugin architecture for analysis
The analysis subsystem MUST support plugin-style extensibility so new methods from collaborating ethicists can be added without rewriting the core pipeline.
