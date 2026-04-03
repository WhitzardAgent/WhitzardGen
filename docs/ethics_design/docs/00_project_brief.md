# Project Brief

## Objective
Create a research-grade pipeline that turns `sandbox_template` YAML files into large sets of naturalistic scenario prompts, queries multiple locally served LLMs, and produces analysis-ready artifacts and automated evaluations.

## End-to-end flow
`sandbox_template` -> variant specification -> rendered prompt -> model request -> raw model output -> normalized response record -> automated analysis -> reports / exports

## Non-goals
- no product UI in the first phase;
- no requirement for cloud-hosted models in the first phase;
- no assumption that all models expose reasoning traces.

## Core research needs
- large-scale controlled scenario variation;
- reproducible prompt compilation;
- multi-model comparability;
- metadata-complete storage;
- automated analysis of decisions, rationales, consistency, and value profiles.
