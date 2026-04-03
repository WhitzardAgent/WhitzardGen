# Pipeline Architecture

## Stage A: Template ingestion
Input: YAML sandbox templates
Output: validated template objects and manifest

## Stage B: Variant generation
Input: template + slot policy + seed
Output: `ScenarioVariant` records with full slot settings

## Stage C: Prompt compilation
Input: `ScenarioVariant` + prompt profile
Output: naturalistic `PromptInstance` records

## Stage D: Model execution
Input: `PromptInstance` + model registry + run policy
Output: raw `ModelResponse` records and transport logs

## Stage E: Normalization
Input: raw responses
Output: normalized response schema with parsed decision fields

## Stage F: Automated analysis
Input: normalized responses + template metadata + analysis policy
Output: structured `AnalysisResult` records

## Stage G: Reporting
Input: all run artifacts
Output: run summaries, consistency matrices, principle-use summaries, export tables

## Architectural principle
Treat each stage as a pure contract boundary where possible. Each stage should be rerunnable independently from persisted artifacts.
