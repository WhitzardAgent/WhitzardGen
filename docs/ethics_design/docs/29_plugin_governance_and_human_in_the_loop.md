# Plugin Governance and Human-in-the-Loop Analysis

## Why this exists
Because this project is collaborative with ethicists, not all analyses should be frozen in advance.

## Human-in-the-loop principle
The pipeline should support three modes:
- fully automated analysis
- automated analysis with expert review
- expert-authored plugin analysis

## Recommended workflow
1. an ethicist defines a new analysis lens;
2. the lens is encoded as a plugin with a versioned config and output schema;
3. the plugin is registered in the analysis policy;
4. the plugin runs on stored artifacts;
5. the output is visible in the workbench alongside other plugin outputs.

## Review requirement
Every plugin should support:
- versioned release notes
- sample fixtures
- test cases
- known failure modes

## UI implication
The workbench should show:
- which plugins ran
- what version ran
- what inputs they consumed
- where outputs disagreed across plugins
