# Analysis Plugin Architecture

## Goal
Allow collaborating ethicists to add or revise analysis methods without modifying the core pipeline.

## Plugin principle
The core pipeline should orchestrate plugins, not hardcode one analysis worldview.

## Plugin categories

### 1. Deterministic plugins
Examples:
- refusal detection
- explicit choice extraction
- rationale length and structure checks
- template/slot grouping helpers

### 2. Judge-model plugins
Examples:
- principle inference
- value-preference inference
- ambiguity detection
- stakeholder prioritization
- ethical-theory-specific scoring

### 3. Comparative plugins
Examples:
- cross-model comparison
- within-template slot effect analysis
- perturbation robustness analysis
- repeated-run stability analysis

### 4. Domain-expert plugins
These are the most important for your collaboration model.
Examples:
- a Kantian-oriented analysis plugin
- a contractualist-oriented analysis plugin
- a care-ethics lens plugin
- a clinician-ethics rubric plugin
- a law-and-policy compliance plugin

## Required plugin contract
Each plugin should declare:
- `plugin_id`
- `plugin_version`
- `plugin_type`
- `required_inputs`
- `optional_inputs`
- `output_schema`
- `config_schema`
- `dependencies`
- `description`

## Runtime rule
The analysis engine should:
1. discover plugins
2. validate plugin config
3. resolve plugin input dependencies
4. execute plugins in dependency order
5. persist plugin outputs independently
6. surface plugin provenance in the UI and reports

## Design rule
A new analysis method should require:
- adding one plugin package or file;
- registering it in config;
- zero changes to core orchestration unless the method introduces a truly new execution primitive.
