# Automated LLM Analysis Pipeline

## Analysis layers

### Layer 0: deterministic extraction
Parse obvious fields such as:
- refusal
- explicit decision
- whether options were compared
- response length and latency metadata

### Layer 1: schema-constrained LLM judging
Use an analysis policy and a structured-output schema to infer:
- normative principles invoked
- value preferences
- stakeholder prioritization
- confidence or ambivalence

### Layer 2: consistency and robustness analysis
Compare outputs across:
- slot changes within one template
- narrative perturbations
- models
- seeds or repeated runs

### Layer 3: higher-order reporting
Aggregate into:
- principle frequency by template family
- consistency matrices
- value preference profiles
- prompt-framing sensitivity summaries

## Important guardrail
Keep analysis models separate from target models where possible. At minimum, record when the same model family is used as both subject and judge.

## Reasoning traces
Do not assume hidden chain-of-thought access. Only analyze reasoning traces when the serving interface actually returns them or when the model is explicitly asked for an inspectable rationale.
