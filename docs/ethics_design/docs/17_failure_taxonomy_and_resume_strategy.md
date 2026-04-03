# Failure Taxonomy and Resume Strategy

## Failure classes

### A. Template-stage failures
- invalid YAML
- schema mismatch
- invariant violation after slot sampling

### B. Prompt-stage failures
- benchmark leakage detected
- unsupported prompt profile combination
- rendering exception

### C. Request-stage failures
- model endpoint unavailable
- timeout
- authentication failure
- malformed request payload
- transient transport failure

### D. Response-stage failures
- empty response
- truncated response
- malformed JSON when structured output expected
- parse failure for normalized response

### E. Analysis-stage failures
- judge model timeout
- schema-constrained output invalid
- unsupported reasoning-trace field

## Resume strategy
A run should be resumable at the unit-of-work level.
Recommended unit keys:
- variant generation unit: `(template_id, seed, slot_policy_hash, index)`
- request unit: `(prompt_id, model_alias, sampling_hash)`
- analysis unit: `(response_id, analysis_policy_id, judge_model_alias)`

## Resume rule
On resume:
- detect completed units from persisted artifacts
- requeue only missing or failed units
- preserve original run id unless a new branching run is intended
