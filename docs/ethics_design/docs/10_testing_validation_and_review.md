# Testing, Validation, and Review Strategy

## Unit tests
Cover:
- template loading and validation
- invariant checking
- slot sampling and policy enforcement
- prompt rendering
- response normalization
- analysis schema parsing

## Integration tests
Cover:
- one template -> one variant -> one prompt -> one mocked model response -> one analysis result
- one real vLLM smoke request in a dedicated test profile

## Snapshot tests
Useful for:
- prompt rendering
- schema-constrained analysis outputs
- report generation

## Review checklist
For each Codex-generated change:
- are contracts explicit?
- are artifacts persisted?
- are retries and resumption handled?
- are invariants enforced before prompt emission?
- is benchmark leakage prevented?
- are tests meaningful rather than superficial?
