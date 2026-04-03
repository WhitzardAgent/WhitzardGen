# Reasoning vs Non-Reasoning Model Handling

## Why this matters
Some subject models expose richer reasoning-like outputs, while others provide only a final answer or short rationale. If the pipeline treats them identically, later analysis will be misleading.

## Required design
Use two lanes.

### Lane A — Comparability lane
Required for all models.
Store:
- forced choice (`A` or `B`)
- short justification
- refusal flag
- confidence signal if available

This lane is the primary source for cross-model statistics.

### Lane B — Introspection lane
Optional and capability-dependent.
Store:
- extended rationale
- reasoning trace if actually available
- trace source metadata

This lane is used for qualitative or auxiliary analysis only.

## Registry additions
Each model entry should expose:
- `reasoning_mode: none | rationale_only | exposed_trace`
- `supports_structured_output`
- `supports_confidence_field`
- `supports_extended_rationale`
- `trace_handling_policy`

## Analysis rule
Never compare raw reasoning-trace richness across models as though it were the same evidence class as forced choice.
