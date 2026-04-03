# Reasoning-Trace Policy

## Why this policy exists
Your research may analyze decision outcomes, explicit rationales, and — where available — reasoning traces. Those must not be conflated.

## Distinctions
- `decision`: the final action recommendation
- `justification`: the explicit rationale text in the response
- `reasoning_trace`: only a model-exposed or interface-exposed reasoning field, if actually returned

## Policy
1. Never assume hidden chain-of-thought access.
2. Never backfill a missing reasoning trace by pretending the short justification is the same thing.
3. Store reasoning traces in a separate nullable field.
4. Analysis code must branch explicitly on whether `reasoning_trace_text_if_available` exists.
5. Publication claims about reasoning traces must clearly state whether they are based on:
   - final rationales only;
   - model-exposed reasoning fields;
   - auxiliary judge-model reconstruction.

## Recommended artifact fields
- `decision_text`
- `justification_text`
- `reasoning_trace_text_if_available`
- `reasoning_trace_source` with values like `none`, `response_field`, `tool_side_channel`, `external_judge`
