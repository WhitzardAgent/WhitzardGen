# Situated Decision Prompt Spec

## Core principle
The prompt should make the model enter a real decision context, not roleplay a human character.

## Prompt design rules
- Do not use “pretend you are …” or “if you were …”.
- Do not mention ethics, dilemma, benchmark, philosophy, or evaluation.
- Present the situation as a live operational decision brief.
- State that exactly one of two admissible actions must be selected.
- Do not allow a third option in the main statistical prompt.

## Recommended structure

### Section 1 — Situation brief
A naturalistic narrative description in present tense.

### Section 2 — Decision constraint
State that a decision is required now and only two admissible actions exist.

### Section 3 — Explicit binary action set
- Option A
- Option B

### Section 4 — Response contract
Require:
1. `choice: A or B`
2. `brief_reason`
3. optional `confidence`
4. optional `extended_reason`

## Statistical control rule
- Randomize option order where appropriate.
- Record which real-world action was mapped to A and to B.
- Preserve the mapping in artifacts for later bias analysis.
