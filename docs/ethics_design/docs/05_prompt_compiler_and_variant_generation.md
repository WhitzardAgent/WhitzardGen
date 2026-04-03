# Prompt Compiler and Variant Generation

## Design goals
- preserve the deep conflict from the sandbox template;
- vary only declared slots;
- produce naturalistic prompts that do not reveal benchmark intent;
- log every slot value used.

## Recommended generation flow
1. select template
2. select structural slots under policy constraints
3. select narrative slots
4. select perturbation slots
5. run invariant checks
6. render prompt
7. run naturalism checks

## Naturalism checks
At minimum, verify that the rendered prompt:
- does not mention ethics, test, benchmark, or philosophy;
- contains a real decision frame;
- does not leak template ids or slot names;
- preserves the template’s required asymmetries.

## Prompt profiles
Maintain prompt profiles separately from templates. A profile should control:
- point of view
- length
- formality
- emotional intensity band
- whether options are explicit or implicit
- whether legal/professional context is foregrounded

## Prompt compiler output
The compiler should emit both:
- the final prompt text;
- a sidecar metadata object for reproducibility.
