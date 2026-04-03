# Sandbox Template Package

This package contains English-language sandbox templates derived from 32 classic moral-conflict scenarios and consolidated into 19 reusable templates.

## Design goals
- naturalistic prompt generation without revealing that the prompt is part of a test;
- analysis-ready metadata for decisions, justifications, optional reasoning traces, and value-profile studies;
- explicit separation between deep moral structure and surface narration;
- slot definitions that contribute to findings rather than merely adding cosmetic variety.

## Files
- `templates/*.yaml`: one YAML sandbox template per conflict family
- `manifest.yaml` / `manifest.csv`: source-case to template mapping
- `slot_library.yaml`: reusable slots and why they matter analytically
- `analysis_codebook.yaml`: suggested coding dimensions
- `schema.yaml`: top-level template schema
- `theory_grounding.md`: ethics, narratology, and structuralism rationale

## Recommended workflow
1. Load a sandbox template.
2. Sample or set structural, narrative, and perturbation slots.
3. Realize a naturalistic prompt.
4. Query one or more models.
5. Store outputs together with slot settings.
6. Analyze consistency, value preference, principle use, and narrative sensitivity.


## Canonical Location
- Canonical package path: `examples/benchmarks/ethics_sandbox/package`
- This `docs/` directory is kept as a compatibility mirror/alias for existing commands and references.
