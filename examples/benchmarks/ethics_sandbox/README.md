# Ethics Sandbox Example

This directory contains an example structural-scenario benchmark implemented on top of the generic benchmark core.

The builder in this example now uses the generic semantic-realization build stage:

```text
slot sampling
-> structure guard
-> synthesis template rendering
-> T2T realization via existing run kernel
-> final case compilation
```

Contents:

- `builder.yaml`: builder manifest used by `aigc benchmark list/build`
- `builder.py`: example benchmark builder and example group analyzer
- `example_build.yaml`: semantic-build config with sampling / synthesis / validation sections
- `synthesis_templates/`: build-time templates used to realize naturalistic scenarios

The actual template package consumed by this example remains in:

- [docs/ethics_design/sandbox_template](/Users/morinop/coding/whitzardgen/docs/ethics_design/sandbox_template)

Operational runbook:

- [examples/experiments/ethics_structural_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural_runbook.zh-CN.md)
