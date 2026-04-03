# Ethics Sandbox Example

This directory contains an example structural-scenario benchmark implemented on top of the generic benchmark core.

The builder in this example now uses the generic semantic-realization build stage:

```text
slot sampling
-> structure guard
-> writer prompt rendering
-> T2T realization via existing run kernel
-> validator prompt rendering
-> validator judgment / retry feedback
-> final case compilation
```

Contents:

- `builder.yaml`: builder manifest used by `aigc benchmark list/build`
- `builder.py`: example benchmark builder and example group analyzer
- `example_build.yaml`: semantic-build config with sampling / synthesis / validator / validation sections
- `synthesis_templates/`: build-time writer and validator templates used to realize and validate naturalistic scenarios

The canonical example-owned benchmark package now lives in:

- [package](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/package)

The legacy docs path remains runnable as a compatibility alias:

- [docs/ethics_design/sandbox_template](/Users/morinop/coding/whitzardgen/docs/ethics_design/sandbox_template)

Operational runbook:

- [examples/experiments/ethics_structural_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural_runbook.zh-CN.md)
