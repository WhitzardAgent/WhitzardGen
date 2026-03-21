# Model Benchmarks

This file is the lightweight operator-facing benchmark table for current core models.

Use it together with:

- [configs/model_benchmarks.yaml](/Users/morinop/coding/whitzardgen/configs/model_benchmarks.yaml)
- [docs/model_capability_matrix.md](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.md)
- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)

## Purpose

Track:

- recommended default generation parameters
- approximate latency and throughput after real runs
- replica scaling observations
- VRAM and environment caveats

The first version is a scaffold. Fill it in after real canary and burn-in runs.

## Core Models To Track First

| Model | Modality | Benchmark Status | Key Things To Measure |
| --- | --- | --- | --- |
| Z-Image | image | pending | batch throughput, single-GPU latency, preferred batch size stability |
| LongCat-Video | video | pending | effect of `num_frames`, `num_inference_steps`, and `use_distill` on latency |
| Helios | video | pending | practicality of 240-frame defaults, callback progress behavior, persistent-worker reuse |
| Wan2.2-T2V-A14B-Diffusers | video | pending | startup reliability, batch throughput, multi-replica scaling |
| CogVideoX-5B | video | pending | best replica count, canary speed, batch throughput |

## Recommended Recording Fields

For each real benchmark run, record at least:

- model name
- resolution
- fps
- num_frames
- num_inference_steps
- batch size
- replica count
- approximate latency per task
- approximate prompt throughput
- approximate VRAM usage
- recommended default params after tuning
- caveats and failure notes

## Suggested Workflow

1. Start from the current defaults in [configs/models](/Users/morinop/coding/whitzardgen/configs/models).
2. Use [configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models) for machine-local overrides.
3. Run `aigc models canary <model>` first.
4. Then run a slightly longer real prompt set.
5. Update [configs/model_benchmarks.yaml](/Users/morinop/coding/whitzardgen/configs/model_benchmarks.yaml) with the measured results and recommended tuned defaults.
