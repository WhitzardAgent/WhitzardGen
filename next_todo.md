# Next TODO

## Current Baseline

- Persistent worker has been validated as workable on the remote server.
- `Wan2.2-T2V-A14B-Diffusers` and `CogVideoX-5B` have both moved into real cluster validation.
- `running.log`, `samples.jsonl`, `run_manifest.json`, and final `exports/dataset.jsonl` are all in place.
- The framework is no longer blocked on the original "one task, one model load" runtime limitation for the main persistent-worker path.

## Highest Priority

### 1. Sequential Replica Warmup

Goal:
- when a model uses multiple replicas, do **not** load all replicas at once

Why:
- the first replica pays the full disk-to-CPU/GPU loading cost
- later replicas can benefit from warmed filesystem cache
- concurrent loading increases disk/CPU contention and can make startup slower or less stable

Implementation target:
- start replica 0
- wait until replica 0 reaches `ready`
- start replica 1
- wait until replica 1 reaches `ready`
- continue until all replicas are ready
- only then begin task dispatch

Desired logs:
- `starting replica 0/4`
- `replica 0 ready`
- `starting replica 1/4`
- `all replicas ready, dispatching tasks`

### 2. Real Multi-Replica Validation

Goal:
- confirm that multi-replica execution is stable and actually improves throughput on the cluster

Priority models:
- `Z-Image`
- `CogVideoX-5B`
- `Wan2.2-T2V-A14B-Diffusers`

Validation points:
- each replica loads exactly once
- task sharding is correct
- `samples.jsonl` shows prompt-level results continuously
- `running.log` clearly shows replica startup, readiness, task assignment, and failures
- output artifacts and final dataset export remain correct

### 3. Persistent Worker Runtime Hardening

Goal:
- make the persistent-worker path robust enough for longer real jobs

Remaining checks:
- verify workers survive multiple task batches without leaking obvious state
- verify task failure attribution stays clear after real inference exceptions
- verify shutdown is clean after partial failure of one replica
- verify per-replica logs are sufficient for postmortem debugging

## Medium Priority

### 4. Better Runtime Progress / Terminal UX

Goal:
- make long-running cluster jobs easier to trust while they are running

Needed improvements:
- clearer "model loading" vs "task running" vs "artifact exporting" boundaries
- better per-replica progress visibility
- more readable final summary for multi-replica runs
- optional concise throughput counters if practical

### 5. Prompt/Batch Throughput Tuning

Goal:
- tune practical batch settings for real diffusers models

Targets:
- `Wan2.2-T2V-A14B-Diffusers`
- `CogVideoX-5B`
- diffusers-based image models

Questions to answer:
- what batch sizes are actually stable on the cluster
- whether prompt batching plus persistent workers gives the expected throughput gains
- whether video batch size should remain conservative for memory safety

### 6. Environment Preflight Hardening

Goal:
- catch cluster env issues earlier, before worker startup

Next checks worth adding:
- better validation for model-specific optional dependencies
- more explicit doctor output for manually prepared envs
- better diagnostics when a local weights directory exists but is incomplete

## Still Missing By Architecture

### 7. Scheduler / Retry / Resume

Still not done:
- dedicated scheduler subsystem
- `aigc runs retry`
- `aigc runs resume`
- interrupted-run recovery

Why it matters next:
- `samples.jsonl` is now in place, so the foundation for retry/resume is better than before
- once multi-replica execution is stable, retry/resume becomes more valuable

### 8. Richer Export / Dataset Organization

Still open:
- parquet export
- more formal artifact validation
- stronger dataset packaging / organization passes after run completion

## Recommended Execution Order

1. Implement sequential replica warmup.
2. Re-run multi-replica real canaries for `Z-Image`, `CogVideoX-5B`, and `Wan2.2-T2V-A14B-Diffusers`.
3. Harden persistent-worker failure handling based on real cluster feedback.
4. Improve runtime progress visibility for long jobs.
5. Move into scheduler / retry / resume work once real runtime behavior is stable.
