# Task P1 Summary

## Executive Answer

No, the project is **not yet at a state where only remote-server validation remains**.

The more accurate status is:

- the **mock-capable MVP framework is largely in place**
- the **cluster bring-up / real-model validation phase is now appropriate**
- but there are still a few **spec-defined features and architecture pieces that are not fully implemented**

So the correct conclusion is:

> We are past the "empty framework" stage and ready for remote real-mode validation, but we are **not yet fully done by the docs**.

---

## High-Level Verdict

### What is already solid

- Prompt loading is implemented for `.txt`, `.csv`, and `.jsonl`.
- Registry coverage exists for all 11 MVP target models.
- Environment management exists and now supports foreground synchronous creation for `aigc run`.
- Image and video adapters exist for all MVP target models.
- Local `mock` mode works across image and video flows.
- `aigc run`, `models`, `runs`, `export dataset`, and `doctor` are present and usable.
- Run directories, manifests, failures summaries, and JSONL dataset export are implemented.
- Local model path overrides (`configs/local_models.yaml`) are implemented.

### What is not yet truly complete

- The **scheduler subsystem** is still only a minimal sequential orchestration path inside `run_flow.py`; the dedicated scheduler described in the docs is not implemented.
- `aigc runs retry` and `aigc runs resume` are **not implemented**.
- Retry / resume behavior from the pipeline and scheduler specs is **not implemented**.
- Prompt preprocessing is still **minimal**; model-specific translation / adaptation / token-length enforcement from the pipeline docs is not implemented.
- Artifact validation is still **minimal**; there is collection and metadata extraction, but not a dedicated validation stage.
- Parquet export is **not implemented**.
- Real end-to-end success for all target models has **not been demonstrated yet**.

---

## Docs-to-Implementation Status

| Area | Status | Notes |
|---|---|---|
| Repository / package skeleton | Done | Package layout, entrypoints, install files, tests are in place. |
| Prompt input system | Done for MVP | `.txt`, `.csv`, `.jsonl` all work. Validation is present. |
| Model registry | Done for MVP | All target image/video models are registered and inspectable. |
| Env manager | Mostly done | Conda env spec resolution, metadata, validation, doctor, local path awareness, and blocking `ensure_ready(...)` are implemented. Needs real cluster validation. |
| Runtime worker | Done for MVP | Worker payloads, result handling, adapter execution, and workdir flow exist. |
| Image adapters | Structurally done | All target image models have adapter bindings; local mock path works; real execution path exists but needs cluster validation. |
| Video adapters | Structurally done | All target video models have adapter bindings; local mock path works; real execution path exists structurally but needs cluster validation. |
| Run flow | Mostly done | Multi-model, modality-consistent, batch-aware, mock/real-aware flow exists. Still acts as a lightweight orchestration layer, not a full scheduler. |
| Dataset export | Mostly done | JSONL export and prompt-to-artifact mapping are in place. Parquet not implemented. |
| CLI MVP | Partially done | `models list`, `models inspect`, `run`, `runs list`, `runs inspect`, `runs failures`, `export dataset`, `doctor`, `version` exist. `runs retry` / `runs resume` do not. |
| Scheduler core | Not done | `src/aigc/scheduler/` is effectively empty; no queue, concurrency control, retry engine, or resume engine. |

---

## What Is Still Missing By Spec

These are the clearest **implementation gaps**, not just validation gaps.

### 1. Scheduler Core Is Still Missing

Per `docs/scheduler_spec.md` and Phase 10 in `docs/codex_tasks.md`, the scheduler should own:

- task queueing
- task state tracking
- concurrency control
- retry handling
- run state tracking
- resume support

Current reality:

- batching and task expansion exist
- sequential execution exists
- run manifest / failures summaries exist
- but there is **no real scheduler subsystem**
- `src/aigc/scheduler/__init__.py` is only a stub

This is the single biggest architecture gap still remaining.

### 2. Retry / Resume Are Missing

The docs explicitly call for:

- basic retry
- interrupted run resume
- CLI commands:
  - `aigc runs retry <run_id>`
  - `aigc runs resume <run_id>`

Current reality:

- failures can be recorded and inspected
- but failed runs cannot yet be retried through the framework
- interrupted runs cannot yet resume
- corresponding CLI commands are absent

### 3. Prompt Preprocessing Is Still Minimal

The pipeline docs describe preprocessing responsibilities such as:

- translation when the model does not support a language
- model-specific prompt adaptation
- negative-prompt normalization
- token-length enforcement

Current reality:

- whitespace normalization exists
- language inference/defaulting exists
- batching exists
- but there is no real model-aware preprocessing layer yet

This is a real missing subsystem, although it may not block first cluster canaries.

### 4. Artifact Validation Stage Is Minimal

The pipeline DAG includes an artifact validation stage after artifact collection.

Current reality:

- artifacts are collected
- metadata is extracted
- existence checks happen implicitly during collection
- but there is no explicit standalone artifact validation stage with pass/fail policy

### 5. CLI Spec Is Not Fully Complete

Compared with `docs/cli_spec.md`, the CLI is in good shape but not fully complete.

Main remaining gaps:

- `aigc runs retry`
- `aigc runs resume`
- richer global conventions like `--verbose` / `--quiet`
- more differentiated exit codes beyond the current simple error handling
- no meaningful run-time terminal progress UI yet for long-running jobs

### 6. Terminal Run UX Is Now A High-Priority Gap

This was not originally emphasized enough in the planning docs, but cluster usage now
makes it clear that it should be treated as a real priority.

Current reality:

- environment creation has some foreground logs
- but task execution has almost no ongoing user feedback
- there is no task-level progress summary
- there is no clear "current stage" display
- there is no heartbeat/spinner/progress bar for long inference windows
- the CLI can look idle even when the framework is still working correctly

Why this matters:

- on long-running GPU jobs, silent terminals create operational uncertainty
- users cannot easily tell whether the framework is stuck, downloading, validating, or generating
- this increases the cost of cluster debugging and hurts trust in the framework

Recommended priority:

- this should be treated as a **high-priority usability and operability task**, not a cosmetic nice-to-have

Recommended MVP scope for terminal UX:

- stage messages:
  - loading prompts
  - resolving models
  - ensuring environments
  - preparing tasks
  - running task `i/n`
  - exporting dataset
- per-task start/finish lines
- a concise run summary footer
- optional richer spinner/progress rendering later, potentially with `rich`

### 7. Parquet Export Is Not Implemented

This is optional/recommended in the task doc, not the biggest blocker, but it is still not done.

---

## What Looks Implemented But Still Needs Remote Real Validation

These parts are **not obviously missing in code**, but they still need cluster proof.

### 1. Real Environment Provisioning

We now have the correct blocking env-creation behavior in the run path, but still need to validate on the GPU cluster that:

- Conda env creation succeeds for each env spec
- pip requirements actually install
- validation imports pass
- metadata and doctor output stay sane under real failures

### 2. Real Model Execution For All Target Models

All target image/video models are integrated structurally, but we have not yet proven in the cluster that each one:

- launches correctly
- resolves local weights / repo paths correctly
- emits real artifacts where the adapter expects them
- behaves correctly under its real runtime constraints

This is especially important for the external-process video models.

### 3. External-Process Video Command Compatibility

For models like:

- `Wan2.2-TI2V-5B`
- `LongCat-Video`
- `MOVA-720p`

the framework has real command construction paths, but those commands still need to be proven against the actual checked-out repos and script interfaces on the cluster.

### 4. Real Batch Validation

Batch-aware image flow exists in code, but the docs require that at least one batch-capable image model **actually runs in batch mode**.

That still needs real cluster validation.

---

## Strict Reading Of `docs/codex_tasks.md`

If we judge completion strictly against the MVP success criteria, the project is **not yet complete**.

The most important reasons are:

1. real successful inference for all target models has not yet been demonstrated
2. scheduler-core done criteria are not met
3. retry/resume CLI and runtime behavior are not implemented

So the honest answer is:

- **Framework MVP skeleton + mock/local development path:** largely complete
- **Cluster bring-up readiness:** yes, mostly ready
- **Strict docs-complete MVP:** not yet

---

## Practical Recommendation

The next work should be split into two tracks.

### Track A: Immediate cluster validation

This is the right next operational step now.

- populate `configs/local_models.yaml`
- create envs via `aigc run` / `aigc doctor`
- run canary image/video jobs in `real` mode
- validate artifact paths, command compatibility, and weights resolution

### Track B: Remaining implementation gaps

These are still real coding tasks after cluster bring-up starts.

- implement scheduler core as a real subsystem
- add `runs retry`
- add `runs resume`
- add at least minimal resume/retry semantics in run metadata and dispatch flow
- decide how much prompt preprocessing must exist for MVP
- optionally add explicit artifact validation and Parquet export

---

## Bottom Line

The shortest truthful summary is:

> We are **not** in a state where only remote validation remains.
> We **are** in a state where remote validation is the correct next step.
> But to claim the project is fully complete against the docs, we still need:
> real-model cluster proof, scheduler-core work, retry/resume support, and a much better run-time terminal UX.
