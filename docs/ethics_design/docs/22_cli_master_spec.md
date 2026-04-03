# CLI Master Spec

## Command groups

### System
- `doctor`
- `ui`
- `test smoke`
- `test all`

### Models
- `models status`
- `models up`
- `models down`
- `models plan`

### Pipeline
- `templates validate`
- `variants generate`
- `prompts compile`
- `run execute`
- `run resume`
- `responses normalize`
- `analysis run`
- `reports build`

## Command design rules
- every command must support `--config` or explicit input/output paths
- commands must emit machine-readable summaries
- destructive commands must confirm intent unless `--yes` is passed
- `run execute` and `test all` must support `--auto-launch`

## Required outputs
- `models plan` writes a launch plan artifact
- `run execute` writes a run manifest immediately at start
- `analysis run` writes analysis manifest and versioned policy snapshot
