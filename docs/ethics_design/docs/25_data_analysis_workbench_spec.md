# Data Analysis Workbench Spec

## First-class analysis operations
The workbench must support these as saved operations, not ad hoc notebook-only logic:

### A. Cross-model comparison
For a chosen slice of runs:
- compare action label distributions across models
- compare refusal rates
- compare principle labels
- compare rationale length / ambiguity

### B. Within-template parameter sensitivity
For a chosen template family:
- estimate how each structural slot shifts decision labels
- estimate how each perturbation slot shifts decision labels
- compare sensitivity across models

### C. Pairwise slot interaction analysis
For selected templates:
- identify whether combinations of slots create decision reversals
- surface interaction heatmaps

### D. Robustness analysis
- repeated-run stability
- perturbation invariance
- sensitivity to narrative ordering or humanization bias

### E. Reasoning-shape analysis
Only when rationale or reasoning trace exists:
- stakeholder coverage
- rule vs consequence balance
- uncertainty acknowledgement
- alternative consideration

## Output requirement
All these analyses must be exportable as structured tables, not only charts.

## Plugin visibility
The workbench should expose installed analysis plugins, plugin versions, plugin outputs, and plugin-specific configuration used for each run.
