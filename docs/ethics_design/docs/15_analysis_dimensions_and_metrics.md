# Analysis Dimensions and Metrics

## 1. Decision outcome metrics
- action label distribution
- refusal rate
- undecided / ambivalent rate
- action polarity under parameter changes

## 2. Principle-use metrics
- primary principle inferred
- secondary principle inferred
- frequency of consequentialist language
- frequency of deontic language
- autonomy-centered vs welfare-centered rationale balance

## 3. Consistency metrics
- within-template slot sensitivity
- perturbation robustness
- cross-template family consistency
- cross-model agreement
- repeated-run stability under same prompt and sampling policy

## 4. Value-preference metrics
- preference for autonomy vs beneficence
- preference for neutrality vs loyalty
- preference for truthfulness vs harm prevention
- preference for opacity tolerance vs explainability demand
- preference for sacred-value protection vs aggregate saving

## 5. Reasoning-shape metrics
Only when rationale or reasoning trace is available:
- number of stakeholders explicitly represented
- whether alternatives are compared
- whether rules and consequences are both considered
- whether uncertainty is acknowledged
- whether the model notices a key invariant of the template

## 6. Operational metrics
- latency
- timeout rate
- transport failure rate
- token usage if available
- parse failure rate

## Implementation note
These metrics should be defined in code as explicit functions or schema-constrained judge tasks, not left as ad hoc notebook logic.

## 7. Extensibility rule
Every analysis metric or judge should be implemented as a named, versioned analysis plugin or plugin-backed function, so new methods can be added without changing core orchestration.
