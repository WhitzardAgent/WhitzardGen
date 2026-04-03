# Fixed Interfaces and Module Names

Codex should implement the codebase toward these stable targets unless there is a compelling reason to change them.

## Required package modules
- `src/ethics_pipeline/templates/loader.py`
- `src/ethics_pipeline/templates/validators.py`
- `src/ethics_pipeline/generation/policies.py`
- `src/ethics_pipeline/generation/engine.py`
- `src/ethics_pipeline/prompts/compiler.py`
- `src/ethics_pipeline/prompts/naturalism.py`
- `src/ethics_pipeline/models/registry.py`
- `src/ethics_pipeline/models/launcher.py`
- `src/ethics_pipeline/models/scheduler.py`
- `src/ethics_pipeline/models/health.py`
- `src/ethics_pipeline/orchestration/run_planner.py`
- `src/ethics_pipeline/orchestration/executor.py`
- `src/ethics_pipeline/storage/writers.py`
- `src/ethics_pipeline/storage/readers.py`
- `src/ethics_pipeline/normalization/parser.py`
- `src/ethics_pipeline/analysis/policies.py`
- `src/ethics_pipeline/analysis/judges.py`
- `src/ethics_pipeline/analysis/metrics.py`
- `src/ethics_pipeline/reporting/builders.py`
- `src/ethics_pipeline/cli/main.py`
- `src/ethics_pipeline/ui/backend/app.py`

## Required public object names
- `SandboxTemplateLoader`
- `VariantGenerationEngine`
- `PromptCompiler`
- `ModelRegistry`
- `ModelLauncherRegistry`
- `ResourceAwareScheduler`
- `RunPlanner`
- `RunExecutor`
- `ResponseNormalizer`
- `AnalysisPolicyLoader`
- `LLMJudgeEngine`
- `ReportBuilder`
