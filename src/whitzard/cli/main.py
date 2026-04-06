import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from whitzard import __version__
from whitzard.analysis import AnalysisConfigError, AnalysisError
from whitzard.annotation import AnnotationConfigError, AnnotationError, annotate_run
from whitzard.benchmarking import (
    BenchmarkingError,
    build_benchmark,
    evaluate_benchmark,
    inspect_benchmark_bundle,
    inspect_experiment,
    list_benchmark_builders,
    list_experiments,
)
from whitzard.benchmarking.recipes import ExperimentRecipeError, load_experiment_recipe
from whitzard.evaluators import EvaluatorConfigError, EvaluatorError
from whitzard.env import EnvManager, EnvManagerError
from whitzard.launching import LaunchError
from whitzard.normalizers import NormalizerConfigError, NormalizerError
from whitzard.prompt_generation import (
    PromptGenerationSummary,
    generate_prompt_bundle,
    inspect_prompt_bundle,
    plan_theme_tree,
)
from whitzard.prompt_generation.service import PromptGenerationError
from whitzard.recovery import (
    RecoveryError,
    build_resume_plan,
    build_retry_plan,
    recovery_plan_to_dict,
)
from whitzard.model_onboarding import (
    build_model_capability_rows,
    render_model_capability_matrix_markdown,
    run_model_canary,
    write_model_capability_docs,
)
from whitzard.run_profiles import (
    RunProfileError,
    apply_profile_runtime_environment,
    load_run_profile,
    resolve_profile_run_request,
)
from whitzard.registry import DEFAULT_LOCAL_MODELS_PATH, RegistryError, load_registry
from whitzard.registry.local_overrides import summarize_local_path_overrides
from whitzard.run_store import (
    RunStoreError,
    export_dataset_for_runs,
    list_runs,
    load_failures_summary,
    load_run_manifest,
)
from whitzard.run_flow import RunFlowError, run_models, run_recovery_plan
from whitzard.settings import get_runs_root
from whitzard.utils.progress import build_run_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whitzard")
    subparsers = parser.add_subparsers(dest="command")

    models_parser = subparsers.add_parser("models", help="Inspect registered models.")
    models_subparsers = models_parser.add_subparsers(dest="models_command")

    models_list_parser = models_subparsers.add_parser("list", help="List registered models.")
    models_list_parser.add_argument("--modality", choices=["image", "video", "audio", "text"])
    models_list_parser.add_argument("--task-type")
    models_list_parser.add_argument("--output", choices=["text", "json"], default="text")
    models_list_parser.set_defaults(handler=handle_models_list)

    models_inspect_parser = models_subparsers.add_parser(
        "inspect", help="Inspect one registered model."
    )
    models_inspect_parser.add_argument("model_name")
    models_inspect_parser.add_argument("--output", choices=["text", "json"], default="text")
    models_inspect_parser.set_defaults(handler=handle_models_inspect)

    models_canary_parser = models_subparsers.add_parser(
        "canary", help="Run a minimal one-model canary generation."
    )
    models_canary_parser.add_argument("model_name")
    models_canary_parser.add_argument("--prompt-file")
    models_canary_parser.add_argument("--out")
    models_canary_parser.add_argument("--mock", action="store_true")
    models_canary_parser.add_argument("--execution-mode", choices=["mock", "real"])
    models_canary_parser.add_argument("--output", choices=["text", "json"], default="text")
    models_canary_parser.set_defaults(handler=handle_models_canary)

    models_matrix_parser = models_subparsers.add_parser(
        "matrix", help="Show or write the model capability matrix."
    )
    models_matrix_parser.add_argument("--output", choices=["text", "json"], default="text")
    models_matrix_parser.add_argument("--write-docs", action="store_true")
    models_matrix_parser.add_argument("--docs-dir")
    models_matrix_parser.set_defaults(handler=handle_models_matrix)

    doctor_parser = subparsers.add_parser("doctor", help="Check environment readiness.")
    doctor_parser.add_argument("--model")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text")
    doctor_parser.set_defaults(handler=handle_doctor)

    prompts_parser = subparsers.add_parser("prompts", help="Build or inspect prompt bundles.")
    prompts_subparsers = prompts_parser.add_subparsers(dest="prompts_command")

    prompts_generate_parser = prompts_subparsers.add_parser(
        "generate", help="Generate a prompt bundle from a theme-tree YAML."
    )
    prompts_generate_parser.add_argument("--tree", required=True)
    prompts_generate_parser.add_argument("--out")
    prompts_generate_parser.add_argument("--count-config")
    prompts_generate_parser.add_argument("--llm-model")
    prompts_generate_parser.add_argument("--mock", action="store_true")
    prompts_generate_parser.add_argument("--execution-mode", choices=["mock", "real"])
    prompts_generate_parser.add_argument("--seed", type=int, default=42)
    prompts_generate_parser.add_argument("--profile")
    prompts_generate_parser.add_argument("--template")
    prompts_generate_parser.add_argument("--style-family")
    prompts_generate_parser.add_argument("--target-model")
    prompts_generate_parser.add_argument("--intended-modality")
    prompts_generate_parser.add_argument("--output", choices=["text", "json"], default="text")
    prompts_generate_parser.set_defaults(handler=handle_prompts_generate)

    prompts_inspect_parser = prompts_subparsers.add_parser(
        "inspect", help="Inspect a generated prompt bundle or prompts JSONL file."
    )
    prompts_inspect_parser.add_argument("path")
    prompts_inspect_parser.add_argument("--output", choices=["text", "json"], default="text")
    prompts_inspect_parser.set_defaults(handler=handle_prompts_inspect)

    prompts_plan_parser = prompts_subparsers.add_parser(
        "plan", help="Plan sampling for a theme-tree YAML without generating prompts."
    )
    prompts_plan_parser.add_argument("--tree", required=True)
    prompts_plan_parser.add_argument("--count-config")
    prompts_plan_parser.add_argument("--seed", type=int, default=42)
    prompts_plan_parser.add_argument("--output", choices=["text", "json"], default="text")
    prompts_plan_parser.set_defaults(handler=handle_prompts_plan)

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Run a post-generation annotation job for one source run.",
    )
    annotate_parser.add_argument("run_id")
    annotate_parser.add_argument("--profile")
    annotate_parser.add_argument("--model")
    annotate_parser.add_argument("--template")
    annotate_parser.add_argument("--out")
    annotate_parser.add_argument("--mock", action="store_true")
    annotate_parser.add_argument("--execution-mode", choices=["mock", "real"])
    annotate_parser.add_argument("--output", choices=["text", "json"], default="text")
    annotate_parser.set_defaults(handler=handle_annotate)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Build or inspect benchmark bundles.",
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command")

    benchmark_list_parser = benchmark_subparsers.add_parser(
        "list",
        help="List available benchmark builders.",
    )
    benchmark_list_parser.add_argument("--output", choices=["text", "json"], default="text")
    benchmark_list_parser.set_defaults(handler=handle_benchmark_list)

    benchmark_build_parser = benchmark_subparsers.add_parser(
        "build",
        help="Build a benchmark bundle from a supported source or builder.",
    )
    benchmark_build_parser.add_argument("--builder", required=True)
    benchmark_build_parser.add_argument("--source")
    benchmark_build_parser.add_argument("--package")
    benchmark_build_parser.add_argument("--entrypoint")
    benchmark_build_parser.add_argument("--builder-config", dest="builder_config")
    benchmark_build_parser.add_argument("--config", dest="builder_config")
    benchmark_build_parser.add_argument("--count-config")
    benchmark_build_parser.add_argument("--llm-model")
    benchmark_build_parser.add_argument("--synthesis-model")
    benchmark_build_parser.add_argument("--profile")
    benchmark_build_parser.add_argument("--template")
    benchmark_build_parser.add_argument("--style-family")
    benchmark_build_parser.add_argument("--target-model")
    benchmark_build_parser.add_argument("--intended-modality")
    benchmark_build_parser.add_argument("--out")
    benchmark_build_parser.add_argument("--benchmark-name")
    benchmark_build_parser.add_argument("--seed", type=int, default=42)
    benchmark_build_parser.add_argument("--realizations-per-template", type=int, default=1)
    benchmark_build_parser.add_argument("--mock", action="store_true")
    benchmark_build_parser.add_argument("--execution-mode", choices=["mock", "real"])
    benchmark_build_parser.add_argument("--build-mode", choices=["static", "matrix"], default="static")
    _add_preview_arguments(
        benchmark_build_parser,
        stage_choices=["writer", "validator", "all"],
    )
    benchmark_build_parser.add_argument("--output", choices=["text", "json"], default="text")
    benchmark_build_parser.set_defaults(handler=handle_benchmark_build)

    benchmark_preview_parser = benchmark_subparsers.add_parser(
        "preview",
        help="Preview rendered benchmark-build requests without executing models.",
    )
    benchmark_preview_parser.add_argument("--builder", required=True)
    benchmark_preview_parser.add_argument("--source")
    benchmark_preview_parser.add_argument("--package")
    benchmark_preview_parser.add_argument("--entrypoint")
    benchmark_preview_parser.add_argument("--builder-config", dest="builder_config")
    benchmark_preview_parser.add_argument("--config", dest="builder_config")
    benchmark_preview_parser.add_argument("--count-config")
    benchmark_preview_parser.add_argument("--llm-model")
    benchmark_preview_parser.add_argument("--synthesis-model")
    benchmark_preview_parser.add_argument("--profile")
    benchmark_preview_parser.add_argument("--template")
    benchmark_preview_parser.add_argument("--style-family")
    benchmark_preview_parser.add_argument("--target-model")
    benchmark_preview_parser.add_argument("--intended-modality")
    benchmark_preview_parser.add_argument("--out")
    benchmark_preview_parser.add_argument("--benchmark-name")
    benchmark_preview_parser.add_argument("--seed", type=int, default=42)
    benchmark_preview_parser.add_argument("--realizations-per-template", type=int, default=1)
    benchmark_preview_parser.add_argument("--mock", action="store_true")
    benchmark_preview_parser.add_argument("--execution-mode", choices=["mock", "real"])
    benchmark_preview_parser.add_argument("--build-mode", choices=["static", "matrix"], default="static")
    benchmark_preview_parser.add_argument("--preview-count", type=int, default=5)
    benchmark_preview_parser.add_argument("--preview-stage", choices=["writer", "validator", "all"], default="all")
    benchmark_preview_parser.add_argument("--preview-format", choices=["text", "json", "md"], default="text")
    benchmark_preview_parser.add_argument("--output", choices=["text", "json"], default="text")
    benchmark_preview_parser.set_defaults(handler=handle_benchmark_preview)

    benchmark_inspect_parser = benchmark_subparsers.add_parser(
        "inspect",
        help="Inspect a benchmark bundle.",
    )
    benchmark_inspect_parser.add_argument("path")
    benchmark_inspect_parser.add_argument("--output", choices=["text", "json"], default="text")
    benchmark_inspect_parser.set_defaults(handler=handle_benchmark_inspect)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run benchmark evaluation experiments.",
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command")

    evaluate_run_parser = evaluate_subparsers.add_parser(
        "run",
        help="Run one benchmark across one or more target models.",
    )
    evaluate_run_parser.add_argument("--recipe")
    evaluate_run_parser.add_argument("--benchmark")
    evaluate_run_parser.add_argument("--targets", nargs="+")
    evaluate_run_parser.add_argument("--normalizers", nargs="+")
    evaluate_run_parser.add_argument("--evaluators", nargs="+")
    evaluate_run_parser.add_argument("--analysis-plugins", nargs="+")
    evaluate_run_parser.add_argument("--normalizer-config")
    evaluate_run_parser.add_argument("--evaluator-model")
    evaluate_run_parser.add_argument("--evaluator-profile")
    evaluate_run_parser.add_argument("--evaluator-template")
    evaluate_run_parser.add_argument("--evaluator-config")
    evaluate_run_parser.add_argument("--analysis-config")
    evaluate_run_parser.add_argument("--launcher-config")
    evaluate_run_parser.add_argument("--auto-launch", action="store_true")
    evaluate_run_parser.add_argument("--out")
    evaluate_run_parser.add_argument("--mock", action="store_true")
    evaluate_run_parser.add_argument("--execution-mode", choices=["mock", "real"])
    _add_preview_arguments(
        evaluate_run_parser,
        stage_choices=["target", "judge", "all"],
    )
    evaluate_run_parser.add_argument("--output", choices=["text", "json"], default="text")
    evaluate_run_parser.set_defaults(handler=handle_evaluate_run)

    evaluate_preview_parser = evaluate_subparsers.add_parser(
        "preview",
        help="Preview rendered evaluation requests without executing target or judge models.",
    )
    evaluate_preview_parser.add_argument("--recipe")
    evaluate_preview_parser.add_argument("--benchmark")
    evaluate_preview_parser.add_argument("--targets", nargs="+")
    evaluate_preview_parser.add_argument("--normalizers", nargs="+")
    evaluate_preview_parser.add_argument("--evaluators", nargs="+")
    evaluate_preview_parser.add_argument("--analysis-plugins", nargs="+")
    evaluate_preview_parser.add_argument("--normalizer-config")
    evaluate_preview_parser.add_argument("--evaluator-model")
    evaluate_preview_parser.add_argument("--evaluator-profile")
    evaluate_preview_parser.add_argument("--evaluator-template")
    evaluate_preview_parser.add_argument("--evaluator-config")
    evaluate_preview_parser.add_argument("--analysis-config")
    evaluate_preview_parser.add_argument("--launcher-config")
    evaluate_preview_parser.add_argument("--auto-launch", action="store_true")
    evaluate_preview_parser.add_argument("--out")
    evaluate_preview_parser.add_argument("--mock", action="store_true")
    evaluate_preview_parser.add_argument("--execution-mode", choices=["mock", "real"])
    evaluate_preview_parser.add_argument("--preview-count", type=int, default=5)
    evaluate_preview_parser.add_argument("--preview-stage", choices=["target", "judge", "all"], default="all")
    evaluate_preview_parser.add_argument("--preview-format", choices=["text", "json", "md"], default="text")
    evaluate_preview_parser.add_argument("--output", choices=["text", "json"], default="text")
    evaluate_preview_parser.set_defaults(handler=handle_evaluate_preview)

    evaluate_inspect_parser = evaluate_subparsers.add_parser(
        "inspect",
        help="Inspect one experiment bundle or experiment id.",
    )
    evaluate_inspect_parser.add_argument("experiment")
    evaluate_inspect_parser.add_argument("--output", choices=["text", "json"], default="text")
    evaluate_inspect_parser.set_defaults(handler=handle_evaluate_inspect)

    experiments_parser = subparsers.add_parser(
        "experiments",
        help="Inspect benchmark evaluation experiments.",
    )
    experiments_subparsers = experiments_parser.add_subparsers(dest="experiments_command")

    experiments_list_parser = experiments_subparsers.add_parser(
        "list",
        help="List recorded experiment bundles.",
    )
    experiments_list_parser.add_argument("--output", choices=["text", "json"], default="text")
    experiments_list_parser.set_defaults(handler=handle_experiments_list)

    experiments_report_parser = experiments_subparsers.add_parser(
        "report",
        help="Show experiment manifest and summary.",
    )
    experiments_report_parser.add_argument("experiment")
    experiments_report_parser.add_argument("--output", choices=["text", "json"], default="text")
    experiments_report_parser.set_defaults(handler=handle_experiments_report)

    runs_parser = subparsers.add_parser("runs", help="Inspect generation runs.")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")

    runs_list_parser = runs_subparsers.add_parser("list", help="List recorded runs.")
    runs_list_parser.add_argument("--output", choices=["text", "json"], default="text")
    runs_list_parser.set_defaults(handler=handle_runs_list)

    runs_inspect_parser = runs_subparsers.add_parser("inspect", help="Inspect one run manifest.")
    runs_inspect_parser.add_argument("run_id")
    runs_inspect_parser.add_argument("--output", choices=["text", "json"], default="text")
    runs_inspect_parser.set_defaults(handler=handle_runs_inspect)

    runs_failures_parser = runs_subparsers.add_parser(
        "failures", help="Show failure summary for one run."
    )
    runs_failures_parser.add_argument("run_id")
    runs_failures_parser.add_argument("--output", choices=["text", "json"], default="text")
    runs_failures_parser.set_defaults(handler=handle_runs_failures)

    runs_retry_parser = runs_subparsers.add_parser("retry", help="Retry failed work from an existing run.")
    runs_retry_parser.add_argument("run_id")
    runs_retry_parser.add_argument("--model")
    runs_retry_parser.add_argument("--output", choices=["text", "json"], default="text")
    runs_retry_parser.set_defaults(handler=handle_runs_retry)

    runs_resume_parser = runs_subparsers.add_parser("resume", help="Resume missing work from an existing run.")
    runs_resume_parser.add_argument("run_id")
    runs_resume_parser.add_argument("--model")
    runs_resume_parser.add_argument("--output", choices=["text", "json"], default="text")
    runs_resume_parser.set_defaults(handler=handle_runs_resume)

    export_parser = subparsers.add_parser("export", help="Inspect exported dataset outputs.")
    export_subparsers = export_parser.add_subparsers(dest="export_command")

    export_dataset_parser = export_subparsers.add_parser(
        "dataset", help="Materialize an organized dataset export bundle for one or more runs."
    )
    export_dataset_parser.add_argument("run_ids", nargs="+")
    export_dataset_parser.add_argument("--out")
    export_dataset_parser.add_argument("--mode", choices=["link", "copy"], default="link")
    export_dataset_parser.add_argument("--model", action="append", default=[])
    export_dataset_parser.add_argument("--output", choices=["text", "json"], default="text")
    export_dataset_parser.set_defaults(handler=handle_export_dataset)

    run_parser = subparsers.add_parser("run", help="Run a minimal generation job.")
    run_parser.add_argument("--profile")
    run_parser.add_argument("--models")
    run_parser.add_argument("--prompts")
    run_parser.add_argument("--run-name")
    run_parser.add_argument("--out")
    run_parser.add_argument("--mock", action="store_true")
    run_parser.add_argument("--execution-mode", choices=["mock", "real"])
    run_parser.add_argument("--continue-on-error", action="store_true")
    run_parser.add_argument("--max-failures", type=int)
    run_parser.add_argument("--max-failure-rate", type=float)
    run_parser.add_argument("--output", choices=["text", "json"], default="text")
    run_parser.set_defaults(handler=handle_run)

    version_parser = subparsers.add_parser("version", help="Show framework version.")
    version_parser.set_defaults(handler=handle_version)

    return parser


def _add_preview_arguments(
    parser: argparse.ArgumentParser,
    *,
    stage_choices: list[str],
) -> None:
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--preview-stage", choices=stage_choices, default="all")
    parser.add_argument("--preview-format", choices=["text", "json", "md"], default="text")


def handle_version(_args: argparse.Namespace) -> int:
    print(f"whitzard {__version__}")
    return 0


def handle_models_list(args: argparse.Namespace) -> int:
    registry = load_registry()
    models = registry.list_models()
    if args.modality:
        models = [model for model in models if model.modality == args.modality]
    if args.task_type:
        models = [model for model in models if model.task_type == args.task_type]

    if args.output == "json":
        print(
            json.dumps(
                [_redacted_model_payload(model) for model in models],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    header = f"{'MODEL':<32} {'MODALITY':<8} {'TASK_TYPE':<9} {'EXECUTION_MODE'}"
    print(header)
    for model in models:
        print(
            f"{model.name:<32} {model.modality:<8} {model.task_type:<9} {model.execution_mode}"
        )
    return 0


def handle_models_inspect(args: argparse.Namespace) -> int:
    registry = load_registry()
    model = registry.get_model(args.model_name)
    adapter_class = registry.resolve_adapter_class(args.model_name)

    if args.output == "json":
        payload = _redacted_model_payload(model)
        payload["adapter_class"] = adapter_class.__name__
        payload["conda_env_name"] = model.conda_env_name
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Model: {model.name}")
    print(f"Version: {model.version}")
    print(f"Modality: {model.modality}")
    print(f"Task Type: {model.task_type}")
    print(f"Adapter: {model.adapter}")
    print(f"Adapter Class: {adapter_class.__name__}")
    print(f"Execution Mode: {model.execution_mode}")
    print(
        f"Batch Support: {'yes' if model.capabilities.get('supports_batch_prompts') else 'no'}"
    )
    print(f"Max Batch Size: {model.capabilities.get('max_batch_size', 1)}")
    print(f"Environment Spec: {model.env_spec}")
    print(f"Conda Env: {model.conda_env_name}")
    if model.generation_defaults:
        print("Generation Defaults:")
        for key, value in sorted(model.generation_defaults.items()):
            print(f"  {key}: {value}")
    hf_repo = model.weights.get("hf_repo", "-")
    print(f"HF Repo: {hf_repo}")
    if model.provider:
        print("Provider:")
        for key, value in sorted(_redact_provider_config(model.provider).items()):
            print(f"  {key}: {value}")
    print(f"Local Override File: {model.local_override_source or DEFAULT_LOCAL_MODELS_PATH}")
    if model.has_local_overrides:
        print("Effective Local Overrides:")
        for line in summarize_local_path_overrides(model.local_paths):
            print(f"  {line}")
    else:
        print("Effective Local Overrides: none")
    return 0


def handle_models_canary(args: argparse.Namespace) -> int:
    execution_mode = "mock" if args.mock else (args.execution_mode or "real")
    progress = build_run_progress(output_mode=args.output)
    summary = run_model_canary(
        model_name=args.model_name,
        prompt_file=args.prompt_file,
        out_dir=args.out,
        execution_mode=execution_mode,
        progress=progress,
    )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Canary model: {args.model_name}")
        print(f"Canary status: {summary.status}")
        print(f"Run ID: {summary.run_id}")
        print(f"Prompt Source: {summary.prompt_file}")
        print(f"Output Dir: {summary.output_dir}")
    return 0


def handle_models_matrix(args: argparse.Namespace) -> int:
    rows = build_model_capability_rows()
    write_result = None
    if args.write_docs:
        docs_dir = Path(args.docs_dir) if args.docs_dir else None
        markdown_path = docs_dir / "model_capability_matrix.md" if docs_dir else None
        json_path = docs_dir / "model_capability_matrix.json" if docs_dir else None
        write_result = write_model_capability_docs(
            **(
                {"markdown_path": markdown_path, "json_path": json_path}
                if docs_dir is not None
                else {}
            )
        )

    if args.output == "json":
        payload: dict[str, object] = {"rows": rows}
        if write_result is not None:
            payload["written"] = write_result
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(render_model_capability_matrix_markdown(rows).rstrip())
    if write_result is not None:
        print("")
        print(f"Wrote: {write_result['markdown_path']}")
        print(f"Wrote: {write_result['json_path']}")
    return 0


def handle_doctor(args: argparse.Namespace) -> int:
    manager = EnvManager()
    records = manager.doctor(model_name=args.model)

    if args.output == "json":
        payload = {
            "conda_available": manager.conda_available(),
            "records": [record.to_dict() for record in records],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Conda: {'OK' if manager.conda_available() else 'MISSING'}")
    for record in records:
        print(f"Model: {record.model_name}")
        print(f"  Status: {record.state.upper()}")
        print(f"  Conda env: {record.conda_env_name}")
        print(f"  Env exists: {'yes' if record.exists else 'no'}")
        if record.path:
            print(f"  Env path: {record.path}")
        if record.last_validation:
            checked_at = record.last_validation.get("checked_at", "-")
            passed = bool(record.last_validation.get("passed"))
            print(f"  Validation: {'passed' if passed else 'failed'}")
            print(f"  Validation checked at: {checked_at}")
        else:
            print("  Validation: not run")
        if record.error:
            print(f"  Error: {record.error}")
        if record.path_checks:
            for field, info in sorted(record.path_checks.items()):
                print(f"  {field}: {info['value']}")
                print(f"  {field}_exists: {'yes' if info['exists'] else 'no'}")
        else:
            print("  local_paths: none configured")
        if getattr(record, "provider_checks", {}):
            for field, info in sorted(record.provider_checks.items()):
                print(f"  provider.{field}: {info.get('value')}")
                print(f"  provider.{field}_ok: {'yes' if info.get('ok') else 'no'}")
    return 0


def handle_prompts_generate(args: argparse.Namespace) -> int:
    execution_mode = "mock" if args.mock else (args.execution_mode or "real")
    progress = build_run_progress(output_mode=args.output)
    summary = generate_prompt_bundle(
        tree_path=args.tree,
        out_dir=args.out,
        llm_model=args.llm_model,
        execution_mode=execution_mode,
        seed=args.seed,
        count_config_path=args.count_config,
        profile_path=args.profile,
        template_name=args.template,
        style_family_name=args.style_family,
        target_model_name=args.target_model,
        intended_modality=args.intended_modality,
        progress=progress,
    )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"Prompt Bundle: {summary.bundle_id}")
    print(f"Prompt Count: {summary.prompt_count}")
    print(f"Execution Mode: {summary.execution_mode}")
    print(f"LLM Model: {summary.llm_model or '-'}")
    print(f"Template: {summary.prompt_template or '-'}")
    print(f"Style Family: {summary.prompt_style_family or '-'}")
    print(f"Target Model: {summary.target_model_name or '-'}")
    print(f"Few-shot Examples: {summary.few_shot_example_count}")
    print(f"Bundle Dir: {summary.bundle_dir}")
    print(f"Prompts: {summary.prompts_path}")
    print(f"Manifest: {summary.manifest_path}")
    return 0


def handle_prompts_inspect(args: argparse.Namespace) -> int:
    payload = inspect_prompt_bundle(args.path)
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(f"Prompt Count: {payload['prompt_count']}")
    if payload.get("bundle_dir"):
        print(f"Bundle Dir: {payload['bundle_dir']}")
    manifest = payload.get("manifest") or {}
    if manifest:
        print(f"Bundle ID: {manifest.get('bundle_id', '-')}")
        print(f"Tree Name: {manifest.get('tree_name', '-')}")
        print(f"Generation Profile: {manifest.get('generation_profile', '-')}")
        print(f"LLM Model: {manifest.get('llm_model', '-')}")
        print(f"Template: {manifest.get('prompt_template', '-')}")
        print(f"Style Family: {manifest.get('prompt_style_family', '-')}")
        print(f"Target Model: {manifest.get('target_model_name', '-')}")
        print(f"Few-shot Example Count: {manifest.get('few_shot_example_count', 0)}")
    print(f"Counts By Category: {payload.get('counts_by_category', {})}")
    return 0


def handle_prompts_plan(args: argparse.Namespace) -> int:
    payload = plan_theme_tree(
        tree_path=args.tree,
        seed=args.seed,
        count_config_path=args.count_config,
    )
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(f"Tree: {payload['tree']['name']}")
    print(f"Planned Samples: {payload['sample_count']}")
    print(f"Resampled: {payload['resampled_count']}")
    print(f"Counts By Category: {payload['counts_by_category']}")
    return 0


def handle_benchmark_list(args: argparse.Namespace) -> int:
    payload = list_benchmark_builders()
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    for item in payload:
        print(f"{item['builder']}: {item['description']}")
    return 0


def handle_benchmark_build(args: argparse.Namespace) -> int:
    execution_mode = "mock" if getattr(args, "mock", False) else (args.execution_mode or "real")
    progress = build_run_progress(output_mode=args.output)
    source_path = args.source or args.package
    summary = build_benchmark(
        builder_name=args.builder,
        source_path=source_path,
        out_dir=args.out,
        benchmark_name=args.benchmark_name,
        seed=args.seed,
        build_mode=args.build_mode,
        builder_config_path=args.builder_config,
        count_config_path=args.count_config,
        llm_model=args.llm_model,
        synthesis_model=getattr(args, "synthesis_model", None),
        execution_mode=execution_mode,
        profile_path=args.profile,
        template_name=args.template,
        style_family_name=args.style_family,
        target_model_name=args.target_model,
        intended_modality=args.intended_modality,
        entrypoint=args.entrypoint,
        preview_enabled=bool(getattr(args, "preview", False) or getattr(args, "preview_only", False)),
        preview_only=bool(getattr(args, "preview_only", False)),
        preview_count=int(getattr(args, "preview_count", 5)),
        preview_stage=str(getattr(args, "preview_stage", "all")),
        preview_format=str(getattr(args, "preview_format", "text")),
        progress=progress,
    )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if _is_preview_summary(summary):
        _print_preview_summary(summary)
        return 0
    print(f"Benchmark: {summary.benchmark_id}")
    print(f"Builder: {summary.builder_name}")
    print(f"Source: {summary.source_path}")
    print(f"Cases: {summary.case_count}")
    print(f"Build Mode: {summary.build_mode}")
    print(f"Bundle Dir: {summary.benchmark_dir}")
    print(f"Cases Path: {summary.cases_path}")
    print(f"Manifest: {summary.manifest_path}")
    if summary.raw_realizations_path:
        print(f"Raw Realizations: {summary.raw_realizations_path}")
    if summary.rejected_realizations_path:
        print(f"Rejected Realizations: {summary.rejected_realizations_path}")
    _print_summary_preview_artifacts(summary)
    print("Next Evaluate: whitzard evaluate run --benchmark " f"{summary.benchmark_dir} --targets <MODEL_NAME>")
    return 0


def handle_benchmark_preview(args: argparse.Namespace) -> int:
    args.preview = True
    args.preview_only = True
    return handle_benchmark_build(args)


def handle_benchmark_inspect(args: argparse.Namespace) -> int:
    payload = inspect_benchmark_bundle(args.path)
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(f"Case Count: {payload['case_count']}")
    print(f"Benchmark Dir: {payload.get('benchmark_dir', '-')}")
    manifest = payload.get("manifest") or {}
    if manifest:
        print(f"Benchmark ID: {manifest.get('benchmark_id', '-')}")
        print(f"Builder: {manifest.get('builder_name', '-')}")
        print(f"Build Mode: {manifest.get('build_mode', '-')}")
        print(f"Source: {manifest.get('source_path', '-')}")
    print(f"Counts By Builder: {payload.get('counts_by_builder', {})}")
    print(f"Counts By Family: {payload.get('counts_by_family', {})}")
    print(f"Counts By Split: {payload.get('counts_by_split', {})}")
    if payload.get("raw_realizations_path"):
        print(f"Raw Realizations: {payload['raw_realizations_path']}")
    if payload.get("rejected_realizations_path"):
        print(f"Rejected Realizations: {payload['rejected_realizations_path']}")
    if payload.get("request_previews_path"):
        print(f"Request Previews: {payload['request_previews_path']}")
    if payload.get("request_preview_summary_path"):
        print(f"Preview Summary: {payload['request_preview_summary_path']}")
    if payload.get("request_previews_markdown_path"):
        print(f"Preview Markdown: {payload['request_previews_markdown_path']}")
    return 0


def handle_evaluate_run(args: argparse.Namespace) -> int:
    recipe = load_experiment_recipe(args.recipe) if getattr(args, "recipe", None) else {}
    benchmark_path = args.benchmark or _resolve_recipe_text(recipe.get("benchmark", {}), "path")
    if benchmark_path in (None, ""):
        benchmark_builder = recipe.get("benchmark", {})
        if isinstance(benchmark_builder, dict) and benchmark_builder.get("builder"):
            build_summary = build_benchmark(
                builder_name=str(benchmark_builder["builder"]),
                source_path=_resolve_recipe_relative_path(args.recipe, benchmark_builder.get("source")),
                out_dir=_resolve_recipe_relative_path(args.recipe, benchmark_builder.get("out")),
                benchmark_name=benchmark_builder.get("benchmark_name"),
                seed=int(benchmark_builder.get("seed", 42)),
                build_mode=str(benchmark_builder.get("build_mode", "static")),
                builder_config_path=_resolve_recipe_relative_path(args.recipe, benchmark_builder.get("config")),
                count_config_path=_resolve_recipe_relative_path(args.recipe, benchmark_builder.get("count_config")),
                llm_model=benchmark_builder.get("llm_model"),
                synthesis_model=benchmark_builder.get("synthesis_model"),
                execution_mode=str(benchmark_builder.get("execution_mode", "real")),
                profile_path=_resolve_recipe_relative_path(args.recipe, benchmark_builder.get("profile")),
                template_name=benchmark_builder.get("template"),
                style_family_name=benchmark_builder.get("style_family"),
                target_model_name=benchmark_builder.get("target_model"),
                intended_modality=benchmark_builder.get("intended_modality"),
                entrypoint=benchmark_builder.get("entrypoint"),
                progress=build_run_progress(output_mode=args.output),
            )
            benchmark_path = build_summary.benchmark_dir
    targets = list(args.targets or recipe.get("targets") or [])
    if not benchmark_path:
        raise BenchmarkingError("evaluate run requires --benchmark or a recipe benchmark.path/builder.")
    if not targets:
        raise BenchmarkingError("evaluate run requires --targets or a recipe targets list.")
    execution_mode = "mock" if args.mock else (args.execution_mode or recipe.get("execution_mode") or "real")
    progress = build_run_progress(output_mode=args.output)
    summary = evaluate_benchmark(
        benchmark_path=benchmark_path,
        target_models=targets,
        normalizer_ids=list(args.normalizers or recipe.get("normalizers") or []),
        evaluator_ids=list(args.evaluators or []),
        analysis_plugin_ids=list(args.analysis_plugins or recipe.get("analysis_plugins") or []),
        evaluator_model=args.evaluator_model,
        evaluator_profile=args.evaluator_profile,
        evaluator_template=args.evaluator_template,
        out_dir=args.out or _resolve_recipe_relative_path(args.recipe, recipe.get("out")),
        execution_mode=execution_mode,
        progress=progress,
        normalizer_config_path=args.normalizer_config or _resolve_recipe_relative_path(args.recipe, recipe.get("normalizer_config")),
        evaluator_config_path=args.evaluator_config,
        analysis_config_path=args.analysis_config or _resolve_recipe_relative_path(args.recipe, recipe.get("analysis_config")),
        recipe_path=args.recipe,
        auto_launch=bool(args.auto_launch or recipe.get("auto_launch", False)),
        launcher_config_path=args.launcher_config or _resolve_recipe_relative_path(args.recipe, recipe.get("launcher_config")),
        execution_policy=dict(recipe.get("execution_policy") or {}),
        preview_enabled=bool(getattr(args, "preview", False) or getattr(args, "preview_only", False)),
        preview_only=bool(getattr(args, "preview_only", False)),
        preview_count=int(getattr(args, "preview_count", 5)),
        preview_stage=str(getattr(args, "preview_stage", "all")),
        preview_format=str(getattr(args, "preview_format", "text")),
    )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if _is_preview_summary(summary):
        _print_preview_summary(summary)
        return 0
    print(f"Experiment: {summary.experiment_id}")
    print(f"Benchmark: {summary.benchmark_id}")
    print(f"Targets: {', '.join(summary.target_models)}")
    print(f"Normalizers: {', '.join(summary.normalizer_ids) or '-'}")
    print(f"Evaluators: {', '.join(summary.evaluator_ids) or '-'}")
    print(f"Analysis Plugins: {', '.join(summary.analysis_plugin_ids) or '-'}")
    print(f"Execution Mode: {summary.execution_mode}")
    print(f"Cases: {summary.case_count}")
    print(f"Target Runs: {summary.target_run_count}")
    print(f"Normalized Results: {summary.normalized_result_count}")
    print(f"Evaluator Results: {summary.evaluator_result_count}")
    print(f"Group Analyses: {summary.group_analysis_count}")
    print(f"Analysis Plugin Results: {summary.analysis_plugin_result_count}")
    print(f"Experiment Dir: {summary.experiment_dir}")
    print(f"Report: {summary.report_path}")
    _print_summary_preview_artifacts(summary)
    return 0


def handle_evaluate_preview(args: argparse.Namespace) -> int:
    args.preview = True
    args.preview_only = True
    return handle_evaluate_run(args)


def handle_evaluate_inspect(args: argparse.Namespace) -> int:
    payload = inspect_experiment(args.experiment)
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    manifest = payload.get("manifest") or {}
    summary = payload.get("summary") or {}
    scorer_ids = manifest.get("scorer_ids", manifest.get("evaluator_ids", []))
    score_record_count = summary.get("score_record_count", summary.get("evaluator_result_count", 0))
    group_analysis_count = summary.get("group_analysis_record_count", summary.get("group_analysis_count", 0))
    print(f"Experiment: {manifest.get('experiment_id', '-')}")
    print(f"Benchmark: {manifest.get('benchmark_id', '-')}")
    print(f"Targets: {', '.join(manifest.get('target_models', []))}")
    print(f"Normalizers: {', '.join(manifest.get('normalizer_ids', []))}")
    print(f"Scorers: {', '.join(scorer_ids)}")
    print(f"Analysis Plugins: {', '.join(manifest.get('analysis_plugin_ids', []))}")
    print(f"Case Count: {manifest.get('case_count', '-')}")
    print(f"Target Result Count: {summary.get('target_result_count', 0)}")
    print(f"Normalized Result Count: {summary.get('normalized_result_count', 0)}")
    print(f"Score Record Count: {score_record_count}")
    print(f"Group Analysis Count: {group_analysis_count}")
    print(f"Analysis Plugin Result Count: {summary.get('analysis_plugin_result_count', 0)}")
    if payload.get("report_path"):
        print(f"Report: {payload['report_path']}")
    if payload.get("request_previews_path"):
        print(f"Request Previews: {payload['request_previews_path']}")
    if payload.get("request_preview_summary_path"):
        print(f"Preview Summary: {payload['request_preview_summary_path']}")
    if payload.get("request_previews_markdown_path"):
        print(f"Preview Markdown: {payload['request_previews_markdown_path']}")
    return 0


def handle_experiments_list(args: argparse.Namespace) -> int:
    manifests = list_experiments()
    if args.output == "json":
        print(json.dumps(manifests, indent=2, ensure_ascii=False))
        return 0
    if not manifests:
        print("No experiments found.")
        return 0
    header = f"{'EXPERIMENT_ID':<36} {'BENCHMARK':<24} {'TARGETS'}"
    print(header)
    for manifest in manifests:
        print(
            f"{manifest.get('experiment_id', '-'):<36} "
            f"{manifest.get('benchmark_id', '-'):<24} "
            f"{','.join(manifest.get('target_models', []))}"
        )
    return 0


def handle_experiments_report(args: argparse.Namespace) -> int:
    payload = inspect_experiment(args.experiment)
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    manifest = payload.get("manifest") or {}
    summary = payload.get("summary") or {}
    scorer_ids = manifest.get("scorer_ids", manifest.get("evaluator_ids", []))
    score_record_count = summary.get("score_record_count", summary.get("evaluator_result_count", 0))
    group_analysis_count = summary.get("group_analysis_record_count", summary.get("group_analysis_count", 0))
    print(f"Experiment: {manifest.get('experiment_id', '-')}")
    print(f"Benchmark: {manifest.get('benchmark_id', '-')}")
    print(f"Targets: {', '.join(manifest.get('target_models', []))}")
    print(f"Normalizers: {', '.join(manifest.get('normalizer_ids', []))}")
    print(f"Scorers: {', '.join(scorer_ids)}")
    print(f"Analysis Plugins: {', '.join(manifest.get('analysis_plugin_ids', []))}")
    print(f"Case Count: {manifest.get('case_count', '-')}")
    print(f"Target Result Count: {summary.get('target_result_count', 0)}")
    print(f"Normalized Result Count: {summary.get('normalized_result_count', 0)}")
    print(f"Score Record Count: {score_record_count}")
    print(f"Group Analysis Count: {group_analysis_count}")
    print(f"Analysis Plugin Result Count: {summary.get('analysis_plugin_result_count', 0)}")
    if payload.get("report_path"):
        print(f"Report: {payload['report_path']}")
    if payload.get("request_previews_path"):
        print(f"Request Previews: {payload['request_previews_path']}")
    if payload.get("request_preview_summary_path"):
        print(f"Preview Summary: {payload['request_preview_summary_path']}")
    if payload.get("request_previews_markdown_path"):
        print(f"Preview Markdown: {payload['request_previews_markdown_path']}")
    return 0


def handle_runs_list(args: argparse.Namespace) -> int:
    manifests = list_runs()
    if args.output == "json":
        print(json.dumps(manifests, indent=2, ensure_ascii=False))
        return 0

    if not manifests:
        print(f"No runs found under {get_runs_root()}")
        return 0

    header = f"{'RUN_ID':<28} {'STATUS':<10} {'MODE':<8} {'MODELS':<32} {'RECORDS'}"
    print(header)
    for manifest in manifests:
        models = ",".join(manifest.get("models", []))
        print(
            f"{manifest.get('run_id', '-'): <28} "
            f"{manifest.get('status', '-'): <10} "
            f"{manifest.get('execution_mode', '-'): <8} "
            f"{models[:32]:<32} "
            f"{manifest.get('records_exported', 0)}"
        )
    return 0


def handle_runs_inspect(args: argparse.Namespace) -> int:
    manifest = load_run_manifest(args.run_id)
    if args.output == "json":
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    print(f"Run: {manifest.get('run_id', args.run_id)}")
    print(f"Status: {manifest.get('status', '-')}")
    print(f"Created At: {manifest.get('created_at', '-')}")
    print(f"Execution Mode: {manifest.get('execution_mode', '-')}")
    print(f"Prompt Source: {manifest.get('prompt_source', '-')}")
    print(f"Prompt Count: {manifest.get('prompt_count', '-')}")
    print(f"Task Count: {manifest.get('task_count', '-')}")
    print(f"Models: {', '.join(manifest.get('models', []))}")
    print(f"Output Dir: {manifest.get('output_dir', '-')}")
    export_paths = manifest.get("export_paths", {})
    if export_paths:
        for name, path in sorted(export_paths.items()):
            print(f"{name}: {path}")
    return 0


def handle_runs_failures(args: argparse.Namespace) -> int:
    failures = load_failures_summary(args.run_id)
    if args.output == "json":
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        return 0

    if not failures:
        print(f"No failures recorded for run {args.run_id}")
        return 0
    for failure in failures:
        print(
            f"{failure.get('task_id', '-')}: {failure.get('model_name', '-')} - "
            f"{failure.get('error', '-')}"
        )
    return 0


def handle_runs_retry(args: argparse.Namespace) -> int:
    plan = build_retry_plan(args.run_id, model_name=args.model)
    if plan.selected_count == 0:
        if args.output == "json":
            print(json.dumps({"run_id": args.run_id, "recovery_mode": "retry", "selected_count": 0}, indent=2, ensure_ascii=False))
        else:
            print(f"Inspecting run {args.run_id}...")
            print("Found 0 failed prompt outputs.")
            print("Nothing to retry.")
        return 0

    if args.output != "json":
        print(f"Inspecting run {args.run_id}...")
        print(f"Found {plan.failed_count} failed prompt outputs.")
        print(f"Retrying {plan.selected_count} prompt outputs in a new run...")
    progress = build_run_progress(output_mode=args.output)
    summary = run_recovery_plan(
        recovery_plan=plan,
        run_name=f"retry-{args.run_id}",
        progress=progress,
    )
    if args.output == "json":
        print(
            json.dumps(
                {
                    "plan": recovery_plan_to_dict(plan),
                    "summary": summary.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


def handle_runs_resume(args: argparse.Namespace) -> int:
    plan = build_resume_plan(args.run_id, model_name=args.model)
    if plan.selected_count == 0:
        if args.output == "json":
            print(json.dumps({"run_id": args.run_id, "recovery_mode": "resume", "selected_count": 0}, indent=2, ensure_ascii=False))
        else:
            print(f"Inspecting run {args.run_id}...")
            print(
                f"Found {plan.completed_count} completed outputs, {plan.missing_count} missing outputs."
            )
            print("Nothing to resume.")
        return 0

    if args.output != "json":
        print(f"Inspecting run {args.run_id}...")
        print(
            f"Found {plan.completed_count} completed outputs, {plan.missing_count} missing outputs."
        )
        print(f"Resuming {plan.selected_count} prompt outputs in a new run...")
    progress = build_run_progress(output_mode=args.output)
    summary = run_recovery_plan(
        recovery_plan=plan,
        run_name=f"resume-{args.run_id}",
        progress=progress,
    )
    if args.output == "json":
        print(
            json.dumps(
                {
                    "plan": recovery_plan_to_dict(plan),
                    "summary": summary.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


def handle_export_dataset(args: argparse.Namespace) -> int:
    export_result = export_dataset_for_runs(
        list(args.run_ids),
        output_path=args.out,
        mode=args.mode,
        selected_models=list(args.model or []),
    )
    payload = export_result.to_dict()
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Runs: {', '.join(export_result.source_run_ids)}")
    print(f"Bundle: {export_result.bundle_path}")
    print(f"Dataset: {export_result.dataset_path}")
    print(f"Manifest: {export_result.manifest_path}")
    print(f"README: {export_result.readme_path}")
    print(f"Mode: {export_result.export_mode}")
    print(f"Records: {export_result.record_count}")
    print(f"Skipped: {export_result.skipped_count}")
    if export_result.filtered_out_count:
        print(f"Filtered Out: {export_result.filtered_out_count}")
    if export_result.selected_models:
        print(f"Selected Models: {', '.join(export_result.selected_models)}")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    profile = load_run_profile(args.profile) if args.profile else None
    request = resolve_profile_run_request(
        profile=profile,
        models_arg=args.models,
        prompts_arg=args.prompts,
        execution_mode_arg=args.execution_mode,
        mock_flag=bool(args.mock),
        out_arg=args.out,
        run_name_arg=args.run_name,
    )
    registry = load_registry()
    if not request["model_names"]:
        raise RunFlowError("whitzard run requires at least one model.")
    models = [registry.get_model(model_name) for model_name in request["model_names"]]
    modality = models[0].modality
    invalid_models = [model.name for model in models if model.modality != modality]
    if invalid_models:
        raise RunFlowError(
            "whitzard run requires all selected models to share one modality. "
            f"Expected {modality}, got mismatched models: {', '.join(invalid_models)}."
        )

    progress = build_run_progress(output_mode=args.output)
    with apply_profile_runtime_environment(profile):
        summary = run_models(
            model_names=request["model_names"],
            prompt_file=request["prompt_file"],
            out_dir=request["out_dir"],
            run_name=request["run_name"],
            execution_mode=request["execution_mode"],
            mock_mode=bool(request["mock_mode"]),
            progress=progress,
            profile_name=request["profile_name"],
            profile_path=request["profile_path"],
            profile_generation_defaults=request["generation_defaults"],
            profile_runtime=request["runtime"],
            profile_global_negative_prompt=request["global_negative_prompt"],
            profile_conditionings=request["conditionings"],
            profile_prompt_rewrites=request["prompt_rewrites"],
            continue_on_error=bool(args.continue_on_error) or None,
            max_failures=args.max_failures,
            max_failure_rate=args.max_failure_rate,
        )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return 0

    return 0


def handle_annotate(args: argparse.Namespace) -> int:
    progress = build_run_progress(output_mode=args.output)
    execution_mode = "mock" if args.mock else (args.execution_mode or "real")
    summary = annotate_run(
        args.run_id,
        annotation_profile=args.profile,
        annotator_model=args.model,
        template_name=args.template,
        out_dir=args.out,
        execution_mode=execution_mode,
        progress=progress,
    )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"Bundle: {summary.bundle_dir}")
    print(f"Source Run: {summary.source_run_id}")
    print(f"Annotator Model: {summary.annotator_model}")
    print(f"Profile: {summary.annotation_profile}")
    print(f"Template: {summary.annotation_template}")
    print(f"Annotated: {summary.annotated_count}")
    print(f"Skipped: {summary.skipped_count}")
    print(f"Failed: {summary.failed_count}")
    print(f"Annotations: {summary.annotations_path}")
    print(f"Manifest: {summary.manifest_path}")
    return 0


def _is_preview_summary(summary: object) -> bool:
    return hasattr(summary, "request_previews_path") and hasattr(summary, "preview_only")


def _print_summary_preview_artifacts(summary: object) -> None:
    request_previews_path = getattr(summary, "request_previews_path", None)
    request_preview_summary_path = getattr(summary, "request_preview_summary_path", None)
    request_previews_markdown_path = getattr(summary, "request_previews_markdown_path", None)
    if request_previews_path:
        print(f"Request Previews: {request_previews_path}")
    if request_preview_summary_path:
        print(f"Preview Summary: {request_preview_summary_path}")
    if request_previews_markdown_path:
        print(f"Preview Markdown: {request_previews_markdown_path}")
    if request_preview_summary_path:
        _print_preview_samples_from_path(request_preview_summary_path)


def _print_preview_summary(summary: object) -> None:
    print(f"Preview Dir: {getattr(summary, 'preview_dir', '-')}")
    print(f"Preview Only: {'yes' if bool(getattr(summary, 'preview_only', False)) else 'no'}")
    print(f"Preview Stage: {getattr(summary, 'preview_stage', '-')}")
    print(f"Preview Count: {getattr(summary, 'preview_count', 0)}")
    print(f"Request Previews: {getattr(summary, 'request_previews_path', '-')}")
    print(f"Preview Summary: {getattr(summary, 'request_preview_summary_path', '-')}")
    markdown_path = getattr(summary, "request_previews_markdown_path", None)
    if markdown_path:
        print(f"Preview Markdown: {markdown_path}")
    counts_by_stage = dict(getattr(summary, "counts_by_stage", {}) or {})
    if counts_by_stage:
        print(f"Counts By Stage: {counts_by_stage}")
    sample_records = list(getattr(summary, "sample_records", []) or [])
    if sample_records:
        _print_preview_samples(sample_records)


def _print_preview_samples_from_path(path: str) -> None:
    preview_summary_path = Path(path)
    if not preview_summary_path.exists():
        return
    payload = json.loads(preview_summary_path.read_text(encoding="utf-8"))
    sample_records = list(payload.get("sample_records", []) or [])
    if sample_records:
        _print_preview_samples(sample_records)


def _print_preview_samples(sample_records: list[dict[str, object]]) -> None:
    print("Preview Samples:")
    for record in sample_records:
        stage = str(record.get("stage", "unknown"))
        entity_id = str(record.get("entity_id", ""))
        case_id = str(record.get("case_id", "") or "-")
        request_id = str(record.get("request_id", "") or "-")
        template_name = str(record.get("template_name", "") or "-")
        print(f"  [{stage}] case={case_id} request={request_id} entity={entity_id} template={template_name}")
        rendered_prompt = str(record.get("rendered_prompt", "") or "").strip()
        if rendered_prompt:
            snippet = rendered_prompt if len(rendered_prompt) <= 400 else rendered_prompt[:400] + "..."
            print(snippet)
            print("")


def _resolve_recipe_text(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _resolve_recipe_relative_path(recipe_path: str | None, value: object) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute() or recipe_path in (None, ""):
        return str(path)
    return str((Path(recipe_path).resolve().parent / path).resolve())


def _redacted_model_payload(model) -> dict[str, object]:
    payload = model.to_dict()
    payload["provider"] = _redact_provider_config(payload.get("provider"))
    local_paths = payload.get("local_paths")
    if isinstance(local_paths, dict) and "provider" in local_paths:
        local_paths = dict(local_paths)
        local_paths["provider"] = _redact_provider_config(local_paths.get("provider"))
        payload["local_paths"] = local_paths
    return payload


def _redact_provider_config(provider: object) -> dict[str, object]:
    if not isinstance(provider, dict):
        return {}
    payload = dict(provider)
    default_headers = payload.get("default_headers")
    if isinstance(default_headers, dict):
        payload["default_headers"] = {
            str(key): "<redacted>" for key in default_headers
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        try:
            return args.handler(args)
        except (
            AnalysisConfigError,
            AnalysisError,
            AnnotationConfigError,
            AnnotationError,
            BenchmarkingError,
            EvaluatorConfigError,
            EvaluatorError,
            ExperimentRecipeError,
            EnvManagerError,
            LaunchError,
            NormalizerConfigError,
            NormalizerError,
            PromptGenerationError,
            RegistryError,
            RunFlowError,
            RunStoreError,
            RecoveryError,
            RunProfileError,
        ) as exc:
            parser.exit(status=1, message=f"{exc}\n")
    parser.print_help()
    return 0
