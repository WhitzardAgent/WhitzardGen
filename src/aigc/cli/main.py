import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aigc import __version__
from aigc.env import EnvManager, EnvManagerError
from aigc.recovery import (
    RecoveryError,
    build_resume_plan,
    build_retry_plan,
    recovery_plan_to_dict,
)
from aigc.run_profiles import (
    RunProfileError,
    apply_profile_runtime_environment,
    load_run_profile,
    resolve_profile_run_request,
)
from aigc.registry import DEFAULT_LOCAL_MODELS_PATH, RegistryError, load_registry
from aigc.registry.local_overrides import summarize_local_path_overrides
from aigc.run_store import (
    RunStoreError,
    export_dataset_for_run,
    list_runs,
    load_failures_summary,
    load_run_manifest,
)
from aigc.run_flow import RunFlowError, run_models, run_recovery_plan
from aigc.settings import get_runs_root
from aigc.utils.progress import build_run_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aigc")
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

    doctor_parser = subparsers.add_parser("doctor", help="Check environment readiness.")
    doctor_parser.add_argument("--model")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text")
    doctor_parser.set_defaults(handler=handle_doctor)

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
        "dataset", help="Locate or copy a run dataset export."
    )
    export_dataset_parser.add_argument("run_id")
    export_dataset_parser.add_argument("--out")
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
    run_parser.add_argument("--output", choices=["text", "json"], default="text")
    run_parser.set_defaults(handler=handle_run)

    version_parser = subparsers.add_parser("version", help="Show framework version.")
    version_parser.set_defaults(handler=handle_version)

    return parser


def handle_version(_args: argparse.Namespace) -> int:
    print(f"aigc {__version__}")
    return 0


def handle_models_list(args: argparse.Namespace) -> int:
    registry = load_registry()
    models = registry.list_models()
    if args.modality:
        models = [model for model in models if model.modality == args.modality]
    if args.task_type:
        models = [model for model in models if model.task_type == args.task_type]

    if args.output == "json":
        print(json.dumps([model.to_dict() for model in models], indent=2, ensure_ascii=False))
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
        payload = model.to_dict()
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
    hf_repo = model.weights.get("hf_repo", "-")
    print(f"HF Repo: {hf_repo}")
    print(f"Local Override File: {model.local_override_source or DEFAULT_LOCAL_MODELS_PATH}")
    if model.has_local_overrides:
        print("Effective Local Overrides:")
        for line in summarize_local_path_overrides(model.local_paths):
            print(f"  {line}")
    else:
        print("Effective Local Overrides: none")
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
    export_path = export_dataset_for_run(args.run_id, output_path=args.out)
    payload = {"run_id": args.run_id, "dataset_path": str(export_path)}
    if args.output == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Run: {args.run_id}")
    print(f"Dataset: {export_path}")
    if args.out:
        print("Action: copied")
    else:
        print("Action: located")
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
        raise RunFlowError("aigc run requires at least one model.")
    models = [registry.get_model(model_name) for model_name in request["model_names"]]
    modality = models[0].modality
    invalid_models = [model.name for model in models if model.modality != modality]
    if invalid_models:
        raise RunFlowError(
            "aigc run requires all selected models to share one modality. "
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
            profile_runtime=request["runtime"],
        )
    if args.output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return 0

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        try:
            return args.handler(args)
        except (
            EnvManagerError,
            RegistryError,
            RunFlowError,
            RunStoreError,
            RecoveryError,
            RunProfileError,
        ) as exc:
            parser.exit(status=1, message=f"{exc}\n")
    parser.print_help()
    return 0
