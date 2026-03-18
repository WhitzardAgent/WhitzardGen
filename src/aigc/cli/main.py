import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aigc import __version__
from aigc.env import EnvManager, EnvManagerError, MissingEnvironmentError
from aigc.registry import DEFAULT_LOCAL_MODELS_PATH, RegistryError, load_registry
from aigc.registry.local_overrides import summarize_local_path_overrides
from aigc.run_store import (
    RunStoreError,
    export_dataset_for_run,
    list_runs,
    load_failures_summary,
    load_run_manifest,
)
from aigc.run_flow import RunFlowError, run_models
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
    run_parser.add_argument("--models", required=True)
    run_parser.add_argument("--prompts", required=True)
    run_parser.add_argument("--run-name")
    run_parser.add_argument("--out")
    run_parser.add_argument("--mock", action="store_true")
    run_parser.add_argument("--execution-mode", choices=["mock", "real"], default="real")
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
        status = record.state.upper()
        suffix = f" ({record.error})" if record.error else ""
        print(f"Model {record.model_name}: {status}{suffix}")
        print(f"  conda_env_name: {record.conda_env_name}")
        print(f"  env_exists: {record.exists}")
        if record.local_paths:
            for field, info in sorted(record.path_checks.items()):
                print(
                    f"  {field}: {info['value']} [{'OK' if info['exists'] else 'MISSING'}]"
                )
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
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    progress = build_run_progress(output_mode=args.output)
    summary = run_models(
        model_names=model_names,
        prompt_file=args.prompts,
        out_dir=args.out,
        run_name=args.run_name,
        execution_mode=args.execution_mode,
        mock_mode=bool(args.mock),
        progress=progress,
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
        except (EnvManagerError, RegistryError, RunFlowError, RunStoreError) as exc:
            parser.exit(status=1, message=f"{exc}\n")
    parser.print_help()
    return 0
