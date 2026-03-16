from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aigc.adapters.base import ExecutionResult
from aigc.registry import load_registry
from aigc.runtime.payloads import TaskPayload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aigc-worker")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--registry-file")
    return parser


def execute_task_payload(payload: TaskPayload, registry_path: str | None = None) -> dict:
    workdir = Path(payload.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(registry_path) if registry_path else load_registry()
    model = registry.get_model(payload.model_name)
    adapter = registry.resolve_adapter_class(payload.model_name)(model_config=model)

    prompts = [item.prompt for item in payload.prompts]
    prompt_ids = [item.prompt_id for item in payload.prompts]
    prepare_params = dict(payload.params)
    runtime = dict(payload.runtime_config)
    runtime["execution_mode"] = payload.execution_mode
    prepare_params["_runtime_config"] = runtime
    plan = adapter.prepare(
        prompts=prompts,
        prompt_ids=prompt_ids,
        params=prepare_params,
        workdir=str(workdir),
    )
    plan.inputs["batch_id"] = payload.batch_id
    plan_runtime = dict(plan.inputs.get("runtime", {}))
    plan_runtime.update(runtime)
    plan.inputs["runtime"] = plan_runtime

    if plan.mode == "external_process":
        if not plan.command:
            raise RuntimeError("External-process plan is missing command.")
        env = os.environ.copy()
        env.update(plan.env)
        result = subprocess.run(
            plan.command,
            cwd=plan.cwd or str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=plan.timeout_sec,
            check=False,
        )
        exec_result = ExecutionResult(
            exit_code=result.returncode,
            logs="\n".join(part for part in [result.stdout, result.stderr] if part),
        )
    else:
        exec_result = adapter.execute(
            plan=plan,
            prompts=prompts,
            params=payload.params,
            workdir=str(workdir),
        )

    model_result = adapter.collect(
        plan=plan,
        exec_result=exec_result,
        prompts=prompts,
        prompt_ids=prompt_ids,
        workdir=str(workdir),
    )
    return {
        "task_id": payload.task_id,
        "model_name": payload.model_name,
        "execution_mode": payload.execution_mode,
        "plan": plan.to_dict(),
        "execution_result": exec_result.to_dict(),
        "model_result": model_result.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task_payload = TaskPayload.from_dict(json.loads(Path(args.task_file).read_text(encoding="utf-8")))
    try:
        result = execute_task_payload(task_payload, registry_path=args.registry_file)
        status_code = 0
    except Exception as exc:  # pragma: no cover - exercised through integration boundaries
        result = {
            "task_id": task_payload.task_id,
            "model_name": task_payload.model_name,
            "execution_mode": task_payload.execution_mode,
            "plan": None,
            "execution_result": {"exit_code": 1, "logs": str(exc), "outputs": {}},
            "model_result": {
                "status": "failed",
                "batch_items": [],
                "logs": str(exc),
                "metadata": {},
            },
        }
        status_code = 1
    Path(args.result_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
