from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from aigc.adapters.base import ProgressCallback


def resolve_video_dimensions(
    params: dict[str, Any],
    *,
    default_width: int = 1280,
    default_height: int = 720,
) -> tuple[int, int]:
    resolution = params.get("resolution")
    if isinstance(resolution, str) and "x" in resolution:
        left, right = resolution.lower().split("x", maxsplit=1)
        return int(left), int(right)
    width = int(params.get("width", default_width))
    height = int(params.get("height", default_height))
    return width, height


def resolve_video_negative_prompts(
    *,
    prompts: list[str],
    params: dict[str, Any],
    supports_negative_prompt: bool,
) -> list[str]:
    if not supports_negative_prompt:
        return ["" for _ in prompts]
    negative_prompts = list(params.get("negative_prompts", []))
    if negative_prompts and len(negative_prompts) != len(prompts):
        raise ValueError("negative_prompts must match the prompt batch length.")
    if not negative_prompts:
        return ["" for _ in prompts]
    return [str(item) for item in negative_prompts]


def compute_duration_sec(*, num_frames: int, fps: int) -> float:
    safe_fps = max(fps, 1)
    return round(num_frames / safe_fps, 4)


def resolve_video_model_reference(model_config: Any) -> str:
    return str(
        model_config.weights.get("weights_path")
        or model_config.weights.get("local_path")
        or model_config.weights.get("diffusers_repo")
        or model_config.weights.get("hf_repo")
    )


def resolve_video_repo_dir(model_config: Any) -> str | None:
    repo_dir = model_config.weights.get("repo_path") or model_config.weights.get("script_root")
    if repo_dir in (None, ""):
        return None
    return str(repo_dir)


def validate_local_diffusers_reference(
    *,
    model_config: Any,
    model_ref: str,
    required_files: tuple[str, ...],
    adapter_specific_hint: str | None = None,
) -> None:
    configured_field = None
    configured_path = None
    for field in ("weights_path", "local_path"):
        raw_value = model_config.weights.get(field)
        if raw_value not in (None, ""):
            configured_field = field
            configured_path = Path(str(raw_value))
            break

    if configured_path is not None:
        if not configured_path.exists():
            raise RuntimeError(
                f"{model_config.name} configured {configured_field} does not exist: {configured_path}"
            )
        missing = [
            relative_path
            for relative_path in required_files
            if not (configured_path / relative_path).exists()
        ]
        if missing:
            message_lines = [
                f"{model_config.name} local weights path does not look like a Diffusers model directory: {configured_path}",
                f"Configured field: {configured_field}",
                f"Missing required files: {', '.join(missing)}",
            ]
            repo_path = model_config.weights.get("repo_path")
            if repo_path not in (None, ""):
                message_lines.append(f"Configured repo_path: {repo_path}")
            if adapter_specific_hint:
                message_lines.append(adapter_specific_hint)
            raise RuntimeError("\n".join(message_lines))

    candidate_path = Path(model_ref)
    if candidate_path.exists():
        missing = [
            relative_path
            for relative_path in required_files
            if not (candidate_path / relative_path).exists()
        ]
        if missing:
            raise RuntimeError(
                f"{model_config.name} local model reference is missing required Diffusers files "
                f"under {candidate_path}: {', '.join(missing)}"
            )


def validate_local_video_directory(
    *,
    model_name: str,
    configured_label: str,
    configured_path: str,
    required_entries: tuple[str, ...],
    repo_path: str | None = None,
    repo_hint: str | None = None,
) -> None:
    candidate_path = Path(configured_path)
    if candidate_path.exists():
        missing = [
            relative_path
            for relative_path in required_entries
            if not (candidate_path / relative_path).exists()
        ]
        if missing:
            lines = [
                f"{model_name} local checkpoint directory is missing required entries: {candidate_path}",
                f"Configured field: {configured_label}",
                f"Missing required entries: {', '.join(missing)}",
            ]
            if repo_path not in (None, ""):
                lines.append(f"Configured repo_path: {repo_path}")
            if repo_hint:
                lines.append(repo_hint)
            raise RuntimeError("\n".join(lines))
    if repo_path not in (None, "") and not Path(str(repo_path)).exists():
        lines = [f"{model_name} configured repo_path does not exist: {repo_path}"]
        if repo_hint:
            lines.append(repo_hint)
        raise RuntimeError("\n".join(lines))


def resolve_video_cache_dir(model_config: Any) -> str | None:
    cache_dir = model_config.weights.get("hf_cache_dir")
    if cache_dir in (None, ""):
        return None
    return str(cache_dir)


def metadata_sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f"{target.stem}.metadata.json")


def build_diffusers_progress_kwargs(
    *,
    pipe: Any,
    total_steps: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    if progress_callback is None:
        return {}
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return {}

    def legacy_callback(step_index: int, _timestep: Any, _latents: Any) -> None:
        progress_callback(
            {
                "phase": "generating",
                "current_step": int(step_index) + 1,
                "total_steps": int(total_steps),
                "supports_true_progress": True,
            }
        )

    def callback_on_step_end(
        _pipe: Any,
        step_index: int,
        _timestep: Any,
        callback_kwargs: dict[str, Any],
    ):
        progress_callback(
            {
                "phase": "generating",
                "current_step": int(step_index) + 1,
                "total_steps": int(total_steps),
                "supports_true_progress": True,
            }
        )
        return callback_kwargs

    parameters = signature.parameters
    if "callback_on_step_end" in parameters:
        kwargs: dict[str, Any] = {"callback_on_step_end": callback_on_step_end}
        if "callback_on_step_end_tensor_inputs" in parameters:
            kwargs["callback_on_step_end_tensor_inputs"] = []
        return kwargs
    if "callback" in parameters:
        kwargs = {"callback": legacy_callback}
        if "callback_steps" in parameters:
            kwargs["callback_steps"] = 1
        return kwargs
    return {}


def normalize_frame_batches(model_name: str, frames: Any) -> list[list[Any]]:
    if frames is None:
        raise RuntimeError(f"{model_name} did not return video frames.")
    try:
        frame_batch_count = len(frames)
    except TypeError as exc:
        raise RuntimeError(f"{model_name} returned invalid video frames payload.") from exc
    if frame_batch_count == 0:
        raise RuntimeError(f"{model_name} did not return video frames.")
    return [list(video_frames) for video_frames in frames]


def normalize_single_video_frames(model_name: str, frames: Any) -> list[Any]:
    if frames is None:
        raise RuntimeError(f"{model_name} did not return video frames.")
    try:
        frame_count = len(frames)
    except TypeError as exc:
        raise RuntimeError(f"{model_name} returned invalid single-video frames payload.") from exc
    if frame_count == 0:
        raise RuntimeError(f"{model_name} did not return video frames.")
    return list(frames)


@contextlib.contextmanager
def temporary_repo_import_path(repo_dir: str | None):
    if not repo_dir:
        yield
        return
    repo_path = str(Path(repo_dir).resolve())
    inserted = repo_path not in sys.path
    if inserted:
        sys.path.insert(0, repo_path)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(repo_path)
            except ValueError:
                pass


def torch_gc(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return
    try:
        if cuda.is_available() and hasattr(cuda, "empty_cache"):
            cuda.empty_cache()
    except Exception:
        return


def extract_video_metadata(path: str | Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    sidecar_path = metadata_sidecar_path(path)
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    if fallback is not None:
        payload = dict(fallback)
        payload.setdefault("format", Path(path).suffix.lstrip(".").lower() or "mp4")
        return payload
    raise FileNotFoundError(f"Video metadata sidecar missing for {path}")


def recover_single_video_output(
    *,
    output_path: str | Path,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
) -> dict[str, Any]:
    return {
        "path": str(output_path),
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "duration_sec": compute_duration_sec(num_frames=num_frames, fps=fps),
        "format": Path(output_path).suffix.lstrip(".").lower() or "mp4",
    }


def mock_fingerprint(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def write_mock_mp4(path: str | Path, *, metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_mock_mp4(metadata.get("fingerprint", "mock-video")))
    payload = dict(metadata)
    payload.setdefault("format", output_path.suffix.lstrip(".").lower() or "mp4")
    metadata_sidecar_path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def build_mock_mp4(fingerprint: str) -> bytes:
    marker = fingerprint.encode("ascii", errors="ignore")[:16].ljust(16, b"0")
    return b"FAKE_MP4" + marker
