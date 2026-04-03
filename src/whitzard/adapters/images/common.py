from __future__ import annotations

import hashlib
import inspect
import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from whitzard.adapters.base import ProgressCallback

if TYPE_CHECKING:
    from whitzard.registry.models import ModelInfo


def resolve_image_dimensions(params: dict[str, Any]) -> tuple[int, int]:
    resolution = params.get("resolution")
    if isinstance(resolution, str) and "x" in resolution:
        left, right = resolution.lower().split("x", maxsplit=1)
        return int(left), int(right)
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    return width, height


def resolve_negative_prompts(
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


def resolve_model_reference(model_config: "ModelInfo") -> str:
    return str(
        model_config.weights.get("local_path")
        or model_config.weights.get("weights_path")
        or model_config.weights.get("hf_repo")
    )


def resolve_cache_dir(model_config: "ModelInfo") -> str | None:
    cache_dir = model_config.weights.get("hf_cache_dir")
    if cache_dir in (None, ""):
        return None
    return str(cache_dir)


def extract_png_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Unsupported image format for {path}; expected PNG.")
        _length = handle.read(4)
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"Invalid PNG header for {path}.")
        width, height = struct.unpack(">II", handle.read(8))
    return {"width": width, "height": height, "format": "png"}


def deterministic_color(*parts: str) -> tuple[int, int, int]:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return digest[0], digest[1], digest[2]


def write_mock_png(
    path: str | Path,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_mock_png(width=width, height=height, color=color))
    return output_path


def build_mock_png(*, width: int, height: int, color: tuple[int, int, int]) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("Mock PNG dimensions must be positive.")
    pixel = bytes(color)
    row = b"\x00" + pixel * width
    image_data = row * height
    compressed = zlib.compress(image_data, level=9)

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", ihdr),
            chunk(b"IDAT", compressed),
            chunk(b"IEND", b""),
        ]
    )


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
