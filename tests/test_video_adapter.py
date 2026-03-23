import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aigc.adapters.video_family import (
    CogVideoX5BAdapter,
    HeliosPyramidAdapter,
    HunyuanVideo15Adapter,
    LongCatVideoAdapter,
    MOVAVideoAdapter,
    WanT2VDiffusersAdapter,
    extract_video_metadata,
    metadata_sidecar_path,
    resolve_video_model_reference,
)
from aigc.registry import load_registry


class VideoAdapterTests(unittest.TestCase):
    def test_mock_video_execution_generates_artifact_and_metadata(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["a cinematic drone shot over a coastline"],
            prompt_ids=["p001"],
            params={
                "width": 640,
                "height": 360,
                "fps": 12,
                "num_frames": 25,
                "num_inference_steps": 8,
                "guidance_scale": 4.0,
                "negative_prompts": ["low quality"],
                "seed": 11,
                "_runtime_config": {"execution_mode": "mock"},
            },
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_001"

        result = adapter.execute(
            plan=plan,
            prompts=["a cinematic drone shot over a coastline"],
            params={},
            workdir=str(tmpdir),
        )
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["a cinematic drone shot over a coastline"],
            prompt_ids=["p001"],
            workdir=str(tmpdir),
        )

        self.assertEqual(collected.status, "success")
        item = collected.batch_items[0]
        self.assertEqual(item.prompt_id, "p001")
        self.assertEqual(item.artifacts[0].type, "video")
        self.assertTrue(Path(item.artifacts[0].path).exists())
        self.assertTrue(metadata_sidecar_path(item.artifacts[0].path).exists())
        self.assertEqual(item.metadata["batch_id"], "batch_001")
        self.assertEqual(item.metadata["batch_index"], 0)
        self.assertTrue(item.metadata["mock"])
        self.assertEqual(collected.metadata["execution_mode"], "mock")
        self.assertEqual(item.artifacts[0].metadata["fps"], 12)
        self.assertEqual(item.artifacts[0].metadata["num_frames"], 25)
        self.assertEqual(item.artifacts[0].metadata["format"], "mp4")

    def test_extract_video_metadata_reads_sidecar(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["a quiet forest with drifting fog"],
            prompt_ids=["p002"],
            params={"_runtime_config": {"execution_mode": "mock"}},
            workdir=str(tmpdir),
        )
        result = adapter.execute(
            plan=plan,
            prompts=["a quiet forest with drifting fog"],
            params={},
            workdir=str(tmpdir),
        )
        path = result.outputs["p002"]["path"]
        metadata = extract_video_metadata(path)
        self.assertEqual(metadata["format"], "mp4")
        self.assertEqual(metadata["width"], 1280)
        self.assertEqual(metadata["height"], 720)

    def test_wan_capabilities_enable_persistent_worker(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )

        self.assertTrue(adapter.capabilities.supports_persistent_worker)
        self.assertEqual(adapter.capabilities.preferred_worker_strategy, "persistent_worker")
        self.assertEqual(adapter.real_execution_mode, "in_process")

    def test_wan_validate_model_reference_requires_diffusers_layout(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        weights_dir = tmpdir / "Wan2.2-T2V-A14B-Diffusers"
        weights_dir.mkdir()
        (weights_dir / "model_index.json").write_text("{}", encoding="utf-8")
        registry = load_registry()
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "weights_path": str(weights_dir),
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)

        with self.assertRaisesRegex(
            RuntimeError,
            "Missing required files: vae/config.json",
        ):
            adapter.validate_model_reference(str(weights_dir))

    def test_video_model_reference_prefers_weights_path(self) -> None:
        registry = load_registry()
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "local_path": "/models/raw-wan",
                "weights_path": "/models/Wan2.2-T2V-A14B-Diffusers",
            },
        )

        self.assertEqual(
            resolve_video_model_reference(model),
            "/models/Wan2.2-T2V-A14B-Diffusers",
        )

    def test_wan_generate_frames_passes_guidance_scale_2(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        plan = adapter.prepare(
            prompts=["a cinematic duel in heavy rain"],
            prompt_ids=["p003"],
            params={
                "_runtime_config": {"execution_mode": "real"},
                "guidance_scale_2": 3.5,
            },
            workdir=tempfile.mkdtemp(),
        )

        class _FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        class _FakeTorch:
            class Generator(_FakeGenerator):
                pass

        class _FakePipe:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return type("Output", (), {"frames": [[b"frame-1"], [b"frame-2"]]})()

        pipe = _FakePipe()
        frames = adapter.generate_frames_batch(
            pipe=pipe,
            plan=plan,
            prompts=["a cinematic duel in heavy rain", "a city made of glass"],
            negative_prompts=["low quality", "blurry"],
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=40,
            guidance_scale=4.0,
            seed=42,
            torch=_FakeTorch,
            device="cuda",
        )

        self.assertEqual(frames, [[b"frame-1"], [b"frame-2"]])
        self.assertEqual(pipe.calls[0]["guidance_scale_2"], 3.5)
        self.assertEqual(pipe.calls[0]["negative_prompt"], ["low quality", "blurry"])
        self.assertEqual(pipe.calls[0]["num_frames"], 81)
        self.assertEqual(pipe.calls[0]["prompt"], ["a cinematic duel in heavy rain", "a city made of glass"])
        self.assertEqual(len(pipe.calls[0]["generator"]), 2)

    def test_wan_generate_frames_handles_numpy_like_frames_without_truthiness_check(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        plan = adapter.prepare(
            prompts=["prompt one"],
            prompt_ids=["p001"],
            params={"_runtime_config": {"execution_mode": "real"}},
            workdir=tempfile.mkdtemp(),
        )

        class _AmbiguousFrames:
            def __init__(self) -> None:
                self._batches = [[b"frame-1"], [b"frame-2"]]

            def __len__(self) -> int:
                return len(self._batches)

            def __iter__(self):
                return iter(self._batches)

            def __bool__(self):
                raise ValueError("ambiguous truth value")

        class _FakePipe:
            def __call__(self, **kwargs):
                return type("Output", (), {"frames": _AmbiguousFrames()})()

        class _FakeTorch:
            class Generator:
                def __init__(self, device: str) -> None:
                    self.device = device

                def manual_seed(self, seed: int):
                    return self

        frames = adapter.generate_frames_batch(
            pipe=_FakePipe(),
            plan=plan,
            prompts=["prompt one", "prompt two"],
            negative_prompts=["", ""],
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=40,
            guidance_scale=4.0,
            seed=None,
            torch=_FakeTorch,
            device="cuda",
        )

        self.assertEqual(frames, [[b"frame-1"], [b"frame-2"]])

    def test_wan_generate_frames_emits_true_progress_steps(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        plan = adapter.prepare(
            prompts=["prompt one"],
            prompt_ids=["p001"],
            params={"_runtime_config": {"execution_mode": "real"}},
            workdir=tempfile.mkdtemp(),
        )
        events: list[dict[str, object]] = []

        class _FakePipe:
            def __call__(
                self,
                *,
                prompt,
                negative_prompt,
                height,
                width,
                num_frames,
                guidance_scale,
                guidance_scale_2,
                num_inference_steps,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
                **kwargs,
            ):
                del (
                    prompt,
                    negative_prompt,
                    height,
                    width,
                    num_frames,
                    guidance_scale,
                    guidance_scale_2,
                    callback_on_step_end_tensor_inputs,
                    kwargs,
                )
                for step_index in range(num_inference_steps):
                    if callback_on_step_end is not None:
                        callback_on_step_end(self, step_index, 0, {})
                return type("Output", (), {"frames": [[b"frame-1"]]})()

        class _FakeTorch:
            class Generator:
                def __init__(self, device: str) -> None:
                    self.device = device

                def manual_seed(self, seed: int):
                    return self

        frames = adapter.generate_frames_batch(
            pipe=_FakePipe(),
            plan=plan,
            prompts=["prompt one"],
            negative_prompts=[""],
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=4,
            guidance_scale=4.0,
            seed=None,
            torch=_FakeTorch,
            device="cuda",
            progress_callback=events.append,
        )

        self.assertEqual(frames, [[b"frame-1"]])
        self.assertEqual([event["current_step"] for event in events], [1, 2, 3, 4])
        self.assertTrue(all(event["supports_true_progress"] for event in events))

    def test_wan_load_pipeline_enables_low_cpu_mem_usage(self) -> None:
        registry = load_registry()
        tmpdir = Path(tempfile.mkdtemp())
        weights_dir = tmpdir / "Wan2.2-T2V-A14B-Diffusers"
        (weights_dir / "vae").mkdir(parents=True)
        (weights_dir / "model_index.json").write_text("{}", encoding="utf-8")
        (weights_dir / "vae" / "config.json").write_text("{}", encoding="utf-8")
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "weights_path": str(weights_dir),
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)
        captured: dict[str, object] = {}

        class _FakeAutoencoderKLWan:
            @classmethod
            def from_pretrained(cls, model_ref: str, **kwargs):
                captured["vae_model_ref"] = model_ref
                captured["vae_kwargs"] = kwargs
                return object()

        class _FakePipe:
            def __init__(self) -> None:
                self.vae = None
                self.device = None

            def to(self, device: str):
                self.device = device
                return self

        class _FakePipelineClass:
            @classmethod
            def from_pretrained(cls, model_ref: str, **kwargs):
                captured["pipe_model_ref"] = model_ref
                captured["pipe_kwargs"] = kwargs
                return _FakePipe()

        class _FakeTorch:
            float32 = "float32"

        with patch.dict(
            "sys.modules",
            {
                "diffusers": type(
                    "_FakeDiffusersModule",
                    (),
                    {"AutoencoderKLWan": _FakeAutoencoderKLWan},
                )(),
            },
        ):
            pipe = adapter.load_pipeline(
                pipeline_class=_FakePipelineClass,
                torch=_FakeTorch(),
                device="cuda",
                dtype="bfloat16",
            )

        self.assertIsInstance(pipe, _FakePipe)
        self.assertEqual(captured["pipe_model_ref"], str(weights_dir))
        self.assertEqual(captured["vae_model_ref"], str(weights_dir))
        self.assertEqual(captured["pipe_kwargs"]["low_cpu_mem_usage"], True)
        self.assertEqual(captured["pipe_kwargs"]["torch_dtype"], "bfloat16")

    def test_cogvideox_batch_generation_uses_prompt_list_and_generators(self) -> None:
        registry = load_registry()
        adapter = CogVideoX5BAdapter(model_config=registry.get_model("CogVideoX-5B"))
        plan = adapter.prepare(
            prompts=["prompt one", "prompt two"],
            prompt_ids=["c001", "c002"],
            params={"_runtime_config": {"execution_mode": "real"}, "seed": 100},
            workdir=tempfile.mkdtemp(),
        )

        class _FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        class _FakeTorch:
            class Generator(_FakeGenerator):
                pass

        class _FakePipe:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return type("Output", (), {"frames": [[b"a"], [b"b"]]})()

        pipe = _FakePipe()
        frames = adapter.generate_frames_batch(
            pipe=pipe,
            plan=plan,
            prompts=["prompt one", "prompt two"],
            negative_prompts=["", ""],
            width=720,
            height=480,
            num_frames=49,
            num_inference_steps=50,
            guidance_scale=6.0,
            seed=100,
            torch=_FakeTorch,
            device="cuda",
        )

        self.assertEqual(frames, [[b"a"], [b"b"]])
        self.assertEqual(pipe.calls[0]["prompt"], ["prompt one", "prompt two"])
        self.assertEqual(pipe.calls[0]["num_videos_per_prompt"], 1)
        self.assertEqual(len(pipe.calls[0]["generator"]), 2)

    def test_helios_capabilities_enable_persistent_worker_and_batching(self) -> None:
        registry = load_registry()
        adapter = HeliosPyramidAdapter(model_config=registry.get_model("Helios"))

        self.assertTrue(adapter.capabilities.supports_batch_prompts)
        self.assertEqual(adapter.capabilities.preferred_batch_size, 2)
        self.assertTrue(adapter.capabilities.supports_negative_prompt)
        self.assertTrue(adapter.capabilities.supports_persistent_worker)
        self.assertEqual(adapter.real_execution_mode, "in_process")

    def test_helios_load_pipeline_uses_auto_model_vae(self) -> None:
        registry = load_registry()
        tmpdir = Path(tempfile.mkdtemp())
        weights_dir = tmpdir / "Helios-Distilled"
        (weights_dir / "vae").mkdir(parents=True)
        (weights_dir / "model_index.json").write_text("{}", encoding="utf-8")
        (weights_dir / "vae" / "config.json").write_text("{}", encoding="utf-8")
        model = replace(
            registry.get_model("Helios"),
            weights={
                **registry.get_model("Helios").weights,
                "local_path": str(weights_dir),
            },
        )
        adapter = HeliosPyramidAdapter(model_config=model)
        captured: dict[str, object] = {}

        class _FakeAutoModel:
            @classmethod
            def from_pretrained(cls, model_ref: str, **kwargs):
                captured["vae_model_ref"] = model_ref
                captured["vae_kwargs"] = kwargs
                return object()

        class _FakePipe:
            def __init__(self) -> None:
                self.vae = None
                self.device = None

            def to(self, device: str):
                self.device = device
                return self

        class _FakePipelineClass:
            @classmethod
            def from_pretrained(cls, model_ref: str, **kwargs):
                captured["pipe_model_ref"] = model_ref
                captured["pipe_kwargs"] = kwargs
                return _FakePipe()

        class _FakeTorch:
            float32 = "float32"

        with patch.dict(
            "sys.modules",
            {
                "diffusers": type(
                    "_FakeDiffusersModule",
                    (),
                    {"AutoModel": _FakeAutoModel},
                )(),
            },
        ):
            pipe = adapter.load_pipeline(
                pipeline_class=_FakePipelineClass,
                torch=_FakeTorch(),
                device="cuda",
                dtype="bfloat16",
            )

        self.assertIsInstance(pipe, _FakePipe)
        self.assertEqual(captured["pipe_model_ref"], str(weights_dir))
        self.assertEqual(captured["vae_model_ref"], str(weights_dir))
        self.assertEqual(captured["pipe_kwargs"]["torch_dtype"], "bfloat16")

    def test_helios_generate_frames_passes_pyramid_settings_and_true_progress(self) -> None:
        registry = load_registry()
        adapter = HeliosPyramidAdapter(model_config=registry.get_model("Helios"))
        plan = adapter.prepare(
            prompts=["prompt one", "prompt two"],
            prompt_ids=["h001", "h002"],
            params={
                "_runtime_config": {"execution_mode": "real"},
                "seed": 5,
                "pyramid_num_inference_steps_list": [2, 2, 2],
                "is_amplify_first_chunk": True,
            },
            workdir=tempfile.mkdtemp(),
        )
        events: list[dict[str, object]] = []

        class _FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        class _FakeTorch:
            float32 = "float32"

            class Generator(_FakeGenerator):
                pass

        class _FakePipe:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(
                self,
                *,
                prompt,
                negative_prompt,
                num_frames,
                pyramid_num_inference_steps_list,
                guidance_scale,
                is_amplify_first_chunk,
                generator=None,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
                width=None,
                height=None,
            ):
                del callback_on_step_end_tensor_inputs
                self.calls.append(
                    {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_frames": num_frames,
                        "pyramid_num_inference_steps_list": pyramid_num_inference_steps_list,
                        "guidance_scale": guidance_scale,
                        "is_amplify_first_chunk": is_amplify_first_chunk,
                        "generator": generator,
                        "width": width,
                        "height": height,
                    }
                )
                for step_index in range(sum(pyramid_num_inference_steps_list)):
                    if callback_on_step_end is not None:
                        callback_on_step_end(self, step_index, 0, {})
                return type("Output", (), {"frames": [[b"a"], [b"b"]]})()

        pipe = _FakePipe()
        frames = adapter.generate_frames_batch(
            pipe=pipe,
            plan=plan,
            prompts=["prompt one", "prompt two"],
            negative_prompts=["neg one", "neg two"],
            width=640,
            height=384,
            num_frames=240,
            num_inference_steps=6,
            guidance_scale=1.0,
            seed=5,
            torch=_FakeTorch,
            device="cuda",
            progress_callback=events.append,
        )

        self.assertEqual(frames, [[b"a"], [b"b"]])
        self.assertEqual(pipe.calls[0]["prompt"], ["prompt one", "prompt two"])
        self.assertEqual(pipe.calls[0]["negative_prompt"], ["neg one", "neg two"])
        self.assertEqual(pipe.calls[0]["pyramid_num_inference_steps_list"], [2, 2, 2])
        self.assertEqual(pipe.calls[0]["width"], 640)
        self.assertEqual(pipe.calls[0]["height"], 384)
        self.assertEqual(len(pipe.calls[0]["generator"]), 2)
        self.assertEqual([event["current_step"] for event in events], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(event["supports_true_progress"] for event in events))

    def test_hunyuan_video_batch_capability_enabled(self) -> None:
        registry = load_registry()
        adapter = HunyuanVideo15Adapter(model_config=registry.get_model("HunyuanVideo-1.5"))
        self.assertTrue(adapter.capabilities.supports_batch_prompts)
        self.assertEqual(adapter.capabilities.preferred_batch_size, 2)
        self.assertTrue(adapter.capabilities.supports_persistent_worker)

    def test_hunyuan_video_diffusers_variant_uses_same_adapter_and_repo(self) -> None:
        registry = load_registry()
        model = registry.get_model("HunyuanVideo-1.5-Diffusers-720p_t2v")
        adapter = HunyuanVideo15Adapter(model_config=model)

        self.assertEqual(model.adapter, "HunyuanVideo15Adapter")
        self.assertEqual(
            model.weights["diffusers_repo"],
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        )
        self.assertTrue(adapter.capabilities.supports_persistent_worker)

    def test_hunyuan_video_generate_frames_omits_guidance_scale(self) -> None:
        registry = load_registry()
        adapter = HunyuanVideo15Adapter(model_config=registry.get_model("HunyuanVideo-1.5"))

        class _FakeTorch:
            class Generator:
                def __init__(self, device: str) -> None:
                    self.device = device
                    self.seed = None

                def manual_seed(self, seed: int):
                    self.seed = seed
                    return self

        class _FakePipe:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return type("Output", (), {"frames": [[b"a"], [b"b"]]})()

        pipe = _FakePipe()
        frames = adapter.generate_frames_batch(
            pipe=pipe,
            plan=type("Plan", (), {"inputs": {}})(),
            prompts=["prompt one", "prompt two"],
            negative_prompts=["neg one", "neg two"],
            width=1280,
            height=720,
            num_frames=121,
            num_inference_steps=50,
            guidance_scale=4.0,
            seed=7,
            torch=_FakeTorch,
            device="cuda",
        )

        self.assertEqual(frames, [[b"a"], [b"b"]])
        self.assertNotIn("guidance_scale", pipe.calls[0])
        self.assertEqual(pipe.calls[0]["negative_prompt"], ["neg one", "neg two"])
        self.assertEqual(len(pipe.calls[0]["generator"]), 2)

    def test_mova_capabilities_enable_persistent_worker_and_negative_prompt(self) -> None:
        registry = load_registry()
        adapter = MOVAVideoAdapter(model_config=registry.get_model("MOVA-720p"))

        self.assertTrue(adapter.capabilities.supports_persistent_worker)
        self.assertEqual(adapter.capabilities.preferred_worker_strategy, "persistent_worker")
        self.assertTrue(adapter.capabilities.supports_negative_prompt)
        self.assertEqual(adapter.real_execution_mode, "in_process")

    def test_mova_execute_real_uses_reference_image_and_saves_video(self) -> None:
        registry = load_registry()
        adapter = MOVAVideoAdapter(model_config=registry.get_model("MOVA-720p"))
        tmpdir = Path(tempfile.mkdtemp())
        ref_path = tmpdir / "reference.png"
        ref_path.write_text("stub", encoding="utf-8")
        plan = adapter.prepare(
            prompts=["a violinist performs on a rainy city rooftop"],
            prompt_ids=["p001"],
            params={
                "width": 1280,
                "height": 720,
                "fps": 24,
                "num_frames": 193,
                "num_inference_steps": 50,
                "guidance_scale": 5.0,
                "negative_prompts": [""],
                "seed": 11,
                "ref_path": str(ref_path),
                "_runtime_config": {"execution_mode": "real"},
            },
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_001"
        events: list[dict[str, object]] = []
        captured: dict[str, object] = {}

        class _FakeImage:
            def convert(self, mode: str):
                captured["convert_mode"] = mode
                return "rgb-image"

        class _FakeImageModule:
            @staticmethod
            def open(path: Path):
                captured["opened_path"] = str(path)
                return _FakeImage()

        class _FakeAudio:
            def cpu(self):
                captured["audio_cpu"] = True
                return self

            def squeeze(self):
                captured["audio_squeeze"] = True
                return self

        class _FakePipe:
            audio_sample_rate = 48000

            def __call__(self, **kwargs):
                captured["pipe_kwargs"] = kwargs
                return [[["frame-1", "frame-2"]], [_FakeAudio()]]

        def _fake_crop_and_resize(image, *, height: int, width: int):
            captured["crop_args"] = {"image": image, "height": height, "width": width}
            return "cropped-image"

        def _fake_save_video_with_audio(video, audio, output_path: str, **kwargs):
            captured["saved"] = {
                "video": video,
                "audio": audio,
                "output_path": output_path,
                "kwargs": kwargs,
            }
            Path(output_path).write_text("video", encoding="utf-8")

        class _FakeTorch:
            @staticmethod
            def cuda():
                return None

        with patch.object(
            adapter,
            "_get_or_load_pipeline",
            return_value=(
                _FakePipe(),
                _FakeTorch,
                "cuda",
                _FakeImageModule,
                _fake_crop_and_resize,
                _fake_save_video_with_audio,
            ),
        ):
            result = adapter.execute(
                plan=plan,
                prompts=["a violinist performs on a rainy city rooftop"],
                params={},
                workdir=str(tmpdir),
                progress_callback=events.append,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(captured["opened_path"], str(ref_path))
        self.assertEqual(captured["convert_mode"], "RGB")
        self.assertEqual(captured["crop_args"]["height"], 720)
        self.assertEqual(captured["crop_args"]["width"], 1280)
        self.assertEqual(captured["pipe_kwargs"]["prompt"], "a violinist performs on a rainy city rooftop")
        self.assertEqual(captured["pipe_kwargs"]["negative_prompt"], "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指")
        self.assertEqual(captured["pipe_kwargs"]["cfg_scale"], 5.0)
        self.assertEqual(captured["pipe_kwargs"]["sigma_shift"], 5.0)
        self.assertEqual(captured["saved"]["kwargs"]["sample_rate"], 48000)
        self.assertTrue(Path(result.outputs["p001"]["path"]).exists())
        self.assertEqual([event["phase"] for event in events], ["preparing_batch", "generating", "exporting", "completed"])

    def test_longcat_capabilities_enable_persistent_worker_and_batching(self) -> None:
        registry = load_registry()
        adapter = LongCatVideoAdapter(model_config=registry.get_model("LongCat-Video"))

        self.assertTrue(adapter.capabilities.supports_batch_prompts)
        self.assertEqual(adapter.capabilities.preferred_batch_size, 2)
        self.assertTrue(adapter.capabilities.supports_negative_prompt)
        self.assertTrue(adapter.capabilities.supports_persistent_worker)
        self.assertEqual(adapter.real_execution_mode, "in_process")

    def test_longcat_load_pipeline_uses_repo_modules_and_loads_loras(self) -> None:
        registry = load_registry()
        tmpdir = Path(tempfile.mkdtemp())
        repo_dir = tmpdir / "LongCat-Video"
        repo_dir.mkdir()
        weights_dir = tmpdir / "weights" / "LongCat-Video"
        for subdir in ("tokenizer", "text_encoder", "vae", "scheduler", "dit", "lora"):
            (weights_dir / subdir).mkdir(parents=True, exist_ok=True)
        (weights_dir / "lora" / "cfg_step_lora.safetensors").write_text("", encoding="utf-8")
        (weights_dir / "lora" / "refinement_lora.safetensors").write_text("", encoding="utf-8")

        model = replace(
            registry.get_model("LongCat-Video"),
            weights={
                **registry.get_model("LongCat-Video").weights,
                "repo_path": str(repo_dir),
                "weights_path": str(weights_dir),
            },
        )
        adapter = LongCatVideoAdapter(model_config=model)
        captured: dict[str, object] = {}

        class _FakeTokenizer:
            @classmethod
            def from_pretrained(cls, checkpoint_dir: str, **kwargs):
                captured["tokenizer"] = (checkpoint_dir, kwargs)
                return object()

        class _FakeTextEncoder:
            @classmethod
            def from_pretrained(cls, checkpoint_dir: str, **kwargs):
                captured["text_encoder"] = (checkpoint_dir, kwargs)
                return object()

        class _FakeVAE:
            @classmethod
            def from_pretrained(cls, checkpoint_dir: str, **kwargs):
                captured["vae"] = (checkpoint_dir, kwargs)
                return object()

        class _FakeScheduler:
            @classmethod
            def from_pretrained(cls, checkpoint_dir: str, **kwargs):
                captured["scheduler"] = (checkpoint_dir, kwargs)
                return object()

        class _FakeDit:
            def __init__(self) -> None:
                self.loaded_loras: list[tuple[str, str]] = []

            def load_lora(self, path: str, name: str) -> None:
                self.loaded_loras.append((path, name))

            @classmethod
            def from_pretrained(cls, checkpoint_dir: str, **kwargs):
                captured["dit"] = (checkpoint_dir, kwargs)
                instance = cls()
                captured["dit_instance"] = instance
                return instance

        class _FakePipeline:
            def __init__(self, **kwargs) -> None:
                captured["pipeline_kwargs"] = kwargs
                self.dit = kwargs["dit"]
                self.device = None

            def to(self, device: str):
                self.device = device
                return self

        class _FakeContextParallelUtil:
            @staticmethod
            def get_optimal_split(value: int):
                captured["cp_split_input"] = value
                return "cp-split"

        class _FakeTorch:
            bfloat16 = "bfloat16"
            float32 = "float32"

        with patch.dict(
            "sys.modules",
            {
                "longcat_video": types.ModuleType("longcat_video"),
                "longcat_video.modules": types.ModuleType("longcat_video.modules"),
                "transformers": type(
                    "_FakeTransformers",
                    (),
                    {"AutoTokenizer": _FakeTokenizer, "UMT5EncoderModel": _FakeTextEncoder},
                )(),
                "longcat_video.context_parallel": type(
                    "_FakeContextParallel",
                    (),
                    {"context_parallel_util": _FakeContextParallelUtil},
                )(),
                "longcat_video.modules.autoencoder_kl_wan": type(
                    "_FakeAutoencoderModule",
                    (),
                    {"AutoencoderKLWan": _FakeVAE},
                )(),
                "longcat_video.modules.scheduling_flow_match_euler_discrete": type(
                    "_FakeSchedulerModule",
                    (),
                    {"FlowMatchEulerDiscreteScheduler": _FakeScheduler},
                )(),
                "longcat_video.modules.longcat_video_dit": type(
                    "_FakeDitModule",
                    (),
                    {"LongCatVideoTransformer3DModel": _FakeDit},
                )(),
                "longcat_video.pipeline_longcat_video": type(
                    "_FakePipelineModule",
                    (),
                    {"LongCatVideoPipeline": _FakePipeline},
                )(),
            },
        ):
            pipe = adapter.load_pipeline(torch=_FakeTorch(), device="cuda", dtype="bfloat16")

        self.assertIsInstance(pipe, _FakePipeline)
        self.assertEqual(captured["cp_split_input"], 1)
        self.assertEqual(captured["tokenizer"][0], str(weights_dir))
        self.assertEqual(captured["dit"][1]["cp_split_hw"], "cp-split")
        self.assertEqual(captured["pipeline_kwargs"]["dit"], captured["dit_instance"])
        self.assertEqual(captured["dit_instance"].loaded_loras[0][1], "cfg_step_lora")
        self.assertEqual(captured["dit_instance"].loaded_loras[1][1], "refinement_lora")

    def test_longcat_generate_frames_batch_reuses_loaded_pipeline_across_prompt_batch(self) -> None:
        registry = load_registry()
        adapter = LongCatVideoAdapter(model_config=registry.get_model("LongCat-Video"))
        plan = adapter.prepare(
            prompts=["prompt one", "prompt two"],
            prompt_ids=["l001", "l002"],
            params={"_runtime_config": {"execution_mode": "real"}, "seed": 7},
            workdir=tempfile.mkdtemp(),
        )

        class _FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _FakeTorch:
            cuda = _FakeCuda()

            class Generator(_FakeGenerator):
                pass

        class _FakeDit:
            def __init__(self) -> None:
                self.enabled: list[list[str]] = []
                self.disabled = 0

            def enable_loras(self, names: list[str]) -> None:
                self.enabled.append(list(names))

            def disable_all_loras(self) -> None:
                self.disabled += 1

        class _FakePipe:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.dit = _FakeDit()

            def generate_t2v(self, **kwargs):
                self.calls.append(kwargs)
                frame_label = kwargs["prompt"]
                return [[[frame_label.encode("utf-8")]]]

        pipe = _FakePipe()
        frames = adapter.generate_frames_batch(
            pipe=pipe,
            plan=plan,
            prompts=["prompt one", "prompt two"],
            negative_prompts=["no blur", "no noise"],
            width=1280,
            height=720,
            num_frames=121,
            num_inference_steps=50,
            guidance_scale=4.0,
            seed=7,
            torch=_FakeTorch,
            device="cpu",
        )

        self.assertEqual(frames, [[[b"prompt one"]], [[b"prompt two"]]])
        self.assertEqual(len(pipe.calls), 2)
        self.assertEqual(pipe.calls[0]["prompt"], "prompt one")
        self.assertEqual(pipe.calls[1]["prompt"], "prompt two")
        self.assertEqual(pipe.calls[0]["negative_prompt"], "no blur")
        self.assertEqual(pipe.calls[1]["negative_prompt"], "no noise")
        self.assertEqual(pipe.calls[0]["generator"].seed, 7)
        self.assertEqual(pipe.calls[1]["generator"].seed, 8)

    def test_cogvideox_mock_video_defaults_match_reference_shape(self) -> None:
        registry = load_registry()
        adapter = CogVideoX5BAdapter(model_config=registry.get_model("CogVideoX-5B"))
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["A panda plays guitar in a bamboo forest."],
            prompt_ids=["c001"],
            params={"_runtime_config": {"execution_mode": "mock"}},
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_cog_001"

        result = adapter.execute(
            plan=plan,
            prompts=["A panda plays guitar in a bamboo forest."],
            params={},
            workdir=str(tmpdir),
        )
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["A panda plays guitar in a bamboo forest."],
            prompt_ids=["c001"],
            workdir=str(tmpdir),
        )

        self.assertEqual(collected.status, "success")
        item = collected.batch_items[0]
        self.assertEqual(item.artifacts[0].metadata["width"], 720)
        self.assertEqual(item.artifacts[0].metadata["height"], 480)
        self.assertEqual(item.artifacts[0].metadata["fps"], 8)
        self.assertEqual(item.artifacts[0].metadata["num_frames"], 49)
        self.assertEqual(item.artifacts[0].metadata["guidance_scale"], 6.0)
