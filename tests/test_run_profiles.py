import tempfile
import unittest
from pathlib import Path

from aigc.run_profiles import RunProfileError, load_run_profile, resolve_profile_run_request


class RunProfilesTests(unittest.TestCase):
    def test_load_run_profile_reads_models_prompts_and_runtime(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "prompts.txt"
        prompts_path.write_text("a lighthouse at dusk\n", encoding="utf-8")
        profile_path = tmpdir / "image_mock.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "name: image_mock",
                    "models:",
                    "  - Z-Image",
                    "  - FLUX.1-dev",
                    "prompts: prompts.txt",
                    "execution_mode: mock",
                    "runtime:",
                    "  available_gpus: [0, 1]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        profile = load_run_profile(profile_path)

        self.assertEqual(profile.name, "image_mock")
        self.assertEqual(profile.model_names, ["Z-Image", "FLUX.1-dev"])
        self.assertEqual(profile.prompt_file, prompts_path.resolve())
        self.assertEqual(profile.execution_mode, "mock")
        self.assertEqual(profile.available_gpus, [0, 1])

    def test_load_run_profile_reads_generation_defaults(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "prompts.jsonl"
        prompts_path.write_text(
            '{"prompt_id":"p001","prompt":"a lighthouse at dusk","language":"en"}\n',
            encoding="utf-8",
        )
        profile_path = tmpdir / "image_real.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "name: image_real",
                    "models: [Z-Image]",
                    "prompts: prompts.jsonl",
                    "execution_mode: real",
                    "generation_defaults:",
                    "  width: 1024",
                    "  height: 1024",
                    "  guidance_scale: 4.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        profile = load_run_profile(profile_path)

        self.assertEqual(
            profile.generation_defaults,
            {"width": 1024, "height": 1024, "guidance_scale": 4.5},
        )

    def test_load_run_profile_rejects_invalid_execution_mode(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "prompts.txt"
        prompts_path.write_text("a lighthouse at dusk\n", encoding="utf-8")
        profile_path = tmpdir / "bad.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "models: [Z-Image]",
                    "prompts: prompts.txt",
                    "execution_mode: invalid",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaises(RunProfileError):
            load_run_profile(profile_path)

    def test_resolve_profile_run_request_prefers_explicit_cli_values(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "prompts.txt"
        prompts_path.write_text("a lighthouse at dusk\n", encoding="utf-8")
        override_prompts = tmpdir / "override.txt"
        override_prompts.write_text("a fox in the snow\n", encoding="utf-8")
        profile_path = tmpdir / "profile.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "name: collection",
                    "models: [Z-Image, FLUX.1-dev]",
                    "prompts: prompts.txt",
                    "execution_mode: real",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        profile = load_run_profile(profile_path)

        request = resolve_profile_run_request(
            profile=profile,
            models_arg="Z-Image-Turbo",
            prompts_arg=str(override_prompts),
            execution_mode_arg="mock",
            mock_flag=False,
            out_arg=None,
            run_name_arg="override-run",
        )

        self.assertEqual(request["model_names"], ["Z-Image-Turbo"])
        self.assertEqual(request["prompt_file"], override_prompts)
        self.assertEqual(request["execution_mode"], "mock")
        self.assertEqual(request["run_name"], "override-run")

    def test_resolve_profile_run_request_includes_generation_defaults(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "prompts.txt"
        prompts_path.write_text("a lighthouse at dusk\n", encoding="utf-8")
        profile_path = tmpdir / "profile.yaml"
        profile_path.write_text(
            "\n".join(
                [
                    "name: collection",
                    "models: [Z-Image]",
                    "prompts: prompts.txt",
                    "execution_mode: real",
                    "generation_defaults:",
                    "  width: 1024",
                    "  num_inference_steps: 40",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        profile = load_run_profile(profile_path)

        request = resolve_profile_run_request(
            profile=profile,
            models_arg=None,
            prompts_arg=None,
            execution_mode_arg=None,
            mock_flag=False,
            out_arg=None,
            run_name_arg=None,
        )

        self.assertEqual(
            request["generation_defaults"],
            {"width": 1024, "num_inference_steps": 40},
        )
