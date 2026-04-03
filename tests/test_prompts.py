import tempfile
import textwrap
import unittest
from pathlib import Path

from whitzard.prompts import PromptValidationError, load_prompts, normalize_text


class PromptLoaderTests(unittest.TestCase):
    def test_txt_loader_generates_ids_and_languages(self) -> None:
        path = self._write_file(
            "example.txt",
            """
            a futuristic city at night

            一只可爱的猫
            """,
        )
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].prompt_id, "prompt_000001")
        self.assertEqual(prompts[0].language, "en")
        self.assertEqual(prompts[1].prompt_id, "prompt_000002")
        self.assertEqual(prompts[1].language, "zh")

    def test_csv_loader_supports_optional_fields(self) -> None:
        path = self._write_file(
            "example.csv",
            """
            prompt_id,prompt,language,negative_prompt,parameters,metadata
            p001,a futuristic city at night,en,blurry,"{""width"": 1024, ""height"": 768}","{""split"": ""train""}"
            ,一只可爱的猫,,低质量,,
            """,
        )
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].prompt_id, "p001")
        self.assertEqual(prompts[0].negative_prompt, "blurry")
        self.assertEqual(prompts[0].parameters["width"], 1024)
        self.assertEqual(prompts[0].parameters["height"], 768)
        self.assertEqual(prompts[1].prompt_id, "prompt_000002")
        self.assertEqual(prompts[1].language, "zh")

    def test_jsonl_loader_requires_prompt_id(self) -> None:
        path = self._write_file(
            "example.jsonl",
            """
            {"prompt":"a futuristic city","language":"en"}
            """,
        )
        with self.assertRaises(PromptValidationError):
            load_prompts(path)

    def test_jsonl_loader_parses_full_records(self) -> None:
        path = self._write_file(
            "example.jsonl",
            """
            {"prompt_id":"p001","prompt":"  a futuristic city  ","language":"en","negative_prompt":" blurry ","parameters":{"width":1024,"guidance_scale":4.5},"metadata":{"split":"train"}}
            {"prompt_id":"p002","prompt":"一只猫","language":"zh"}
            """,
        )
        prompts = load_prompts(path)
        self.assertEqual(prompts[0].prompt, "a futuristic city")
        self.assertEqual(prompts[0].negative_prompt, "blurry")
        self.assertEqual(prompts[0].parameters["width"], 1024)
        self.assertEqual(prompts[0].parameters["guidance_scale"], 4.5)
        self.assertEqual(len(prompts), 2)

    def test_unknown_prompt_parameter_emits_warning_but_is_preserved(self) -> None:
        path = self._write_file(
            "unknown.jsonl",
            """
            {"prompt_id":"p001","prompt":"a futuristic city","language":"en","parameters":{"style":"cinematic"}}
            """,
        )
        warnings: list[str] = []

        prompts = load_prompts(path, warn=warnings.append)

        self.assertEqual(prompts[0].parameters["style"], "cinematic")
        self.assertEqual(len(warnings), 1)
        self.assertIn("Unknown generation parameter key", warnings[0])
        self.assertIn("prompt_id=p001", warnings[0])

    def test_invalid_prompt_parameter_value_fails_validation(self) -> None:
        path = self._write_file(
            "invalid.jsonl",
            """
            {"prompt_id":"p001","prompt":"a futuristic city","language":"en","parameters":{"width":"abc"}}
            """,
        )

        with self.assertRaises(PromptValidationError):
            load_prompts(path)

    def test_duplicate_prompt_ids_fail_validation(self) -> None:
        path = self._write_file(
            "dup.csv",
            """
            prompt_id,prompt
            p001,hello
            p001,world
            """,
        )
        with self.assertRaises(PromptValidationError):
            load_prompts(path)

    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  hello   world  "), "hello world")

    def test_canary_prompt_assets_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        image_txt = load_prompts(root / "prompts" / "canary_image.txt")
        image_csv = load_prompts(root / "prompts" / "canary_image.csv")
        video_csv = load_prompts(root / "prompts" / "canary_video.csv")
        image_jsonl = load_prompts(root / "prompts" / "canary_image.jsonl")
        video_jsonl = load_prompts(root / "prompts" / "canary_video.jsonl")
        image_rich = load_prompts(root / "prompts" / "example_image_rich.jsonl")
        video_rich = load_prompts(root / "prompts" / "example_video_rich.jsonl")

        self.assertEqual(len(image_txt), 2)
        self.assertEqual(len(image_csv), 2)
        self.assertEqual(len(video_csv), 2)
        self.assertEqual(len(image_jsonl), 2)
        self.assertEqual(len(video_jsonl), 2)
        self.assertEqual(len(image_rich), 2)
        self.assertEqual(len(video_rich), 2)
        self.assertEqual(image_txt[0].language, "en")
        self.assertEqual(image_txt[1].language, "zh")
        self.assertTrue(image_csv[0].negative_prompt)
        self.assertTrue(video_csv[0].negative_prompt)
        self.assertEqual(video_csv[0].prompt_id, "canary_vid_en_001")
        self.assertEqual(image_jsonl[1].prompt_id, "canary_img_jsonl_zh_001")
        self.assertTrue(image_jsonl[0].negative_prompt)
        self.assertTrue(video_jsonl[0].negative_prompt)
        self.assertEqual(image_rich[0].negative_prompt, image_rich[1].negative_prompt)
        self.assertEqual(video_rich[0].negative_prompt, video_rich[1].negative_prompt)
        self.assertEqual(image_rich[0].parameters["width"], 1024)
        self.assertEqual(video_rich[0].parameters["num_frames"], 81)

    def test_large_test_prompt_assets_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        image_txt = load_prompts(root / "prompts" / "test_image_100.txt")
        video_txt = load_prompts(root / "prompts" / "test_video_100.txt")

        self.assertEqual(len(image_txt), 100)
        self.assertEqual(len(video_txt), 100)
        self.assertEqual(image_txt[0].prompt_id, "prompt_000001")
        self.assertEqual(image_txt[-1].prompt_id, "prompt_000100")
        self.assertEqual(video_txt[0].language, "en")
        self.assertEqual(video_txt[50].language, "zh")

    def _write_file(self, name: str, content: str) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        path = tmpdir / name
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        return path
