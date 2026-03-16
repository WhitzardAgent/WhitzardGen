import tempfile
import textwrap
import unittest
from pathlib import Path

from aigc.prompts import PromptValidationError, load_prompts, normalize_text


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
            p001,a futuristic city at night,en,blurry,"{""style"": ""cinematic""}","{""split"": ""train""}"
            ,一只可爱的猫,,低质量,,
            """,
        )
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].prompt_id, "p001")
        self.assertEqual(prompts[0].parameters["style"], "cinematic")
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
            {"prompt_id":"p001","prompt":"  a futuristic city  ","language":"en","negative_prompt":" blurry ","parameters":{"steps":50},"metadata":{"split":"train"}}
            {"prompt_id":"p002","prompt":"一只猫","language":"zh"}
            """,
        )
        prompts = load_prompts(path)
        self.assertEqual(prompts[0].prompt, "a futuristic city")
        self.assertEqual(prompts[0].negative_prompt, "blurry")
        self.assertEqual(prompts[0].parameters["steps"], 50)
        self.assertEqual(len(prompts), 2)

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
        video_csv = load_prompts(root / "prompts" / "canary_video.csv")
        image_jsonl = load_prompts(root / "prompts" / "canary_image.jsonl")

        self.assertEqual(len(image_txt), 2)
        self.assertEqual(len(video_csv), 2)
        self.assertEqual(len(image_jsonl), 2)
        self.assertEqual(image_txt[0].language, "en")
        self.assertEqual(image_txt[1].language, "zh")
        self.assertEqual(video_csv[0].prompt_id, "canary_vid_en_001")
        self.assertEqual(image_jsonl[1].prompt_id, "canary_img_jsonl_zh_001")

    def _write_file(self, name: str, content: str) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        path = tmpdir / name
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        return path
