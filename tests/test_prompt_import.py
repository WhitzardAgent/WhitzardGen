from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from aigc.utils.prompt_import import convert_legacy_prompt_csv_to_jsonl


class PromptImportTests(unittest.TestCase):
    def test_convert_legacy_prompt_csv_to_jsonl(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        csv_path = tmpdir / "legacy.csv"
        jsonl_path = tmpdir / "converted.jsonl"
        csv_path.write_text(
            textwrap.dedent(
                """\
                idx,uuid,prompt,keyword,class_level_0,class_level_1,model_type,lang
                1,uuid-001,a realistic city street at dusk,city street,urban scene,street life,t2i,en
                2,uuid-002,一只猫坐在窗边,cat by window,animals,domestic pet,t2i,zh
                """
            ),
            encoding="utf-8",
        )

        summary = convert_legacy_prompt_csv_to_jsonl(
            csv_path=csv_path,
            jsonl_path=jsonl_path,
        )

        self.assertEqual(summary["record_count"], 2)
        rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(rows[0]["prompt_id"], "uuid-001")
        self.assertEqual(rows[0]["language"], "en")
        self.assertEqual(rows[0]["metadata"]["category"], "aigc_safety")
        self.assertEqual(rows[0]["metadata"]["subcategory"], "street life")
        self.assertEqual(rows[0]["metadata"]["subtopic"], "urban scene")
        self.assertEqual(
            rows[0]["metadata"]["theme_path"],
            ["aigc_safety", "street life", "urban scene", "city street"],
        )
        self.assertEqual(rows[1]["language"], "zh")

    def test_convert_legacy_prompt_csv_to_jsonl_falls_back_when_uuid_missing(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        csv_path = tmpdir / "legacy.csv"
        jsonl_path = tmpdir / "converted.jsonl"
        csv_path.write_text(
            textwrap.dedent(
                """\
                idx,uuid,prompt,keyword,class_level_0,class_level_1,model_type,lang
                7,,a close-up portrait,portrait,human,faces,t2i,en-US
                """
            ),
            encoding="utf-8",
        )

        convert_legacy_prompt_csv_to_jsonl(
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            category="custom_root",
        )

        rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(rows[0]["prompt_id"], "legacy_csv_7")
        self.assertEqual(rows[0]["language"], "en")
        self.assertEqual(rows[0]["metadata"]["category"], "custom_root")

    def test_convert_legacy_prompt_csv_to_jsonl_can_force_language(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        csv_path = tmpdir / "legacy.csv"
        jsonl_path = tmpdir / "converted.jsonl"
        csv_path.write_text(
            textwrap.dedent(
                """\
                idx,uuid,prompt,keyword,class_level_0,class_level_1,model_type,lang
                1,uuid-001,a realistic city street at dusk,city street,urban scene,street life,t2i,garbled-lang
                2,uuid-002,一只猫坐在窗边,cat by window,animals,domestic pet,t2i,???
                """
            ),
            encoding="utf-8",
        )

        summary = convert_legacy_prompt_csv_to_jsonl(
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            forced_language="zh",
        )

        self.assertEqual(summary["forced_language"], "zh")
        rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(rows[0]["language"], "zh")
        self.assertEqual(rows[1]["language"], "zh")
