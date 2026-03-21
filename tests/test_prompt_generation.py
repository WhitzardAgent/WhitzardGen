import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from aigc.prompt_generation import generate_prompt_bundle, inspect_prompt_bundle, plan_theme_tree


class PromptGenerationTests(unittest.TestCase):
    def test_plan_theme_tree_honors_quota_and_seed(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: test_tree
            defaults:
              generation_profile: photorealistic
              language: en
            categories:
              - name: Animals
                count: 4
                children:
                  - name: Marine
                    count: 2
                    children:
                      - name: Fish
                        children:
                          - name: Coral reef
                          - name: Open water
                  - name: Birds
                    children:
                      - name: Shore birds
                        children:
                          - name: Wetland landing
                          - name: Shoreline foraging
            """,
        )

        payload = plan_theme_tree(tree_path=tree_path, seed=7)

        self.assertEqual(payload["sample_count"], 4)
        self.assertEqual(payload["counts_by_category"]["Animals"], 4)
        self.assertEqual(
            payload["counts_by_subcategory"]["Animals / Marine"],
            2,
        )

        payload_again = plan_theme_tree(tree_path=tree_path, seed=7)
        self.assertEqual(payload["items"], payload_again["items"])

    def test_generate_prompt_bundle_writes_bundle_in_preview_mode(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_video_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: video
            categories:
              - name: Animals
                count: 3
                children:
                  - name: Marine Animals
                    children:
                      - name: Tropical Fish
                        children:
                          - name: Coral reef close-up
                          - name: Open-water school movement
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"

        summary = generate_prompt_bundle(
            tree_path=tree_path,
            out_dir=out_dir,
            execution_mode="mock",
            seed=11,
        )

        self.assertTrue(Path(summary.prompts_path).exists())
        self.assertTrue(Path(summary.manifest_path).exists())
        self.assertTrue(Path(summary.sampling_plan_path).exists())
        self.assertTrue(Path(summary.generation_log_path).exists())
        self.assertTrue(Path(summary.stats_path).exists())

        prompts = [
            json.loads(line)
            for line in Path(summary.prompts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0]["metadata"]["generation_profile"], "photorealistic")
        self.assertEqual(prompts[0]["metadata"]["realism_target"], "photorealistic")
        self.assertEqual(prompts[0]["metadata"]["intended_modality"], "video")
        self.assertEqual(prompts[0]["metadata"]["prompt_template"], "photorealistic_base")
        self.assertEqual(prompts[0]["metadata"]["prompt_style_family"], "detailed_sentence")
        self.assertGreaterEqual(len(prompts[0]["metadata"]["few_shot_example_ids"]), 1)
        self.assertIn("annotation_hints", prompts[0]["metadata"])

        inspected = inspect_prompt_bundle(out_dir)
        self.assertEqual(inspected["prompt_count"], 3)
        self.assertEqual(inspected["manifest"]["tree_name"], "realistic_video_prompts")
        self.assertEqual(inspected["manifest"]["prompt_template"], "photorealistic_base")
        self.assertEqual(inspected["manifest"]["prompt_style_family"], "detailed_sentence")

    def test_generate_prompt_bundle_can_use_llm_kernel_path(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_image_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: image
            categories:
              - name: Urban Life
                count: 2
                children:
                  - name: Street Scenes
                    children:
                      - name: Commuters
                        children:
                          - name: Crosswalk rush at golden hour
                          - name: Rainy sidewalk umbrella flow
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"

        def fake_run_single_model(**kwargs):
            llm_run_dir = Path(kwargs["out_dir"])
            exports_dir = llm_run_dir / "exports"
            artifacts_dir = llm_run_dir / "artifacts"
            exports_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            requests = [
                json.loads(line)
                for line in Path(kwargs["prompt_file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records = []
            for index, request in enumerate(requests, start=1):
                artifact_path = artifacts_dir / f"{request['prompt_id']}.txt"
                artifact_path.write_text(
                    json.dumps(
                        {
                            "prompt": f"Photorealistic prompt {index} for {request['metadata']['theme_path'][-1]}",
                            "negative_prompt": "anime, stylized, low quality",
                            "annotation_hints": {
                                "scene_category": "Urban Life",
                                "subject_category": "Street Scenes",
                            },
                            "tags": ["photorealistic", "urban"],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                records.append(
                    {
                        "record_id": f"rec_{index:08d}",
                        "prompt_id": request["prompt_id"],
                        "artifact_path": str(artifact_path),
                    }
                )
            dataset_path = exports_dir / "dataset.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )
            return type(
                "Summary",
                (),
                {
                    "run_id": "llm_run_001",
                    "export_path": str(dataset_path),
                },
            )()

        with patch("aigc.prompt_generation.service.run_single_model", side_effect=fake_run_single_model):
            summary = generate_prompt_bundle(
                tree_path=tree_path,
                out_dir=out_dir,
                llm_model="Qwen3-32B",
                execution_mode="real",
                seed=5,
            )

        prompts = [
            json.loads(line)
            for line in Path(summary.prompts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all(prompt["metadata"]["llm_model"] == "Qwen3-32B" for prompt in prompts))
        self.assertTrue(all(prompt["negative_prompt"] for prompt in prompts))

    def test_generate_prompt_bundle_uses_profile_default_llm_model_when_cli_omitted(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_image_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: image
            categories:
              - name: Urban Life
                count: 1
                children:
                  - name: Street Scenes
                    children:
                      - name: Commuters
                        children:
                          - name: Crosswalk rush at golden hour
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"

        summary = generate_prompt_bundle(
            tree_path=tree_path,
            out_dir=out_dir,
            execution_mode="mock",
            seed=9,
        )

        manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
        prompts = [
            json.loads(line)
            for line in Path(summary.prompts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary.llm_model, "Qwen3-32B")
        self.assertEqual(manifest["llm_model"], "Qwen3-32B")
        self.assertEqual(prompts[0]["metadata"]["llm_model"], "Qwen3-32B")

    def test_generate_prompt_bundle_cli_llm_model_overrides_profile_default(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_image_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: image
            categories:
              - name: Urban Life
                count: 1
                children:
                  - name: Street Scenes
                    children:
                      - name: Commuters
                        children:
                          - name: Crosswalk rush at golden hour
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"

        summary = generate_prompt_bundle(
            tree_path=tree_path,
            out_dir=out_dir,
            execution_mode="mock",
            llm_model="Local-Transformers-T2T",
            seed=9,
        )

        manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
        prompts = [
            json.loads(line)
            for line in Path(summary.prompts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary.llm_model, "Local-Transformers-T2T")
        self.assertEqual(manifest["llm_model"], "Local-Transformers-T2T")
        self.assertEqual(prompts[0]["metadata"]["llm_model"], "Local-Transformers-T2T")

    def test_generate_prompt_bundle_honors_template_and_style_precedence(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_image_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: image
              prompt_template: documentary_scene
              prompt_style_family: short_sentence
            categories:
              - name: Urban Life
                count: 1
                children:
                  - name: Street Scenes
                    children:
                      - name: Commuters
                        children:
                          - name: Crosswalk rush at golden hour
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"

        summary = generate_prompt_bundle(
            tree_path=tree_path,
            out_dir=out_dir,
            execution_mode="mock",
            seed=3,
            template_name="synthetic_dataset_v1",
            style_family_name="keyword_list",
            target_model_name="Z-Image",
        )

        manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
        prompts = [
            json.loads(line)
            for line in Path(summary.prompts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(manifest["prompt_template"], "synthetic_dataset_v1")
        self.assertEqual(manifest["prompt_style_family"], "keyword_list")
        self.assertEqual(manifest["target_model_name"], "Z-Image")
        self.assertEqual(manifest["resolved_style_source"], "cli")
        self.assertEqual(prompts[0]["metadata"]["prompt_template"], "synthetic_dataset_v1")
        self.assertEqual(prompts[0]["metadata"]["prompt_style_family"], "keyword_list")
        self.assertEqual(prompts[0]["metadata"]["target_model_name"], "Z-Image")

    def test_generate_prompt_bundle_renders_template_and_few_shots_into_llm_requests(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_image_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: image
            categories:
              - name: Animals
                count: 1
                children:
                  - name: Marine Animals
                    children:
                      - name: Tropical Fish
                        children:
                          - name: Coral reef close-up
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"
        captured_prompts: list[str] = []

        def fake_run_single_model(**kwargs):
            requests = [
                json.loads(line)
                for line in Path(kwargs["prompt_file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            captured_prompts.extend(request["prompt"] for request in requests)
            llm_run_dir = Path(kwargs["out_dir"])
            exports_dir = llm_run_dir / "exports"
            artifacts_dir = llm_run_dir / "artifacts"
            exports_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for index, request in enumerate(requests, start=1):
                artifact_path = artifacts_dir / f"{request['prompt_id']}.txt"
                artifact_path.write_text(
                    json.dumps(
                        {
                            "prompt": "tropical fish, coral reef, realistic underwater light, documentary detail",
                            "negative_prompt": "anime, stylized, low quality",
                            "annotation_hints": {"scene_category": "Animals"},
                            "tags": ["marine", "keyword-list"],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                records.append(
                    {
                        "record_id": f"rec_{index:08d}",
                        "prompt_id": request["prompt_id"],
                        "artifact_path": str(artifact_path),
                    }
                )
            dataset_path = exports_dir / "dataset.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )
            return type("Summary", (), {"run_id": "llm_run_001", "export_path": str(dataset_path)})()

        with patch("aigc.prompt_generation.service.run_single_model", side_effect=fake_run_single_model):
            summary = generate_prompt_bundle(
                tree_path=tree_path,
                out_dir=out_dir,
                llm_model="Qwen3-32B",
                execution_mode="real",
                seed=9,
                template_name="synthetic_dataset_v1",
                style_family_name="keyword_list",
            )

        self.assertEqual(summary.prompt_template, "synthetic_dataset_v1")
        self.assertEqual(summary.prompt_style_family, "keyword_list")
        self.assertTrue(captured_prompts)
        self.assertIn("Prompt writing style:", captured_prompts[0])
        self.assertIn("Few-shot examples:", captured_prompts[0])
        self.assertIn("keyword_list_animals_001", captured_prompts[0])

    def test_generate_prompt_bundle_allows_empty_matching_few_shots(self) -> None:
        tree_path = self._write_file(
            "theme_tree.yaml",
            """
            version: v1
            name: realistic_audio_prompts
            defaults:
              generation_profile: photorealistic
              language: en
              intended_modality: audio
            categories:
              - name: Weather
                count: 1
                children:
                  - name: Rain
                    children:
                      - name: City Rain
                        children:
                          - name: Downtown rain ambience
            """,
        )
        out_dir = Path(tempfile.mkdtemp()) / "prompt_bundle"
        captured_prompts: list[str] = []

        def fake_run_single_model(**kwargs):
            requests = [
                json.loads(line)
                for line in Path(kwargs["prompt_file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            captured_prompts.extend(request["prompt"] for request in requests)
            llm_run_dir = Path(kwargs["out_dir"])
            exports_dir = llm_run_dir / "exports"
            artifacts_dir = llm_run_dir / "artifacts"
            exports_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / "synth_000001.txt"
            artifact_path.write_text(
                json.dumps(
                    {
                        "prompt": "steady downtown rain ambience, distant traffic, realistic city reverb",
                        "negative_prompt": "synthetic, cartoonish",
                        "annotation_hints": {"scene_category": "Weather"},
                        "tags": ["rain", "audio"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            dataset_path = exports_dir / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "record_id": "rec_00000001",
                        "prompt_id": "synth_000001",
                        "artifact_path": str(artifact_path),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            return type("Summary", (), {"run_id": "llm_run_001", "export_path": str(dataset_path)})()

        with patch("aigc.prompt_generation.service.run_single_model", side_effect=fake_run_single_model):
            summary = generate_prompt_bundle(
                tree_path=tree_path,
                out_dir=out_dir,
                llm_model="Qwen3-32B",
                execution_mode="real",
                seed=10,
                template_name="photorealistic_base",
                style_family_name="keyword_list",
                intended_modality="audio",
            )

        self.assertEqual(summary.few_shot_example_count, 0)
        self.assertIn("No few-shot examples selected.", captured_prompts[0])

    def _write_file(self, name: str, content: str) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        path = tmpdir / name
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        return path
