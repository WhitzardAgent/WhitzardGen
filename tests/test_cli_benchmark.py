import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch


class BenchmarkCliTests(unittest.TestCase):
    def test_handle_benchmark_preview_prints_preview_summary(self) -> None:
        from whitzard.cli.main import handle_benchmark_preview

        summary = type(
            "PreviewSummary",
            (),
            {
                "preview_dir": "/tmp/benchmarks/ethics_preview",
                "preview_only": True,
                "preview_stage": "all",
                "preview_count": 2,
                "request_previews_path": "/tmp/benchmarks/ethics_preview/request_previews.jsonl",
                "request_preview_summary_path": "/tmp/benchmarks/ethics_preview/request_preview_summary.json",
                "request_previews_markdown_path": "/tmp/benchmarks/ethics_preview/request_previews.md",
                "counts_by_stage": {"writer": 2, "validator": 2},
                "sample_records": [],
                "to_dict": lambda self: {
                    "preview_dir": "/tmp/benchmarks/ethics_preview",
                    "preview_only": True,
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "builder": "ethics_sandbox",
                "source": "examples/benchmarks/ethics_sandbox/package",
                "package": None,
                "entrypoint": None,
                "builder_config": "/tmp/builder.yaml",
                "count_config": None,
                "llm_model": None,
                "synthesis_model": None,
                "profile": None,
                "template": None,
                "style_family": None,
                "target_model": None,
                "intended_modality": None,
                "out": None,
                "benchmark_name": "ethics_suite",
                "seed": 42,
                "realizations_per_template": 2,
                "mock": False,
                "execution_mode": None,
                "build_mode": "matrix",
                "preview_count": 2,
                "preview_stage": "all",
                "preview_format": "text",
                "output": "text",
            },
        )()

        with patch("whitzard.cli.main.build_benchmark", return_value=summary):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_benchmark_preview(args), 0)
        output = stream.getvalue()
        self.assertIn("Preview Dir: /tmp/benchmarks/ethics_preview", output)
        self.assertIn("Request Previews: /tmp/benchmarks/ethics_preview/request_previews.jsonl", output)

    def test_handle_benchmark_build_prints_summary(self) -> None:
        from whitzard.cli.main import handle_benchmark_build

        summary = type(
            "Summary",
            (),
            {
                "benchmark_id": "ethics_suite",
                "builder_name": "ethics_sandbox",
                "source_path": "examples/benchmarks/ethics_sandbox/package",
                "case_count": 38,
                "build_mode": "matrix",
                "benchmark_dir": "/tmp/benchmarks/ethics_suite",
                "cases_path": "/tmp/benchmarks/ethics_suite/cases.jsonl",
                "manifest_path": "/tmp/benchmarks/ethics_suite/benchmark_manifest.json",
                "to_dict": lambda self: {
                    "benchmark_id": "ethics_suite",
                    "builder_name": "ethics_sandbox",
                    "case_count": 38,
                    "build_mode": "matrix",
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "builder": "ethics_sandbox",
                "source": "examples/benchmarks/ethics_sandbox/package",
                "package": None,
                "entrypoint": None,
                "builder_config": "/tmp/builder.yaml",
                "count_config": None,
                "llm_model": None,
                "synthesis_model": None,
                "profile": None,
                "template": None,
                "style_family": None,
                "target_model": None,
                "intended_modality": None,
                "out": None,
                "benchmark_name": "ethics_suite",
                "seed": 42,
                "realizations_per_template": 2,
                "mock": False,
                "execution_mode": None,
                "build_mode": "matrix",
                "output": "json",
            },
        )()

        with patch("whitzard.cli.main.build_benchmark", return_value=summary):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_benchmark_build(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["benchmark_id"], "ethics_suite")

    def test_handle_benchmark_build_text_output_prints_next_evaluate_hint(self) -> None:
        from whitzard.cli.main import handle_benchmark_build

        summary = type(
            "Summary",
            (),
            {
                "benchmark_id": "ethics_suite",
                "builder_name": "ethics_sandbox",
                "source_path": "examples/benchmarks/ethics_sandbox/package",
                "case_count": 38,
                "build_mode": "matrix",
                "benchmark_dir": "/tmp/benchmarks/ethics_suite",
                "cases_path": "/tmp/benchmarks/ethics_suite/cases.jsonl",
                "manifest_path": "/tmp/benchmarks/ethics_suite/benchmark_manifest.json",
                "raw_realizations_path": "/tmp/benchmarks/ethics_suite/raw_realizations.jsonl",
                "rejected_realizations_path": "/tmp/benchmarks/ethics_suite/rejected_realizations.jsonl",
            },
        )()
        args = type(
            "Args",
            (),
            {
                "builder": "ethics_sandbox",
                "source": "examples/benchmarks/ethics_sandbox/package",
                "package": None,
                "entrypoint": None,
                "builder_config": "/tmp/builder.yaml",
                "count_config": None,
                "llm_model": None,
                "synthesis_model": None,
                "profile": None,
                "template": None,
                "style_family": None,
                "target_model": None,
                "intended_modality": None,
                "out": None,
                "benchmark_name": "ethics_suite",
                "seed": 42,
                "realizations_per_template": 2,
                "mock": False,
                "execution_mode": None,
                "build_mode": "matrix",
                "output": "text",
            },
        )()

        with patch("whitzard.cli.main.build_benchmark", return_value=summary):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_benchmark_build(args), 0)
        output = stream.getvalue()
        self.assertIn("Bundle Dir: /tmp/benchmarks/ethics_suite", output)
        self.assertIn("Cases Path: /tmp/benchmarks/ethics_suite/cases.jsonl", output)
        self.assertIn("Raw Realizations: /tmp/benchmarks/ethics_suite/raw_realizations.jsonl", output)
        self.assertIn("Rejected Realizations: /tmp/benchmarks/ethics_suite/rejected_realizations.jsonl", output)
        self.assertIn("Next Evaluate: whitzard evaluate run --benchmark /tmp/benchmarks/ethics_suite --targets <MODEL_NAME>", output)

    def test_handle_evaluate_run_uses_requested_targets_and_evaluators(self) -> None:
        from whitzard.cli.main import handle_evaluate_run

        summary = type(
            "Summary",
            (),
            {
                "experiment_id": "experiment_ethics_suite",
                "benchmark_id": "ethics_suite",
                "target_models": ["Qwen3-32B", "Helios"],
                "normalizer_ids": ["ethics_structural_normalizer"],
                "scorer_ids": ["ethics_structural_judge"],
                "analysis_plugin_ids": ["ethics_family_consistency"],
                "execution_mode": "mock",
                "case_count": 40,
                "target_run_count": 2,
                "normalized_result_count": 40,
                "score_record_count": 40,
                "group_analysis_record_count": 19,
                "analysis_plugin_result_count": 12,
                "experiment_dir": "/tmp/experiments/experiment_ethics_suite",
                "report_path": "/tmp/experiments/experiment_ethics_suite/report.md",
                "task_id": "task_ethics_suite",
                "to_dict": lambda self: {
                    "experiment_id": "experiment_ethics_suite",
                    "benchmark_id": "ethics_suite",
                    "target_models": ["Qwen3-32B", "Helios"],
                    "normalizer_ids": ["ethics_structural_normalizer"],
                    "scorer_ids": ["ethics_structural_judge"],
                    "analysis_plugin_ids": ["ethics_family_consistency"],
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "recipe": None,
                "benchmark": "/tmp/benchmarks/ethics_suite",
                "targets": ["Qwen3-32B", "Helios"],
                "normalizers": ["ethics_structural_normalizer"],
                "evaluators": ["ethics_structural_judge"],
                "analysis_plugins": ["ethics_family_consistency"],
                "normalizer_config": None,
                "evaluator_model": None,
                "evaluator_profile": None,
                "evaluator_template": None,
                "evaluator_config": None,
                "analysis_config": None,
                "launcher_config": None,
                "auto_launch": False,
                "out": None,
                "mock": True,
                "execution_mode": None,
                "output": "json",
            },
        )()

        def fake_evaluate_benchmark(**kwargs):
            self.assertEqual(kwargs["benchmark_path"], "/tmp/benchmarks/ethics_suite")
            self.assertEqual(kwargs["target_models"], ["Qwen3-32B", "Helios"])
            self.assertEqual(kwargs["normalizer_ids"], ["ethics_structural_normalizer"])
            self.assertEqual(kwargs["evaluator_ids"], ["ethics_structural_judge"])
            self.assertEqual(kwargs["analysis_plugin_ids"], ["ethics_family_consistency"])
            self.assertEqual(kwargs["execution_mode"], "mock")
            return summary

        with patch("whitzard.cli.main.evaluate_benchmark", side_effect=fake_evaluate_benchmark):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_evaluate_run(args), 0)
            payload = json.loads(stream.getvalue())

        self.assertEqual(payload["experiment_id"], "experiment_ethics_suite")

    def test_handle_evaluate_preview_prints_preview_summary(self) -> None:
        from whitzard.cli.main import handle_evaluate_preview

        summary = type(
            "PreviewSummary",
            (),
            {
                "preview_dir": "/tmp/experiments/ethics_preview",
                "preview_only": True,
                "preview_stage": "all",
                "preview_count": 2,
                "request_previews_path": "/tmp/experiments/ethics_preview/request_previews.jsonl",
                "request_preview_summary_path": "/tmp/experiments/ethics_preview/request_preview_summary.json",
                "request_previews_markdown_path": "/tmp/experiments/ethics_preview/request_previews.md",
                "counts_by_stage": {"target": 2, "judge": 2},
                "sample_records": [],
                "to_dict": lambda self: {
                    "preview_dir": "/tmp/experiments/ethics_preview",
                    "preview_only": True,
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "recipe": None,
                "benchmark": "/tmp/benchmarks/ethics_suite",
                "targets": ["Qwen3-32B", "Helios"],
                "normalizers": ["ethics_structural_normalizer"],
                "evaluators": ["ethics_structural_judge"],
                "analysis_plugins": ["ethics_family_consistency"],
                "normalizer_config": None,
                "evaluator_model": None,
                "evaluator_profile": None,
                "evaluator_template": None,
                "evaluator_config": None,
                "analysis_config": None,
                "launcher_config": None,
                "auto_launch": False,
                "out": None,
                "mock": True,
                "execution_mode": None,
                "preview_count": 2,
                "preview_stage": "all",
                "preview_format": "text",
                "output": "text",
            },
        )()

        with patch("whitzard.cli.main.evaluate_benchmark", return_value=summary):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_evaluate_preview(args), 0)
        output = stream.getvalue()
        self.assertIn("Preview Dir: /tmp/experiments/ethics_preview", output)
        self.assertIn("Request Previews: /tmp/experiments/ethics_preview/request_previews.jsonl", output)

    def test_handle_evaluate_run_recipe_builds_benchmark_and_uses_recipe_layers(self) -> None:
        from whitzard.cli.main import handle_evaluate_run

        summary = type(
            "Summary",
            (),
            {
                "experiment_id": "experiment_recipe_suite",
                "benchmark_id": "recipe_suite",
                "target_models": ["Qwen3-32B"],
                "normalizer_ids": ["ethics_structural_normalizer"],
                "scorer_ids": [],
                "analysis_plugin_ids": ["ethics_family_consistency", "ethics_slot_sensitivity"],
                "execution_mode": "real",
                "case_count": 24,
                "target_run_count": 1,
                "normalized_result_count": 24,
                "score_record_count": 0,
                "group_analysis_record_count": 4,
                "analysis_plugin_result_count": 9,
                "experiment_dir": "/tmp/experiments/experiment_recipe_suite",
                "report_path": "/tmp/experiments/experiment_recipe_suite/report.md",
                "task_id": "task_recipe_suite",
                "to_dict": lambda self: {
                    "experiment_id": "experiment_recipe_suite",
                    "benchmark_id": "recipe_suite",
                    "analysis_plugin_ids": ["ethics_family_consistency", "ethics_slot_sensitivity"],
                },
            },
        )()
        args = type(
            "Args",
            (),
            {
                "recipe": "/tmp/recipes/ethics.yaml",
                "benchmark": None,
                "targets": None,
                "normalizers": None,
                "evaluators": None,
                "analysis_plugins": None,
                "normalizer_config": None,
                "evaluator_model": None,
                "evaluator_profile": None,
                "evaluator_template": None,
                "evaluator_config": None,
                "analysis_config": None,
                "launcher_config": None,
                "auto_launch": False,
                "out": None,
                "mock": False,
                "execution_mode": None,
                "output": "json",
            },
        )()
        recipe = {
            "benchmark": {
                "builder": "ethics_sandbox",
                "source": "../examples/benchmarks/ethics_sandbox/package",
                "config": "../examples/benchmarks/ethics_sandbox/example_build.yaml",
                "build_mode": "matrix",
                "synthesis_model": "Qwen3-32B",
            },
            "targets": ["Qwen3-32B"],
            "normalizers": ["ethics_structural_normalizer"],
            "analysis_plugins": ["ethics_family_consistency", "ethics_slot_sensitivity"],
            "execution_mode": "real",
        }
        build_summary = type(
            "BuildSummary",
            (),
            {
                "benchmark_dir": "/tmp/benchmarks/recipe_suite",
            },
        )()

        with patch("whitzard.cli.main.load_experiment_recipe", return_value=recipe):
            with patch("whitzard.cli.main.build_benchmark", return_value=build_summary) as build_mock:
                with patch("whitzard.cli.main.evaluate_benchmark", return_value=summary) as evaluate_mock:
                    with redirect_stdout(StringIO()) as stream:
                        self.assertEqual(handle_evaluate_run(args), 0)
                    payload = json.loads(stream.getvalue())

        self.assertEqual(payload["experiment_id"], "experiment_recipe_suite")
        self.assertEqual(build_mock.call_args.kwargs["builder_name"], "ethics_sandbox")
        self.assertEqual(build_mock.call_args.kwargs["synthesis_model"], "Qwen3-32B")
        self.assertEqual(evaluate_mock.call_args.kwargs["benchmark_path"], "/tmp/benchmarks/recipe_suite")
        self.assertEqual(
            evaluate_mock.call_args.kwargs["normalizer_ids"],
            ["ethics_structural_normalizer"],
        )
        self.assertEqual(
            evaluate_mock.call_args.kwargs["analysis_plugin_ids"],
            ["ethics_family_consistency", "ethics_slot_sensitivity"],
        )

    def test_handle_experiments_report_prints_summary(self) -> None:
        from whitzard.cli.main import handle_experiments_report

        args = type(
            "Args",
            (),
            {
                "experiment": "experiment_ethics_suite",
                "output": "text",
            },
        )()
        payload = {
            "manifest": {
                "experiment_id": "experiment_ethics_suite",
                "benchmark_id": "ethics_suite",
                "target_models": ["Qwen3-32B"],
                "normalizer_ids": ["ethics_structural_normalizer"],
                "scorer_ids": ["ethics_structural_judge"],
                "analysis_plugin_ids": ["ethics_family_consistency"],
                "case_count": 40,
            },
            "summary": {
                "target_result_count": 40,
                "normalized_result_count": 40,
                "score_record_count": 40,
                "group_analysis_record_count": 19,
                "analysis_plugin_result_count": 12,
            },
            "report_path": "/tmp/experiments/experiment_ethics_suite/report.md",
        }

        with patch("whitzard.cli.main.inspect_experiment", return_value=payload):
            with redirect_stdout(StringIO()) as stream:
                self.assertEqual(handle_experiments_report(args), 0)

        output = stream.getvalue()
        self.assertIn("Experiment: experiment_ethics_suite", output)
        self.assertIn("Benchmark: ethics_suite", output)
        self.assertIn("Scorers: ethics_structural_judge", output)
