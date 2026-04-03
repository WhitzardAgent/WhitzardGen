import json
import tempfile
import unittest
from pathlib import Path

from whitzard.registry import load_registry
from whitzard.run_ledger import RunLedgerWriter, build_sample_ledger_records
from whitzard.runtime.payloads import TaskPayload, TaskPrompt


class RunLedgerTests(unittest.TestCase):
    def test_writer_creates_ledger_file_and_flushes_immediately(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        ledger_path = tmpdir / "samples.jsonl"
        writer = RunLedgerWriter(ledger_path)

        writer.append_records([{"run_id": "run_001", "prompt_id": "p001"}])

        self.assertTrue(ledger_path.exists())
        self.assertEqual(
            [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()],
            [{"run_id": "run_001", "prompt_id": "p001"}],
        )
        writer.close()

    def test_build_sample_ledger_records_flattens_success_batch_items(self) -> None:
        registry = load_registry()
        model = registry.get_model("Z-Image")
        payload = TaskPayload(
            task_id="task_000001",
            model_name="Z-Image",
            execution_mode="mock",
            prompts=[
                TaskPrompt(prompt_id="p001", prompt="a cat", language="en"),
                TaskPrompt(prompt_id="p002", prompt="a dog", language="en"),
            ],
            params={},
            workdir="/tmp/demo",
            batch_id="z-image_batch_000001",
            runtime_config={"replica_id": 0, "gpu_assignment": [0]},
        )
        task_result = {
            "execution_mode": "mock",
            "model_result": {
                "status": "success",
                "batch_items": [
                    {
                        "prompt_id": "p001",
                        "status": "success",
                        "metadata": {"batch_index": 0},
                        "artifacts": [{"type": "image", "path": "/tmp/p001.png", "metadata": {}}],
                    },
                    {
                        "prompt_id": "p002",
                        "status": "success",
                        "metadata": {"batch_index": 1},
                        "artifacts": [{"type": "image", "path": "/tmp/p002.png", "metadata": {}}],
                    },
                ],
            },
        }

        records = build_sample_ledger_records(
            run_id="run_001",
            model=model,
            task_payload=payload,
            task_result=task_result,
            timestamp="2026-03-18T10:00:00+00:00",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual([record["prompt_id"] for record in records], ["p001", "p002"])
        self.assertEqual([record["artifact_path"] for record in records], ["/tmp/p001.png", "/tmp/p002.png"])
        self.assertTrue(all(record["status"] == "success" for record in records))

    def test_build_sample_ledger_records_emits_failure_per_prompt_when_batch_items_missing(self) -> None:
        registry = load_registry()
        model = registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        payload = TaskPayload(
            task_id="task_000002",
            model_name=model.name,
            execution_mode="real",
            prompts=[
                TaskPrompt(prompt_id="p101", prompt="storm over the sea", language="en"),
                TaskPrompt(prompt_id="p102", prompt="山谷中的云海", language="zh"),
            ],
            params={},
            workdir="/tmp/video",
            batch_id="wan_batch_000001",
        )
        task_result = {
            "execution_mode": "real",
            "execution_result": {"logs": "Traceback: boom"},
            "model_result": {"status": "failed", "batch_items": [], "logs": "Traceback: boom"},
        }

        records = build_sample_ledger_records(
            run_id="run_002",
            model=model,
            task_payload=payload,
            task_result=task_result,
            timestamp="2026-03-18T10:00:00+00:00",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual([record["prompt_id"] for record in records], ["p101", "p102"])
        self.assertTrue(all(record["status"] == "failed" for record in records))
        self.assertTrue(all(record["artifact_path"] is None for record in records))
        self.assertTrue(all("boom" in (record["error_message"] or "") for record in records))


if __name__ == "__main__":
    unittest.main()
