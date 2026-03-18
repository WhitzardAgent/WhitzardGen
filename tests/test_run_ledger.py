import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from aigc.run_ledger import (
    LEDGER_FILENAME,
    RunLedgerWriter,
    SampleLedgerRecord,
    load_ledger_records,
)


class RunLedgerWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ledger_file_is_created_on_open(self) -> None:
        run_root = self.tmpdir / "run_001"
        ledger_path = run_root / LEDGER_FILENAME

        self.assertFalse(ledger_path.exists())

        with RunLedgerWriter(run_root, "run_001"):
            self.assertTrue(ledger_path.exists())

    def test_append_success_writes_record(self) -> None:
        run_root = self.tmpdir / "run_002"

        with RunLedgerWriter(run_root, "run_002") as ledger:
            ledger.append_success(
                task_id="task_001",
                model_name="Z-Image",
                prompt_id="p001",
                prompt="a futuristic city",
                artifact_type="image",
                artifact_path="runs/run_002/Z-Image/p001.png",
                replica_id=0,
                batch_id="batch_001",
                batch_index=0,
                execution_mode="mock",
            )

        records = load_ledger_records(run_root)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["run_id"], "run_002")
        self.assertEqual(record["task_id"], "task_001")
        self.assertEqual(record["model_name"], "Z-Image")
        self.assertEqual(record["prompt_id"], "p001")
        self.assertEqual(record["prompt"], "a futuristic city")
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["artifact_type"], "image")
        self.assertEqual(record["artifact_path"], "runs/run_002/Z-Image/p001.png")
        self.assertEqual(record["replica_id"], 0)
        self.assertEqual(record["batch_id"], "batch_001")
        self.assertEqual(record["batch_index"], 0)
        self.assertEqual(record["execution_mode"], "mock")
        self.assertIsNone(record["error_message"])

    def test_append_failure_writes_record(self) -> None:
        run_root = self.tmpdir / "run_003"

        with RunLedgerWriter(run_root, "run_003") as ledger:
            ledger.append_failure(
                task_id="task_002",
                model_name="Z-Image",
                prompt_id="p002",
                prompt="a cat on a chair",
                error_message="CUDA out of memory",
                replica_id=1,
                batch_id="batch_002",
                execution_mode="real",
            )

        records = load_ledger_records(run_root)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_message"], "CUDA out of memory")
        self.assertIsNone(record["artifact_type"])
        self.assertIsNone(record["artifact_path"])

    def test_append_from_task_result_flattens_batch_items(self) -> None:
        run_root = self.tmpdir / "run_004"

        prompts = [
            {"prompt_id": "p001", "prompt": "prompt one", "language": "en"},
            {"prompt_id": "p002", "prompt": "prompt two", "language": "en"},
        ]
        batch_items = [
            {
                "prompt_id": "p001",
                "status": "success",
                "artifacts": [
                    {"type": "image", "path": "runs/run_004/Z-Image/p001.png"}
                ],
                "metadata": {"batch_index": 0},
            },
            {
                "prompt_id": "p002",
                "status": "success",
                "artifacts": [
                    {"type": "image", "path": "runs/run_004/Z-Image/p002.png"}
                ],
                "metadata": {"batch_index": 1},
            },
        ]

        with RunLedgerWriter(run_root, "run_004") as ledger:
            ledger.append_from_task_result(
                task_id="task_001",
                model_name="Z-Image",
                prompts=prompts,
                batch_items=batch_items,
                execution_mode="mock",
                replica_id=0,
                batch_id="batch_001",
            )

        records = load_ledger_records(run_root)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["prompt_id"], "p001")
        self.assertEqual(records[0]["batch_index"], 0)
        self.assertEqual(records[1]["prompt_id"], "p002")
        self.assertEqual(records[1]["batch_index"], 1)

    def test_append_from_task_result_handles_failures(self) -> None:
        run_root = self.tmpdir / "run_005"

        prompts = [
            {"prompt_id": "p001", "prompt": "prompt one", "language": "en"},
            {"prompt_id": "p002", "prompt": "prompt two", "language": "en"},
        ]
        batch_items = [
            {
                "prompt_id": "p001",
                "status": "success",
                "artifacts": [
                    {"type": "image", "path": "runs/run_005/Z-Image/p001.png"}
                ],
            },
            {
                "prompt_id": "p002",
                "status": "failed",
                "error": "Generation failed",
            },
        ]

        with RunLedgerWriter(run_root, "run_005") as ledger:
            ledger.append_from_task_result(
                task_id="task_001",
                model_name="Z-Image",
                prompts=prompts,
                batch_items=batch_items,
                execution_mode="real",
            )

        records = load_ledger_records(run_root)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "success")
        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(records[1]["error_message"], "Generation failed")

    def test_append_is_flushed_immediately(self) -> None:
        run_root = self.tmpdir / "run_006"
        ledger_path = run_root / LEDGER_FILENAME

        ledger = RunLedgerWriter(run_root, "run_006")
        ledger.open()

        ledger.append_success(
            task_id="task_001",
            model_name="Z-Image",
            prompt_id="p001",
            prompt="test prompt",
            artifact_type="image",
            artifact_path="test.png",
        )

        records = load_ledger_records(run_root)
        self.assertEqual(len(records), 1)

        ledger.close()

    def test_multiple_appends_in_sequence(self) -> None:
        run_root = self.tmpdir / "run_007"

        with RunLedgerWriter(run_root, "run_007") as ledger:
            for i in range(5):
                ledger.append_success(
                    task_id=f"task_{i:03d}",
                    model_name="Z-Image",
                    prompt_id=f"p{i:03d}",
                    prompt=f"prompt {i}",
                    artifact_type="image",
                    artifact_path=f"image_{i}.png",
                )

        records = load_ledger_records(run_root)
        self.assertEqual(len(records), 5)
        for i, record in enumerate(records):
            self.assertEqual(record["prompt_id"], f"p{i:03d}")

    def test_sample_ledger_record_to_dict(self) -> None:
        record = SampleLedgerRecord(
            timestamp="2026-03-18T10:00:00Z",
            run_id="run_001",
            task_id="task_001",
            model_name="Z-Image",
            prompt_id="p001",
            prompt="test prompt",
            status="success",
            artifact_type="image",
            artifact_path="test.png",
            error_message=None,
            replica_id=0,
            batch_id="batch_001",
            batch_index=0,
            execution_mode="mock",
        )

        d = record.to_dict()
        self.assertEqual(d["run_id"], "run_001")
        self.assertEqual(d["status"], "success")
        self.assertIsNone(d["error_message"])

    def test_load_ledger_records_returns_empty_list_for_missing_file(self) -> None:
        run_root = self.tmpdir / "nonexistent"
        records = load_ledger_records(run_root)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
