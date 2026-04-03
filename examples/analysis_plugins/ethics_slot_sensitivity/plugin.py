from __future__ import annotations

from collections import Counter, defaultdict

from aigc.benchmarking.interfaces import AnalysisPlugin, AnalysisPluginRequest
from aigc.benchmarking.models import AnalysisPluginResult


class EthicsSlotSensitivityPlugin(AnalysisPlugin):
    plugin_id = "ethics_slot_sensitivity"
    plugin_version = "v1"
    plugin_type = "comparative"

    def execute(self, request: AnalysisPluginRequest) -> list[AnalysisPluginResult]:
        normalized_by_record = {
            (item.target_model, item.source_record_id): item
            for item in request.normalized_results
            if item.source_record_id
        }
        grouped = defaultdict(Counter)
        for target_result in request.target_results:
            slot_assignments = dict(target_result.metadata.get("slot_assignments", {}) or {})
            normalized = normalized_by_record.get((target_result.target_model, target_result.source_record_id))
            decision_text = normalized.decision_text if normalized is not None else None
            if not decision_text:
                continue
            for slot_id, slot_value in sorted(slot_assignments.items()):
                grouped[(target_result.target_model, target_result.split, str(slot_id), str(slot_value))][decision_text] += 1

        results: list[AnalysisPluginResult] = []
        for (target_model, split, slot_id, slot_value), counts in sorted(grouped.items()):
            results.append(
                AnalysisPluginResult(
                    benchmark_id=request.benchmark_id,
                    plugin_id=self.plugin_id,
                    plugin_version=self.plugin_version,
                    plugin_type=self.plugin_type,
                    task_id=request.task.task_id,
                    target_model=target_model,
                    split=split,
                    status="success",
                    labels=[],
                    scores={"distinct_decision_count": len(counts)},
                    output={
                        "slot_id": slot_id,
                        "slot_value": slot_value,
                        "decision_counts": dict(sorted(counts.items())),
                    },
                    metadata={"aggregation": "slot_value"},
                )
            )
        return results
