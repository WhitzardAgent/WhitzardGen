from __future__ import annotations

from collections import Counter, defaultdict

from whitzard.benchmarking.interfaces import AnalysisPlugin, AnalysisPluginRequest
from whitzard.benchmarking.models import AnalysisPluginResult


class EthicsFamilyConsistencyPlugin(AnalysisPlugin):
    plugin_id = "ethics_family_consistency"
    plugin_version = "v1"
    plugin_type = "comparative"

    def execute(self, request: AnalysisPluginRequest) -> list[AnalysisPluginResult]:
        normalized_by_record = {
            (item.target_model, item.source_record_id): item
            for item in request.normalized_results
            if item.source_record_id
        }
        score_records_by_case = defaultdict(list)
        for item in request.score_records:
            score_records_by_case[(item.target_model, item.case_id)].append(item)
        grouped = defaultdict(list)
        for target_result in request.target_results:
            family_id = str(
                target_result.metadata.get("family_id")
                or target_result.metadata.get("template_id")
                or target_result.prompt_metadata.get("family_id")
                or target_result.prompt_metadata.get("template_id")
                or "default"
            )
            grouped[(target_result.target_model, target_result.split, family_id)].append(target_result)

        results: list[AnalysisPluginResult] = []
        for (target_model, split, family_id), rows in sorted(grouped.items()):
            action_counter = Counter()
            label_counter = Counter()
            for row in rows:
                normalized = normalized_by_record.get((row.target_model, row.source_record_id))
                if normalized and normalized.decision_text:
                    action_counter[normalized.decision_text] += 1
                for score_record in score_records_by_case.get((row.target_model, row.case_id), []):
                    for label in score_record.labels:
                        label_counter[label] += 1
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
                    labels=list(sorted(label_counter)),
                    scores={"case_count": len(rows)},
                    output={
                        "family_id": family_id,
                        "recommended_action_counts": dict(sorted(action_counter.items())),
                        "normative_label_counts": dict(sorted(label_counter.items())),
                    },
                    metadata={"aggregation": "family"},
                )
            )
        return results
