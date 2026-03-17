from __future__ import annotations

from pathlib import Path

from aigc.adapters.base import (
    AdapterCapabilities,
    ArtifactRecord,
    BaseAdapter,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
)
from aigc.adapters.image_family import (
    FluxImageAdapter,
    HunyuanImageAdapter,
    QwenImageAdapter,
    StableDiffusionXLAdapter,
    ZImageTurboAdapter,
)
from aigc.adapters.video_family import (
    CogVideoX5BAdapter,
    HunyuanVideo15Adapter,
    LongCatVideoAdapter,
    MOVAVideoAdapter,
    WanT2VDiffusersAdapter,
    WanTI2VAdapter,
)


class PlaceholderAdapter(BaseAdapter):
    """Temporary adapter placeholder until real model integrations land."""

    capabilities = AdapterCapabilities()

    def prepare(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, object],
        workdir: str,
    ) -> ExecutionPlan:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented yet.")

    def collect(
        self,
        plan: ExecutionPlan,
        exec_result: ExecutionResult,
        prompts: list[str],
        prompt_ids: list[str],
        workdir: str,
    ) -> ModelResult:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented yet.")


MOVAAdapter = MOVAVideoAdapter


class EchoTestAdapter(BaseAdapter):
    """Small in-process adapter used to verify worker execution in tests."""

    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=8,
        preferred_batch_size=4,
        output_types=["text"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )

    def load_for_persistent_worker(self) -> None:
        counter_file = self.model_config.weights.get("load_counter_file")
        if not counter_file:
            return
        path = Path(str(counter_file))
        current = int(path.read_text(encoding="utf-8")) if path.exists() else 0
        path.write_text(str(current + 1), encoding="utf-8")

    def prepare(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, object],
        workdir: str,
    ) -> ExecutionPlan:
        return ExecutionPlan(mode="in_process", inputs={"prompt_ids": prompt_ids})

    def execute(
        self,
        plan: ExecutionPlan,
        prompts: list[str],
        params: dict[str, object],
        workdir: str,
    ) -> ExecutionResult:
        prompt_ids = list(plan.inputs.get("prompt_ids", []))
        outputs = {}
        for prompt_id, prompt in zip(prompt_ids, prompts, strict=True):
            output_path = Path(workdir) / f"{prompt_id}.txt"
            output_path.write_text(prompt, encoding="utf-8")
            outputs[prompt_id] = str(output_path)
        return ExecutionResult(exit_code=0, logs="echo test adapter", outputs=outputs)

    def collect(
        self,
        plan: ExecutionPlan,
        exec_result: ExecutionResult,
        prompts: list[str],
        prompt_ids: list[str],
        workdir: str,
    ) -> ModelResult:
        items = []
        for prompt_id in prompt_ids:
            path = str(exec_result.outputs[prompt_id])
            items.append(
                BatchItemResult(
                    prompt_id=prompt_id,
                    artifacts=[ArtifactRecord(type="text", path=path, metadata={})],
                    status="success",
                )
            )
        return ModelResult(status="success", batch_items=items, logs=exec_result.logs)
