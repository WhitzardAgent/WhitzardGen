"""Model adapter subsystem."""

from whitzard.adapters.images import (
    FluxImageAdapter,
    HunyuanImageAdapter,
    QwenImageAdapter,
    StableDiffusionXLAdapter,
    ZImageAdapter,
    ZImageTurboAdapter,
)
from whitzard.adapters.stubs import (
    EchoTestAdapter,
    HeliosPyramidAdapter,
    HunyuanVideo15Adapter,
    LongCatVideoAdapter,
    MOVAAdapter,
    PlaceholderAdapter,
    CogVideoX5BAdapter,
    WanT2VDiffusersAdapter,
    WanTI2VAdapter,
)
from whitzard.adapters.texts import (
    LocalTransformersTextAdapter,
    OpenAICompatibleTextAdapter,
    Qwen25InstructTextAdapter,
    Qwen3TextAdapter,
)

ADAPTER_REGISTRY = {
    "FluxImageAdapter": FluxImageAdapter,
    "StableDiffusionXLAdapter": StableDiffusionXLAdapter,
    "QwenImageAdapter": QwenImageAdapter,
    "ZImageTurboAdapter": ZImageTurboAdapter,
    "ZImageAdapter": ZImageAdapter,
    "HunyuanImageAdapter": HunyuanImageAdapter,
    "HeliosPyramidAdapter": HeliosPyramidAdapter,
    "LongCatVideoAdapter": LongCatVideoAdapter,
    "WanTI2VAdapter": WanTI2VAdapter,
    "WanT2VDiffusersAdapter": WanT2VDiffusersAdapter,
    "CogVideoX5BAdapter": CogVideoX5BAdapter,
    "MOVAAdapter": MOVAAdapter,
    "HunyuanVideo15Adapter": HunyuanVideo15Adapter,
    "EchoTestAdapter": EchoTestAdapter,
    "LocalTransformersTextAdapter": LocalTransformersTextAdapter,
    "Qwen25InstructTextAdapter": Qwen25InstructTextAdapter,
    "Qwen3TextAdapter": Qwen3TextAdapter,
    "OpenAICompatibleTextAdapter": OpenAICompatibleTextAdapter,
}

__all__ = ["ADAPTER_REGISTRY", "PlaceholderAdapter"]
