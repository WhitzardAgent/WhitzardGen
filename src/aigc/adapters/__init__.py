"""Model adapter subsystem."""

from aigc.adapters.zimage import ZImageAdapter
from aigc.adapters.stubs import (
    EchoTestAdapter,
    FluxImageAdapter,
    HunyuanImageAdapter,
    HunyuanVideo15Adapter,
    LongCatVideoAdapter,
    MOVAAdapter,
    PlaceholderAdapter,
    CogVideoX5BAdapter,
    QwenImageAdapter,
    StableDiffusionXLAdapter,
    WanT2VDiffusersAdapter,
    WanTI2VAdapter,
    ZImageTurboAdapter,
)

ADAPTER_REGISTRY = {
    "FluxImageAdapter": FluxImageAdapter,
    "StableDiffusionXLAdapter": StableDiffusionXLAdapter,
    "QwenImageAdapter": QwenImageAdapter,
    "ZImageTurboAdapter": ZImageTurboAdapter,
    "ZImageAdapter": ZImageAdapter,
    "HunyuanImageAdapter": HunyuanImageAdapter,
    "LongCatVideoAdapter": LongCatVideoAdapter,
    "WanTI2VAdapter": WanTI2VAdapter,
    "WanT2VDiffusersAdapter": WanT2VDiffusersAdapter,
    "CogVideoX5BAdapter": CogVideoX5BAdapter,
    "MOVAAdapter": MOVAAdapter,
    "HunyuanVideo15Adapter": HunyuanVideo15Adapter,
    "EchoTestAdapter": EchoTestAdapter,
}

__all__ = ["ADAPTER_REGISTRY", "PlaceholderAdapter"]
