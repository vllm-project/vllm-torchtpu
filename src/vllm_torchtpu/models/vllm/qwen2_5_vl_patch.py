# SPDX-License-Identifier: Apache-2.0
"""Normalize Qwen2.5-VL inputs on the encoder graph paths."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def maybe_patch_qwen2_5_vl(model_config: "ModelConfig | None" = None) -> bool | None:
    if model_config is None or model_config.hf_config.model_type != "qwen2_5_vl":
        return False

    # Upstream vLLM (>= 0.30) natively normalizes inputs in
    # Qwen2_5_VisionTransformer.forward via self.visual.input_norm;
    # monkeypatching top-level input_norm is no longer required.
    return None
