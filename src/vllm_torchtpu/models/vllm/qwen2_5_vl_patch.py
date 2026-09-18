# SPDX-License-Identifier: Apache-2.0
"""Normalize Qwen2.5-VL inputs on the vLLM 0.29 encoder graph paths."""

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def maybe_patch_qwen2_5_vl(
        model_config: "ModelConfig | None" = None) -> bool | None:
    if (model_config is None
            or model_config.hf_config.model_type != "qwen2_5_vl"):
        return False

    from vllm.model_executor.models.qwen2_5_vl import \
        Qwen2_5_VLForConditionalGeneration

    model_cls = Qwen2_5_VLForConditionalGeneration
    graph_forward = model_cls.encoder_cudagraph_forward
    eager_forward = model_cls.encoder_eager_forward

    # vLLM 0.29 moves normalization from the processor to input_norm in
    # _process_image/video_input. The encoder graph protocol bypasses those
    # methods, including its eager fallback, so both entrypoints must apply it.
    @wraps(graph_forward)
    def normalized_graph_forward(self, values, path="default"):
        values = {
            **values,
            "pixel_values":
            self.input_norm(values["pixel_values"], self.visual.dtype),
        }
        return graph_forward(self, values, path=path)

    @wraps(eager_forward)
    def normalized_eager_forward(self, mm_kwargs, path="default"):
        key = ("pixel_values" if self.get_input_modality(mm_kwargs) == "image"
               else "pixel_values_videos")
        mm_kwargs = {
            **mm_kwargs,
            key: self.input_norm(mm_kwargs[key], self.visual.dtype),
        }
        return eager_forward(self, mm_kwargs, path=path)

    model_cls.encoder_cudagraph_forward = normalized_graph_forward
    model_cls.encoder_eager_forward = normalized_eager_forward
    return None
