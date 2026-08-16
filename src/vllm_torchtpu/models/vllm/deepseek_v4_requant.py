# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Requantization utilities for DeepSeek-V4 attention weights."""

# TODO(patemotter): fold this into the dedicated DSv4 model.py's `load_weights`
# along with the rest of `deepseek_v4_patch.py`.

import torch

from vllm_torchtpu.layers.common.quantization import e8m0_to_fp32
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The weight dtypes that carry the block scales this module rebases.
_FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def re_quantize_attention_weight(
        weight_fp8: torch.Tensor,
        scale_block_fp8: torch.Tensor,
        block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_block_fp8.dtype == torch.float8_e8m0fnu:
        scale_block_fp8 = scale_block_fp8.view(torch.uint8)
    scale_block_fp32 = e8m0_to_fp32(scale_block_fp8)
    scale_expanded = scale_block_fp32.repeat_interleave(
        block_size, dim=0).repeat_interleave(block_size, dim=1)
    weight_fp32 = weight_fp8.to(torch.float32) * scale_expanded

    max_fp8_value = 448.0
    row_max = torch.max(torch.abs(weight_fp32), dim=1, keepdim=True).values
    scale_per_channel = torch.clamp(row_max / max_fp8_value, min=1e-12)
    weight_quantized = torch.clamp(weight_fp32 / scale_per_channel,
                                   -max_fp8_value,
                                   max_fp8_value).to(torch.float8_e4m3fn)
    return weight_quantized, scale_per_channel


def attention_requantizer_generator(weights_iter):
    buffered = {}

    def is_fp8_projection_to_requantize(name: str) -> bool:
        if ".attn." in name:
            for proj in [
                    "wq_a", "wq_b", "wo_a", "wo_b", "wkv", "indexer.wq_b"
            ]:
                if f".attn.{proj}." in name:
                    return True
        if ".shared_experts." in name:
            for proj in ["w1", "w2", "w3"]:
                if f".shared_experts.{proj}." in name:
                    return True
        return False

    for name, loaded_weight in weights_iter:
        if is_fp8_projection_to_requantize(name):
            if name.endswith(".weight"):
                if loaded_weight.dtype not in _FP8_WEIGHT_DTYPES:
                    yield name, loaded_weight
                    continue
                base_key = name[:-7]
                target_type = "weight"
            elif name.endswith(".scale"):
                base_key = name[:-6]
                target_type = "scale"
            elif name.endswith(".weight_scale"):
                base_key = name[:-13]
                target_type = "scale"
            elif name.endswith(".weight_scale_inv"):
                base_key = name[:-17]
                target_type = "scale"
            else:
                yield name, loaded_weight
                continue

            entry = buffered.setdefault(base_key, {})
            entry[target_type] = loaded_weight

            if "weight" in entry and "scale" in entry:
                w_new, s_new = re_quantize_attention_weight(
                    entry["weight"], entry["scale"])
                yield f"{base_key}.weight", w_new
                yield f"{base_key}.weight_scale", s_new
                del buffered[base_key]
        else:
            yield name, loaded_weight

    if buffered:
        # Every fp8 projection must arrive with both its weight and its scale;
        # only a paired entry is yielded on to the loader.
        raise ValueError(
            "DeepSeek-V4 requant: fp8 projections missing their weight/scale "
            f"partner: {sorted(buffered)}")
