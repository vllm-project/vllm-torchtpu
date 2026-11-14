# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/models/vllm/qwen4_exp/ple_geometry.py at
# e070c0f7ebd439082fb9993a9aae88e5aab04d70.
#   Owned by models/qwen38_flash from this copy on.
"""PLE cache and storage planning using the common layer's vocabulary geometry."""

import math
from typing import cast

try:
    from transformers import Qwen4ExpTextConfig
except ImportError:
    from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

from .ple import head_vocab_sizes


def ple_vocab_rows(config: Qwen4ExpTextConfig, layer_idx: int) -> int:
    ple_layer_ids = cast(list[int], config.ple_layer_ids)
    dense_index = sorted(set(ple_layer_ids)).index(layer_idx + 1)
    sizes = head_vocab_sizes(
        config.ngram_vocab_size_base,
        (config.ngram_size - 1) * config.heads_per_ngram,
        dense_index,
    )
    divisor = config.make_ngram_vocab_size_divisible_by
    return math.ceil(sum(sizes) / divisor) * divisor


def ple_state_shape(config: Qwen4ExpTextConfig) -> tuple[int, int]:
    return (
        (config.ple_conv_kernel_size - 1) * config.ngram_size,
        config.hidden_size * config.hc_count,
    )
