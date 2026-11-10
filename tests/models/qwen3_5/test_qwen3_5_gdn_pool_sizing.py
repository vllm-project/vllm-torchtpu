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
"""Main's unified-pool block sizing (platforms/tpu_block_size_utils.py) for
the Qwen3.5-397B GDN geometry at the IX serving layouts.

The pool keeps the GDN conv state in float32 and the SSM state in the dtype
the layer declares, which follows vLLM's `--mamba-ssm-cache-dtype` (Qwen3.5
checkpoints declare float32; bfloat16 is an explicit opt-in). One manager
block holds a request's whole GDN state, so the block can be no smaller than
`ssm tokens + conv tokens`, and a user `--block-size` below that is raised.
The real vLLM Qwen3.5 model class supplies the state shapes and dtypes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

from vllm_torchtpu.platforms.tpu_block_size_utils import (
    _hybrid_mamba_state_layout,
    update_tpu_block_size_and_slot_config,
)

ARCH = "Qwen3_5MoeForConditionalGeneration"
# One fp8 KV token row of a full-attention layer: 2 KV heads (1 per rank,
# padded to the fp8 packing of 4) x head_dim 256 = 1024 bytes at TP1, TP2 and
# TP8 alike.
TOKEN_BYTES = 1024


class _RpaBackend:
    """The batched RPA backend's sizing at the 397B fp8 KV shape."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_min_page_size(_vllm_config):
        return 64

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128, 256]

    @staticmethod
    def get_kv_cache_page_size_bytes(block_size, *_args, **_kwargs):
        return block_size * TOKEN_BYTES


def _config(*, tp: int, num_spec: int, ssm_dtype: str, block_size: int):
    vllm_config = MagicMock(spec=VllmConfig)
    model_config = MagicMock(spec=ModelConfig)
    model_config.dtype = torch.bfloat16
    model_config.is_hybrid = True
    model_config.architecture = ARCH
    model_config.hf_text_config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    model_config.get_num_kv_heads.return_value = max(1, 2 // tp)
    model_config.get_head_size.return_value = 256
    vllm_config.model_config = model_config
    cache_config = MagicMock(spec=CacheConfig)
    cache_config.block_size = block_size
    cache_config.user_specified_block_size = True
    cache_config.mamba_block_size = block_size
    cache_config.mamba_cache_mode = "align"
    cache_config.mamba_page_size_padded = None
    cache_config.cache_dtype = "fp8"
    cache_config.mamba_cache_dtype = "auto"
    cache_config.mamba_ssm_cache_dtype = ssm_dtype
    vllm_config.cache_config = cache_config
    vllm_config.kv_transfer_config = None
    vllm_config.speculative_config = SimpleNamespace(num_speculative_tokens=num_spec)
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = tp
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    return vllm_config


def _resolve():
    return patch(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        return_value=(Qwen3_5MoeForConditionalGeneration, None),
    )


@pytest.mark.parametrize(
    ("tp", "num_spec", "ssm_dtype", "ssm_tokens", "conv_tokens"),
    [
        # Prefill legs: TP1, MTP K = 1 widens the conv state to 4 taps.
        (1, 1, "bfloat16", 2048, 256),
        (1, 1, "float32", 4096, 256),
        # #1 decode (DP4 x TP2) and #5-#12 decode (TP8): MTP K = 3.
        (2, 3, "bfloat16", 1024, 256),
        (2, 3, "float32", 2048, 256),
        (8, 3, "bfloat16", 256, 64),
        (8, 3, "float32", 512, 64),
    ],
)
def test_the_ssm_region_follows_the_declared_ssm_dtype(
    tp, num_spec, ssm_dtype, ssm_tokens, conv_tokens
):
    config = _config(tp=tp, num_spec=num_spec, ssm_dtype=ssm_dtype, block_size=16)
    with _resolve():
        layout = _hybrid_mamba_state_layout(config, TOKEN_BYTES)
    assert layout is not None
    assert (layout.ssm_tokens, layout.conv_tokens) == (ssm_tokens, conv_tokens)


@pytest.mark.parametrize(
    ("tp", "num_spec", "ssm_dtype", "user_block", "final_block"),
    [
        # The record's --block-size values under main's fp32 conv state.
        (1, 1, "bfloat16", 2176, 2304),
        (2, 3, "bfloat16", 640, 1280),
        (8, 3, "bfloat16", 640, 640),
        # The values the validation entries pass on the main-based stack.
        (1, 1, "bfloat16", 2304, 2304),
        (2, 3, "bfloat16", 1280, 1280),
        # Without the bf16 opt-in (main's default float32 SSM state).
        (1, 1, "float32", 2304, 4352),
        (2, 3, "float32", 1280, 2304),
        (8, 3, "float32", 640, 640),
    ],
)
def test_the_block_holds_the_whole_gdn_state(
    tp, num_spec, ssm_dtype, user_block, final_block
):
    config = _config(
        tp=tp, num_spec=num_spec, ssm_dtype=ssm_dtype, block_size=user_block
    )
    with _resolve():
        update_tpu_block_size_and_slot_config(config, _RpaBackend)
    assert config.cache_config.block_size == final_block
