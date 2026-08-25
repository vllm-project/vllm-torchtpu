# SPDX-License-Identifier: Apache-2.0

from math import prod
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig

from vllm_torchtpu.gdn_state_declaration import (gdn_state_declaration_applies,
                                                 gdn_state_dtypes,
                                                 gdn_state_shapes)

POOLED_ARCH = "Qwen3_5MoeForConditionalGeneration"


class FakeGDNModel:
    """Conv (state_len, dim) in SD layout, SSM (v_heads, head_v, head_k).

    Its dtypes are deliberately not the declared ones, so a declaration that
    forwards the model class's view shows up as a failure.
    """

    @staticmethod
    def get_mamba_state_shape_from_config(vllm_config):
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config else 0)
        return ((3 + num_spec, 4096 // tp_size), (16 // tp_size, 128, 128))

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.bfloat16)


@pytest.fixture
def vllm_config():
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = MagicMock(spec=ModelConfig)
    vllm_config.model_config.architecture = POOLED_ARCH
    vllm_config.cache_config = MagicMock(spec=CacheConfig)
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.speculative_config = None
    return vllm_config


def _declare(vllm_config, model_cls=FakeGDNModel):
    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(model_cls, None)):
        return gdn_state_shapes(vllm_config), gdn_state_dtypes(vllm_config)


def test_applies_only_to_the_gdn_architectures(vllm_config):
    assert gdn_state_declaration_applies(vllm_config)
    vllm_config.model_config.architecture = "KimiLinearForCausalLM"
    assert not gdn_state_declaration_applies(vllm_config)


def test_state_dtypes_are_pinned_f32(vllm_config, monkeypatch):
    _, pooled = _declare(vllm_config)
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    _, per_layer = _declare(vllm_config)

    # One answer for both layouts, and never the model class's (bf16, bf16).
    assert pooled == per_layer == (torch.float32, torch.float32)


def test_conv_state_gains_the_kernel_layout_axis(vllm_config):
    shapes, _ = _declare(vllm_config)

    assert shapes[0] == (3, 1, 4096)
    assert shapes[1] == (16, 128, 128)


def test_spec_decode_widens_the_conv_state(vllm_config):
    vllm_config.speculative_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 2

    shapes, _ = _declare(vllm_config)

    assert shapes[0] == (5, 1, 4096)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("pcp_size", [1, 2, 4])
def test_pcp_shards_the_state_like_another_tp_factor(vllm_config, tp_size,
                                                     pcp_size):
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size

    shapes, _ = _declare(vllm_config)

    # Identical to asking the shape calculator for tp_size * pcp_size, which
    # is how the block-slot derivation used to reach the same numbers.
    assert prod(shapes[0]) == 3 * 4096 // (tp_size * pcp_size)
    assert shapes[1] == (16 // (tp_size * pcp_size), 128, 128)


@pytest.mark.parametrize(
    ("tp_size", "pcp_size"),
    [
        (4, 3),  # 1024 conv channels do not divide by 3
        (4, 8),  # 4096/4 conv channels do, but the 4 SSM heads do not
    ])
def test_pcp_rejects_unshardable_state(vllm_config, tp_size, pcp_size):
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size

    with pytest.raises(ValueError, match="divisible by PCP size"):
        _declare(vllm_config)
