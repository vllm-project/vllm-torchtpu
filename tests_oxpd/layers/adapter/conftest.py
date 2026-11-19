# SPDX-License-Identifier: Apache-2.0
"""Real GDN construction shared by loader and TPU computation tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.distributed as distributed_mod
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter as parameter_mod
from vllm.model_executor.layers import linear as linear_mod
from vllm.model_executor.layers.mamba.gdn import base as gdn_base
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.model_loader import weight_utils
from vllm.platforms import current_platform

from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture(autouse=True)
def _empty_shared_gdn_ops(monkeypatch):
    """Keep an op built from one test's mocks out of every other test."""
    monkeypatch.setattr(gdn_attention_op, "_shared_gdn_ops", {})


@pytest.fixture
def make_gdn_attention(monkeypatch):
    """Keep construction/loading real while avoiding accelerator ownership."""
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    monkeypatch.setattr(current_platform, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(gdn_attention_op.pallas, "jax_op", MagicMock())

    # TPU's FP8 method changes this module-level kernel factory in __init__.
    # Restore it at teardown so the test does not leak that production side
    # effect into other test modules.
    from vllm.model_executor.layers.quantization import fp8

    monkeypatch.setattr(fp8, "init_fp8_linear_kernel", fp8.init_fp8_linear_kernel)

    def make(
        tp_size,
        tp_rank,
        quantization="bf16",
        activation_scheme="dynamic",
        gqa_interleaved_layout=False,
        num_key_heads=4,
        num_value_heads=32,
        hidden_size=128,
    ):
        monkeypatch.setattr(
            weight_utils, "get_tensor_model_parallel_rank", lambda: tp_rank
        )
        for module in (gdn_base, linear_mod, parameter_mod, distributed_mod):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_rank", lambda: tp_rank
            )
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_world_size", lambda: tp_size
            )

        config = SimpleNamespace(
            hidden_size=hidden_size,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            linear_num_key_heads=num_key_heads,
            linear_num_value_heads=num_value_heads,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        # CPU CI does not register torch's TPU device type. Loading these
        # weights only needs CPU tensors, including for the SPMD test setup.
        vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
        vllm_config.model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            architecture="Qwen3_5MoeForCausalLM",
            hf_text_config=config,
            is_hybrid=True,
        )
        vllm_config.parallel_config.tensor_parallel_size = tp_size
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.mamba_block_size = 4096
        vllm_config.cache_config.mamba_cache_dtype = "bfloat16"
        vllm_config.cache_config.mamba_ssm_cache_dtype = "float32"
        if quantization != "bf16":
            vllm_config.quant_config = VllmFp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme=activation_scheme,
                weight_block_size=([128, 128] if quantization == "fp8_block" else None),
                ignored_layers=["model.layers.0.linear_attn.in_proj_ba"],
            )
            vllm_config.quant_config.is_channel_quant = quantization == "fp8_channel"

        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with (
                torch.device("cpu"),
                set_current_vllm_config(vllm_config),
                set_vllm_model_wrapper_context(
                    mesh=SimpleNamespace(shape={"model": 1}), vllm_config=vllm_config
                ),
            ):
                # Go through vLLM's OOT dispatch, as model construction does.
                attention = QwenGatedDeltaNetAttention(
                    config=config,
                    vllm_config=vllm_config,
                    prefix="model.layers.0.linear_attn",
                    gqa_interleaved_layout=gqa_interleaved_layout,
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        assert isinstance(attention, VllmGatedDeltaNetAttention)
        return attention, vllm_config

    return make
