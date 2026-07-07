# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig

from vllm_torchtpu.platforms.tpu_block_size_utils import \
    update_tpu_block_size_and_slot_config


class FakeBatchedRPAAttentionBackend:

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_min_page_size(_vllm_config):
        return 16

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [256]

    @staticmethod
    def get_kv_cache_page_size_bytes(block_size, *_args, **_kwargs):
        return block_size * 1024


class FakeQwenMambaModel:

    @staticmethod
    def get_mamba_state_shape_from_config(_):
        return ((3, 4096), (16, 128, 128))

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.float32)


@pytest.fixture
def vllm_config():
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = MagicMock(spec=ModelConfig)
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.model_config.is_hybrid = False
    vllm_config.model_config.get_num_kv_heads.return_value = 1
    vllm_config.model_config.get_head_size.return_value = 256
    vllm_config.cache_config = MagicMock(spec=CacheConfig)
    vllm_config.cache_config.block_size = None
    vllm_config.cache_config.mamba_cache_mode = None
    vllm_config.cache_config.mamba_page_size_padded = None
    vllm_config.cache_config.mamba_block_size = None
    vllm_config.cache_config.cache_dtype = "auto"
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = 1
    return vllm_config


def _format_logs(mock_logger_info):
    return "\n".join(call.args[0] %
                     call.args[1:] if len(call.args) > 1 else call.args[0]
                     for call in mock_logger_info.call_args_list)


def test_custom_backend_does_not_lower_non_hybrid_block_size(vllm_config):
    vllm_config.cache_config.block_size = 2112

    update_tpu_block_size_and_slot_config(vllm_config,
                                          FakeBatchedRPAAttentionBackend)

    assert vllm_config.cache_config.block_size == 2112


def test_hybrid_mamba_state_drives_power2_block_size_and_slot_logs(
        vllm_config):
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.cache_config.block_size = 2112
    vllm_config.cache_config.mamba_block_size = 2112
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.mamba_page_size_padded = None
    vllm_config.cache_config.cache_dtype = "fp8"

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)), patch(
                   "vllm_torchtpu.platforms.tpu_block_size_utils.logger.info"
               ) as mock_logger_info:
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)

    assert vllm_config.cache_config.block_size == 1024
    assert vllm_config.cache_config.mamba_block_size == 1024
    assert vllm_config.cache_config.mamba_page_size_padded == 1073152

    logs = _format_logs(mock_logger_info)
    assert "TPU block_size derivation path" in logs
    assert "backend=CUSTOM -> backend_supported_kernel_block_sizes=[256]" in logs
    assert "-> input_block_size=2112" in logs
    assert "backend_kernel_block_multiple" not in logs
    assert "backend_lowering" not in logs
    assert "mamba_raw_state_bytes=1073152" in logs
    assert ("-> fa_bytes_per_token(raw_payload=512 + layout_padding=512 "
            "-> physical=1024)" in logs)
    assert ("-> mamba_fit_block_size=ceil(1073152 / 1024) "
            "rounded_to_16 = 1056" in logs)
    assert "-> power2_lowering=floor_power2(1056) = 1024" in logs
    assert "-> final_block_size=1024" in logs
    assert "TPU block_slot derivation path" in logs
    assert "final_block_size=1024 -> fa_raw_payload_slot_bytes=524288" in logs
    assert "-> fa_layout_padding_slot_bytes=524288" in logs
    assert "-> fa_physical_slot_bytes=1048576" in logs
    assert ("-> slot_base_bytes=max(mamba_raw_state_bytes=1073152, "
            "fa_physical_slot_bytes=1048576) = 1073152" in logs)
    assert ("-> slot_alignment_bytes=16 -> "
            "final_block_slot_bytes=round_up(1073152, 16) = 1073152" in logs)
    assert "-> mamba_slot_padding_bytes=0" in logs
    assert "-> fa_slot_tail_padding_bytes=24576" in logs
    assert "kernel_blocks_per_logical_block" not in logs
