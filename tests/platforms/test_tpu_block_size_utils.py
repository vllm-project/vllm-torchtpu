# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig

from vllm_torchtpu.platforms.tpu_block_size_utils import (
    _ceil_power_of_two, update_tpu_block_size_and_slot_config)


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
    def get_mamba_state_shape_from_config(vllm_config):
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return ((3, 4096 // tp_size), (16 // tp_size, 128, 128))

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.float32)


class FakeNonShardingMambaModel(FakeQwenMambaModel):

    @staticmethod
    def get_mamba_state_shape_from_config(_):
        return ((3, 4096), (16, 128, 128))


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
    vllm_config.cache_config.user_specified_block_size = False
    vllm_config.cache_config.mamba_cache_mode = None
    vllm_config.cache_config.mamba_page_size_padded = None
    vllm_config.cache_config.mamba_block_size = None
    vllm_config.cache_config.cache_dtype = "auto"
    vllm_config.kv_transfer_config = None
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
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


def test_hybrid_mamba_state_drives_fit_block_size_and_slot_logs(vllm_config):
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

    assert vllm_config.cache_config.block_size == 1280
    assert vllm_config.cache_config.mamba_block_size == 1280
    assert vllm_config.cache_config.mamba_page_size_padded == 1310720

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
    assert "-> final_block_size=1280" in logs
    assert "(source=mamba_state_fit)" in logs
    assert "TPU block_slot derivation path" in logs
    assert "final_block_size=1280 -> fa_raw_payload_slot_bytes=655360" in logs
    assert "-> fa_layout_padding_slot_bytes=655360" in logs
    assert "-> fa_physical_slot_bytes=1310720" in logs
    assert ("-> slot_base_bytes=max(mamba_raw_state_bytes=1073152, "
            "fa_physical_slot_bytes=1310720) = 1310720" in logs)
    assert ("-> slot_alignment_bytes=16 -> "
            "final_block_slot_bytes=round_up(1310720, 16) = 1310720" in logs)
    assert "-> mamba_slot_padding_bytes=237568" in logs
    assert "-> fa_slot_tail_padding_bytes=0" in logs
    assert "kernel_blocks_per_logical_block" not in logs


def test_hybrid_mode_none_still_sizes_the_envelope_slot(vllm_config):
    # The unified pool holds mamba state in every cache mode; only the
    # block-size retarget is align-specific.
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.model_config.get_head_size.return_value = 128
    vllm_config.cache_config.block_size = 256
    vllm_config.cache_config.mamba_block_size = 256
    vllm_config.cache_config.mamba_cache_mode = "none"

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)):
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)

    assert vllm_config.cache_config.block_size == 1280
    assert vllm_config.cache_config.mamba_block_size == 256
    assert vllm_config.cache_config.mamba_page_size_padded == 1310720


def test_user_block_size_at_or_above_fit_is_honored(vllm_config):
    # The disaggregated launch passes one explicit block size to both
    # roles; the fit size is a floor, not a mandate, so the user's choice
    # wins (aligned up to the backend's kernel block when needed).
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.cache_config.block_size = 2112
    vllm_config.cache_config.user_specified_block_size = True
    vllm_config.cache_config.mamba_block_size = 2112
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.cache_dtype = "fp8"

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)), patch(
                   "vllm_torchtpu.platforms.tpu_block_size_utils.logger.info"
               ) as mock_logger_info:
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)

    # 2112 contains the 1056-token slot; aligned up to the 256-token
    # kernel block: 2304.
    assert vllm_config.cache_config.block_size == 2304
    assert vllm_config.cache_config.mamba_block_size == 2304
    logs = _format_logs(mock_logger_info)
    assert "(source=user_block_size)" in logs


def test_user_block_size_below_fit_is_rejected_for_kv_transfer(vllm_config):
    # Unified P/D uses attention-shaped physical pages for Mamba state, so an
    # explicit block size below the fit floor cannot be honored safely.
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.kv_transfer_config = object()
    vllm_config.cache_config.block_size = 1024
    vllm_config.cache_config.user_specified_block_size = True
    vllm_config.cache_config.mamba_block_size = 1024
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.cache_dtype = "fp8"

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)), pytest.raises(
                   ValueError, match="below Mamba fit floor"):
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)


def test_hybrid_gdn_pcp_uses_effective_tp_for_block_slot(vllm_config):
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.model_config.hf_text_config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    vllm_config.cache_config.block_size = 2112
    vllm_config.cache_config.mamba_block_size = 2112
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.mamba_page_size_padded = None
    vllm_config.cache_config.cache_dtype = "fp8"
    vllm_config.parallel_config.prefill_context_parallel_size = 4

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)), patch(
                   "vllm_torchtpu.platforms.tpu_block_size_utils.logger.info"
               ) as mock_logger_info:
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)

    assert vllm_config.parallel_config.tensor_parallel_size == 1
    assert vllm_config.cache_config.block_size == 512
    assert vllm_config.cache_config.mamba_block_size == 512
    assert vllm_config.cache_config.mamba_page_size_padded == 524288

    logs = _format_logs(mock_logger_info)
    assert "mamba_raw_state_bytes=268288" in logs
    assert ("mamba_fit_block_size=ceil(268288 / 1024) "
            "rounded_to_16 = 272") in logs
    assert "final_block_size=512 (source=mamba_state_fit)" in logs
    assert "final_block_slot_bytes=round_up(524288, 16) = 524288" in logs


def test_hybrid_gdn_pcp_rejects_non_sharding_shape_calculator(vllm_config):
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = "FutureHybridForCausalLM"
    vllm_config.cache_config.block_size = 256
    vllm_config.cache_config.mamba_block_size = 256
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.parallel_config.prefill_context_parallel_size = 4

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeNonShardingMambaModel, None)), pytest.raises(
                   ValueError,
                   match="PCP-local Mamba state size must be exactly"):
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)


class FakePlainAttentionBackend(FakeBatchedRPAAttentionBackend):

    @staticmethod
    def get_name():
        return "PALLAS"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return []


def test_user_block_size_below_fit_is_rejected(vllm_config):
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.cache_config.block_size = 512
    vllm_config.cache_config.user_specified_block_size = True
    vllm_config.cache_config.mamba_block_size = 512
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.cache_dtype = "fp8"

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)), pytest.raises(
                   ValueError, match="below Mamba fit floor"):
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakePlainAttentionBackend)


def test_disagg_fit_block_size_rounds_up_to_power_of_two(vllm_config):
    # Disaggregated P/D: the fit size is rounded up to a power of two so the
    # prefill/decode block sizes nest for the KV connector.
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.cache_config.block_size = 256
    vllm_config.cache_config.mamba_block_size = 256
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.cache_dtype = "fp8"
    vllm_config.kv_transfer_config = MagicMock()

    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(FakeQwenMambaModel, None)), patch(
                   "vllm_torchtpu.platforms.tpu_block_size_utils.logger.info"
               ) as mock_logger_info:
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeBatchedRPAAttentionBackend)

    # fit 1056 -> aligned 1280 -> next power of two 2048.
    assert vllm_config.cache_config.block_size == 2048
    logs = _format_logs(mock_logger_info)
    assert "(source=mamba_state_fit_pow2)" in logs


def test_ceil_power_of_two_outputs_pairwise_nest():
    # Any two power-of-two block sizes nest (larger % smaller == 0), which is
    # what lets independently-derived per-TP disagg block sizes satisfy the
    # connector's divisibility constraint.
    outs = sorted({_ceil_power_of_two(f) for f in range(1, 4097)})
    for x in outs:
        assert x & (x - 1) == 0  # power of two
    for i in range(len(outs) - 1):
        assert outs[i + 1] % outs[i] == 0
