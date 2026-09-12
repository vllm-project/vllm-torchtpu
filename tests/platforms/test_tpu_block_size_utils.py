# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig

from vllm_torchtpu.platforms.tpu_block_size_utils import \
    update_tpu_block_size_and_slot_config

pytestmark = pytest.mark.cpu_test


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


class FakePlainAttentionBackend(FakeBatchedRPAAttentionBackend):

    @staticmethod
    def get_name():
        return "PALLAS"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return []


class FakeLargeMinimumBackend(FakeBatchedRPAAttentionBackend):

    @staticmethod
    def get_min_page_size(_vllm_config):
        return 1536


class FakeNonLinearPageBackend(FakeBatchedRPAAttentionBackend):

    @staticmethod
    def get_kv_cache_page_size_bytes(block_size, *_args, **_kwargs):
        tail_bytes = 16 if block_size > 1 else 0
        return block_size * 1024 + tail_bytes


class FakeMisalignedSlotBackend(FakePlainAttentionBackend):

    @staticmethod
    def get_kv_cache_page_size_bytes(block_size, *_args, **_kwargs):
        return block_size * 1025


class FakeQwenMambaModel:

    @staticmethod
    def get_mamba_state_shape_from_config(vllm_config):
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return ((3, 4096 // tp_size), (16 // tp_size, 128, 128))

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.float32)


class FakeCompactMambaModel:

    @staticmethod
    def get_mamba_state_shape_from_config(_):
        # 1 physical conv row followed by 256 physical SSM rows.
        return ((1, 512), (2, 128, 256))

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.float32)


class FakeBf16StateMambaModel(FakeQwenMambaModel):

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.bfloat16)


class FakeNonShardingMambaModel(FakeQwenMambaModel):

    @staticmethod
    def get_mamba_state_shape_from_config(_):
        return ((3, 4096), (16, 128, 128))


class FakeKimiLinearModel:
    """Kimi-Linear KDA state per TP shard: conv (taps, qkv, heads, head_dim)
    bf16 and recurrent (heads, head_dim, head_dim) fp32."""

    @staticmethod
    def get_mamba_state_shape_from_config(vllm_config):
        local_heads = (32 // vllm_config.parallel_config.tensor_parallel_size)
        return ((3, 3, local_heads, 128), (local_heads, 128, 128))

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        return (torch.bfloat16, torch.float32)


class FakeKimiMLABackend(FakePlainAttentionBackend):
    """MLA page geometry: the pool packs the token axis (bf16 packing 2), so
    a pool row is a 2560 B packed pair; a 1-token page already takes a whole
    row. The real PallasMLAttentionBackend inherits MultipleOf(1) kernel
    block sizes, so the derivation's alignment step is a no-op ([] matches)."""

    @staticmethod
    def get_name():
        return "PALLAS_MLA"

    @staticmethod
    def get_min_page_size(_vllm_config):
        # Representative for max_model_len=32768, max_num_seqs=256.
        return 64

    @staticmethod
    def get_kv_cache_page_size_bytes(block_size, *_args, **_kwargs):
        return ((block_size + 1) // 2) * 2560


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


def _configure_hybrid(vllm_config,
                      *,
                      block_size=256,
                      user_specified=False,
                      tp_size=1,
                      kv_transfer_config=None):
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration")
    vllm_config.cache_config.block_size = block_size
    vllm_config.cache_config.user_specified_block_size = user_specified
    vllm_config.cache_config.mamba_block_size = block_size
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.cache_dtype = "fp8"
    vllm_config.kv_transfer_config = kv_transfer_config
    vllm_config.parallel_config.tensor_parallel_size = tp_size


def _update(vllm_config,
            backend_cls=FakeBatchedRPAAttentionBackend,
            model_cls=FakeQwenMambaModel):
    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls",
               return_value=(model_cls, None)):
        update_tpu_block_size_and_slot_config(vllm_config, backend_cls)


def test_custom_backend_does_not_lower_non_hybrid_block_size(vllm_config):
    vllm_config.cache_config.block_size = 2112

    update_tpu_block_size_and_slot_config(vllm_config,
                                          FakeBatchedRPAAttentionBackend)

    assert vllm_config.cache_config.block_size == 2112


def test_cache_slot_requires_16_byte_alignment(vllm_config):
    vllm_config.cache_config.block_size = 17

    with pytest.raises(AssertionError):
        update_tpu_block_size_and_slot_config(vllm_config,
                                              FakeMisalignedSlotBackend)


def test_hybrid_fit_uses_physical_gdn_extent(vllm_config):
    _configure_hybrid(vllm_config)

    _update(vllm_config)

    # SSM=1024 rows, raw conv=24 rows, tiled conv=32 rows.
    assert vllm_config.cache_config.block_size == 1280
    assert vllm_config.cache_config.mamba_block_size == 1280
    assert vllm_config.cache_config.mamba_page_size_padded == 1280 * 1024


def test_hybrid_fit_uses_configured_bf16_ssm_dtype(vllm_config):
    _configure_hybrid(vllm_config)

    _update(vllm_config, model_cls=FakeBf16StateMambaModel)

    # BF16 halves the SSM extent from 1024 to 512 rows. With 32 Conv rows,
    # the 544-row fit is aligned to this backend's 256-token block size.
    assert vllm_config.cache_config.block_size == 768
    assert vllm_config.cache_config.mamba_page_size_padded == 768 * 1024


def test_kimi_linear_padded_ssm_fit(vllm_config):
    """Kimi-Linear TP8: the f32 KDA state (2^18 B) does not divide the MLA
    pool row (2^9 * 5 B), so the derivation pads the SSM region up to whole
    rows; the pool's token-axis packing (2 tokens/row) then doubles the
    manager block the fit floor implies."""
    vllm_config.model_config.is_hybrid = True
    vllm_config.model_config.architecture = "KimiLinearForCausalLM"
    vllm_config.model_config.get_head_size.return_value = 576
    vllm_config.cache_config.block_size = 16
    vllm_config.cache_config.mamba_block_size = 16
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.parallel_config.tensor_parallel_size = 8

    _update(vllm_config,
            backend_cls=FakeKimiMLABackend,
            model_cls=FakeKimiLinearModel)

    # State regions: ceil(262144 / 2560) = 103 pool rows + pow2(ceil(9216 /
    # 2560)) = 4 -> 107 packed rows = 214 tokens at packing 2. The fit floor
    # 214 beats the backend minimum of 64 and the input block size of 16.
    assert vllm_config.cache_config.block_size == 214
    assert vllm_config.cache_config.mamba_block_size == 214
    assert vllm_config.cache_config.mamba_page_size_padded == 107 * 2560


def test_tp_sharding_changes_the_physical_gdn_fit(vllm_config):
    _configure_hybrid(vllm_config, tp_size=2)

    _update(vllm_config)

    # SSM=512 rows, raw conv=12 rows, tiled conv=16 rows.
    assert vllm_config.cache_config.block_size == 768
    assert vllm_config.cache_config.mamba_page_size_padded == 768 * 1024


def test_fit_has_no_fixed_16_token_alignment(vllm_config):
    _configure_hybrid(vllm_config)

    _update(vllm_config,
            backend_cls=FakePlainAttentionBackend,
            model_cls=FakeCompactMambaModel)

    assert vllm_config.cache_config.block_size == 257
    assert vllm_config.cache_config.mamba_page_size_padded == 257 * 1024


def test_user_block_size_below_fit_is_promoted(vllm_config):
    _configure_hybrid(vllm_config, block_size=1024, user_specified=True)

    _update(vllm_config)

    assert vllm_config.cache_config.block_size == 1280
    assert vllm_config.cache_config.mamba_page_size_padded == 1280 * 1024


def test_decode_explicit_nested_block_is_honored(vllm_config):
    _configure_hybrid(vllm_config, block_size=1536, user_specified=True)

    _update(vllm_config)

    assert vllm_config.cache_config.block_size == 1536
    assert vllm_config.cache_config.mamba_block_size == 1536
    assert vllm_config.cache_config.mamba_page_size_padded == 1536 * 1024


def test_disagg_uses_same_fit_as_single_server(vllm_config):
    _configure_hybrid(vllm_config, kv_transfer_config=object())

    _update(vllm_config)

    assert vllm_config.cache_config.block_size == 1280
    assert vllm_config.cache_config.mamba_page_size_padded == 1280 * 1024


def test_pcp_uses_effective_tp_mamba_state(vllm_config):
    _configure_hybrid(vllm_config, kv_transfer_config=object())
    vllm_config.parallel_config.prefill_context_parallel_size = 4

    _update(vllm_config)

    # Effective TP = 1 * 4 = 4. Fit = 272 -> aligned to 256 kernel block is 512.
    assert vllm_config.cache_config.block_size == 512
    assert vllm_config.cache_config.mamba_page_size_padded == 512 * 1024


def test_backend_minimum_is_part_of_the_block_floor(vllm_config):
    _configure_hybrid(vllm_config)

    _update(vllm_config, backend_cls=FakeLargeMinimumBackend)

    assert vllm_config.cache_config.block_size == 1536
    assert vllm_config.cache_config.mamba_page_size_padded == 1536 * 1024


def test_unified_slot_rejects_non_compact_fa_pages(vllm_config):
    _configure_hybrid(vllm_config)

    with pytest.raises(ValueError, match="FA page is not token-linear"):
        _update(vllm_config, backend_cls=FakeNonLinearPageBackend)


def test_spec_ckpt_blocks_keep_block_size_independent_of_num_spec(
        vllm_config, monkeypatch):
    """One block per checkpoint means the fit floor stops scaling with K.

    This is what the unified pool does for every speculative config: the
    checkpoints live in `num_spec` extra ordinary blocks, so a block only
    has to fit a single state and the floor is the no-spec one for every K.
    The affine test above still lands on 2560 for K=1 because it leaves the
    unified pool off, which is the only remaining user of that addressing.
    """
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    sizes = {}
    for num_spec in (1, 3, 7):
        vllm_config.model_config.is_hybrid = True
        vllm_config.model_config.architecture = (
            "Qwen3_5MoeForConditionalGeneration")
        vllm_config.cache_config.block_size = 2112
        vllm_config.cache_config.mamba_block_size = 2112
        vllm_config.cache_config.mamba_cache_mode = "align"
        vllm_config.cache_config.mamba_page_size_padded = None
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.num_speculative_tokens = num_spec

        with patch(
                "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
                return_value=(FakeQwenMambaModel, None)):
            update_tpu_block_size_and_slot_config(
                vllm_config, FakeBatchedRPAAttentionBackend)
        sizes[num_spec] = vllm_config.cache_config.block_size

    assert len(set(sizes.values())) == 1, sizes
    # One checkpoint's 1056 tokens padded to this backend's 256-token
    # kernel block = 1280, versus 2560 for K=1 on the affine path.
    assert sizes[1] == 1280, sizes


def test_hybrid_mode_none_still_sizes_the_envelope_slot(vllm_config):
    _configure_hybrid(vllm_config)
    vllm_config.cache_config.mamba_cache_mode = "none"

    _update(vllm_config)

    assert vllm_config.cache_config.block_size == 1280
    assert vllm_config.cache_config.mamba_block_size == 256
    assert vllm_config.cache_config.mamba_page_size_padded == 1280 * 1024


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
