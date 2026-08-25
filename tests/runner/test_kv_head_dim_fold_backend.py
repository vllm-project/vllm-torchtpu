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
"""`TPU_RPA_FOLD_KV_HEAD_DIM` has to reach the vLLM-side page accounting too.

A half-applied flag is the failure mode that matters: the kernel would read a
512 B/token cache with a 1024 B/token stride, or the pool would be sized for a
page the backend never allocates. Both are silent.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import \
    wrapper as longctx_wrapper
from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.vllm import attention as vllm_attention
from vllm_torchtpu.layers.vllm.attention import (
    PallasAttentionBackend, PallasBatchedRPAAttentionBackend)
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    _align_block_to_backend
from vllm_torchtpu.platforms.tpu_platform import (
    _refuse_kv_head_dim_fold, _reject_seq_along_lane_unified_pool)


class _FakeTpuInfo:
    num_lanes = 128
    num_sublanes = 8


@pytest.fixture(autouse=True)
def _deviceless_tpu_info(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _FakeTpuInfo())
    # `attention` picks the wrapper module at import time from
    # USE_BATCHED_RPA_LONGCTX, so setting the env afterwards is not enough.
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_LONGCTX", True)
    monkeypatch.setattr(vllm_attention, "rpa_batched_wrapper", longctx_wrapper)


@pytest.fixture
def longctx_fold(monkeypatch):
    monkeypatch.setattr(envs, "TPU_RPA_FOLD_KV_HEAD_DIM", True)


def test_default_is_the_current_layout(monkeypatch):
    monkeypatch.setattr(envs, "TPU_RPA_FOLD_KV_HEAD_DIM", False)
    assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        4, 256, 1, 256, torch.float8_e4m3fn) == (4, 256, 1, 4, 256)


def test_backend_shape_and_page_bytes_move_together(longctx_fold):
    shape = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        4, 256, 1, 256, torch.float8_e4m3fn)
    assert shape == (4, 256, 1, 4, 128)

    page_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 256, torch.float8_e4m3fn)
    # This is what `fa_bytes_per_token` reports, and what the GDN pool sizes
    # its state regions against.
    assert page_bytes == 256 * 512
    assert page_bytes * 2 == PallasAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 256, torch.float8_e4m3fn)


def test_dp_rank_page_is_unchanged(longctx_fold):
    """Two KV heads already fill the packed word; the flag must not move them."""
    assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        4, 256, 2, 256, torch.float8_e4m3fn) == (4, 256, 1, 4, 256)


def test_spec_normalizer_follows_the_selected_backend(longctx_fold):
    """The pool is padded to the page the *selected* backend allocates.

    `allocate_raw_kv_cache_tensors` asserts `fa_page_bytes * split ==
    pool_page_bytes`; padding the spec with `PallasAttentionBackend` while the
    batched backend allocates a folded page breaks that assert (and, before it
    existed, silently over-allocated).
    """
    spec = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    normalized = normalize_kv_cache_specs_for_tpu(
        {"attn": spec},
        torch.float8_e4m3fn,
        attn_backend_cls=PallasBatchedRPAAttentionBackend,
    )
    assert normalized["attn"].page_size_bytes == (
        PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
            256, 1, 256, torch.float8_e4m3fn))


# ---------------------------------------------------------------------------
# The block the GDN state fit asks for, and why this needs power-of-two kernel
# blocks (PR #540).
# ---------------------------------------------------------------------------

# Back-solved from the Qwen3.5-397B TP8 startup line
# `fa_bytes_per_token(... physical=1024) -> mamba_fit_block_size=4224`:
# 4224 = 4096 SSM tokens + 128 conv tokens at 1024 B/token.
_397B_SSM_BYTES = 4096 * 1024
_397B_CONV_BYTES = 100 * 1024


@pytest.mark.parametrize(
    "token_bytes, expected_fit",
    [(1024, 4224), (512, 8448)],
    ids=["unfolded", "folded"],
)
def test_fold_doubles_the_gdn_state_fit_block(token_bytes, expected_fit):
    """Halving bytes/token doubles how many tokens the same state spans."""
    layout = derive_pooled_gdn_state_layout(ssm_bytes=_397B_SSM_BYTES,
                                            conv_bytes=_397B_CONV_BYTES,
                                            token_bytes=token_bytes)
    assert layout.required_tokens == expected_fit


def test_folded_fit_block_needs_a_power_of_two_kernel_block():
    """8448 = 2**8 * 33, so it cannot itself be the kernel block.

    The kernel addresses pages with `page_size_log2` shifts. Under
    `MultipleOf(128)` vLLM would accept 8448 as an exact kernel block and
    mis-address every page; the explicit power-of-two list leaves 128 and 256
    as the only kernel blocks that divide it.
    """
    supported = PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes(
    )
    manager_block = _align_block_to_backend(8448, supported)
    assert manager_block == 8448
    divisors = [b for b in supported if manager_block % b == 0]
    assert divisors, "no supported kernel block divides the folded fit block"
    for block in divisors:
        assert block & (block - 1) == 0, block


# ---------------------------------------------------------------------------
# The refusal boundary. Every combination that cannot read a folded page has to
# turn the flag off at config time, because nothing downstream would notice.
# ---------------------------------------------------------------------------


def _fake_config(**overrides):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        speculative_config=None,
        kv_transfer_config=None,
        model_config=SimpleNamespace(architecture="Qwen3NextForCausalLM"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def fold_env(monkeypatch):
    monkeypatch.setattr(envs, "TPU_RPA_FOLD_KV_HEAD_DIM", True)
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_SEQ_ON_LANE", False)
    monkeypatch.setenv("TPU_RPA_FOLD_KV_HEAD_DIM", "1")


def test_fold_survives_a_supported_config(fold_env):
    _refuse_kv_head_dim_fold(_fake_config())
    assert os.environ["TPU_RPA_FOLD_KV_HEAD_DIM"] == "1"


@pytest.mark.parametrize(
    "attr, overrides",
    [
        ("USE_BATCHED_RPA_LONGCTX", {}),
        ("USE_BATCHED_RPA_SEQ_ON_LANE", {}),
        (None, {
            "parallel_config": SimpleNamespace(prefill_context_parallel_size=8)
        }),
        (None, {
            "speculative_config": SimpleNamespace(num_speculative_tokens=2)
        }),
        (None, {
            "kv_transfer_config": SimpleNamespace(kv_connector="x")
        }),
    ],
    ids=[
        "not-longctx", "seq-along-lane", "pcp", "spec-decode", "kv-connector"
    ],
)
def test_fold_is_refused_where_the_page_is_read_unfolded(
        fold_env, monkeypatch, attr, overrides):
    if attr == "USE_BATCHED_RPA_LONGCTX":
        monkeypatch.setattr(envs, attr, False)
    elif attr == "USE_BATCHED_RPA_SEQ_ON_LANE":
        monkeypatch.setattr(envs, attr, True)
    _refuse_kv_head_dim_fold(_fake_config(**overrides))
    assert os.environ["TPU_RPA_FOLD_KV_HEAD_DIM"] == "0"


def test_seq_along_lane_is_rejected_under_the_unified_pool(monkeypatch):
    """The pool reads `pool.shape[1]` as a token count; seq-lane puts heads there.

    Normalizing the spec from the selected backend removes the byte-count
    mismatch that used to abort this combination in
    `allocate_raw_kv_cache_tensors`, so it has to be rejected on purpose.
    """
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_LONGCTX", True)
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_SEQ_ON_LANE", True)
    monkeypatch.setattr(envs, "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", True)
    with pytest.raises(ValueError, match="unified KV pool"):
        _reject_seq_along_lane_unified_pool(_fake_config())


def test_seq_along_lane_is_fine_without_the_unified_pool(monkeypatch):
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_LONGCTX", True)
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_SEQ_ON_LANE", True)
    monkeypatch.setattr(envs, "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", False)
    _reject_seq_along_lane_unified_pool(_fake_config())
