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
"""CPU unit tests for non-causal attention support in batched RPA."""

import jax
import jax.numpy as jnp
import pytest

from tests.kernels.experimental.batched_rpa.correctness_test import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    schedule,
    wrapper,
)


def test_model_configs_use_causal_mask_default_and_override():
    """Verify ModelConfigs defaults use_causal_mask to True and accepts False."""
    default_cfg = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    assert default_cfg.use_causal_mask

    non_causal_cfg = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=False,
    )
    assert not non_causal_cfg.use_causal_mask


def test_generate_mask_causal_vs_non_causal():
    """Verify generate_mask differentiates causal vs non-causal masking."""
    m_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=True,
    )
    m_non_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=False,
    )
    block_cfg = configs.BlockSizes(bq_sz=64, bq_c_sz=32, bkv_sz=256, batch_size=1)
    serve_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=16,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    rpa_causal = configs.RpaConfigs(
        block=block_cfg,
        model=m_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    rpa_non_causal = configs.RpaConfigs(
        block=block_cfg,
        model=m_non_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    step_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.array(0)],
        local_k_start=None,
        local_k_end=None,
    )
    stitch_meta = kernel.StitchMetadata(
        bkv_sz_frm_cache=[jnp.array(128)],
        bkv_sz_frm_new=[jnp.array(128)],
        new_kv_len_start=[jnp.array(0)],
    )

    mask_causal = kernel.generate_mask(
        shape=(1, 8, 32, 256),
        bq_start=0,
        step_meta=step_meta,
        stitch_meta=stitch_meta,
        cfgs=rpa_causal,
    )
    mask_non_causal = kernel.generate_mask(
        shape=(1, 8, 32, 256),
        bq_start=0,
        step_meta=step_meta,
        stitch_meta=stitch_meta,
        cfgs=rpa_non_causal,
    )

    # In causal mask, query 0 cannot attend to future key 10
    assert not mask_causal[0][0, 0, 10]
    # In causal mask, query 10 can attend to past key 0
    assert mask_causal[0][0, 10, 0]

    # In non-causal mask, query 0 CAN attend to key 10
    assert mask_non_causal[0][0, 0, 10]
    # In non-causal mask, all queries attend to all 256 valid keys
    assert bool(jnp.all(mask_non_causal[0]))


def test_generate_mask_non_causal_partial_kv_block():
    """Verify non-causal masking isolates tail padding keys in partial blocks."""
    m_non_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=False,
    )
    block_cfg = configs.BlockSizes(bq_sz=64, bq_c_sz=32, bkv_sz=256, batch_size=1)
    serve_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=16,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    rpa_cfg = configs.RpaConfigs(
        block=block_cfg,
        model=m_non_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    step_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.array(0)],
        local_k_start=None,
        local_k_end=None,
    )
    # Partial block: 100 cached + 40 new = 140 valid tokens out of 256
    stitch_meta = kernel.StitchMetadata(
        bkv_sz_frm_cache=[jnp.array(100)],
        bkv_sz_frm_new=[jnp.array(40)],
        new_kv_len_start=[jnp.array(0)],
    )

    mask = kernel.generate_mask(
        shape=(1, 8, 32, 256),
        bq_start=0,
        step_meta=step_meta,
        stitch_meta=stitch_meta,
        cfgs=rpa_cfg,
    )

    # Valid tokens [0, 140) must be attended by all queries
    assert bool(jnp.all(mask[0][:, :, :140]))
    # Tail padding tokens [140, 256) must be masked out to prevent NaN/stale leakage
    assert bool(jnp.all(~mask[0][:, :, 140:]))


def test_generate_mask_sliding_window_causal_vs_non_causal():
    """Verify sliding window applies to causal mask and is ignored for non-causal.

    Matches RPA v3 and ref_ragged_paged_attention semantics where sliding_window
    is only active when use_causal_mask is True.
    """
    m_sw_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        sliding_window=64,
        use_causal_mask=True,
    )
    m_sw_non_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        sliding_window=64,
        use_causal_mask=False,
    )
    block_cfg = configs.BlockSizes(bq_sz=64, bq_c_sz=32, bkv_sz=256, batch_size=1)
    serve_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=16,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    rpa_causal = configs.RpaConfigs(
        block=block_cfg,
        model=m_sw_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    rpa_non_causal = configs.RpaConfigs(
        block=block_cfg,
        model=m_sw_non_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    step_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.array(0)],
        local_k_start=None,
        local_k_end=None,
    )
    stitch_meta = kernel.StitchMetadata(
        bkv_sz_frm_cache=[jnp.array(256)],
        bkv_sz_frm_new=[jnp.array(0)],
        new_kv_len_start=[jnp.array(0)],
    )

    mask_causal = kernel.generate_mask(
        shape=(1, 8, 32, 256),
        bq_start=0,
        step_meta=step_meta,
        stitch_meta=stitch_meta,
        cfgs=rpa_causal,
    )
    mask_non_causal = kernel.generate_mask(
        shape=(1, 8, 32, 256),
        bq_start=0,
        step_meta=step_meta,
        stitch_meta=stitch_meta,
        cfgs=rpa_non_causal,
    )

    # For causal attention with sliding window:
    # Query 0 can attend to key 0
    assert mask_causal[0][0, 0, 0]
    # Query 0 cannot attend to key 10 (causal bound)
    assert not mask_causal[0][0, 0, 10]

    # For non-causal attention:
    # Query 0 can attend to key 0, key 10, key 70 (all valid tokens)
    assert mask_non_causal[0][0, 0, 0]
    assert mask_non_causal[0][0, 0, 10]
    assert mask_non_causal[0][0, 0, 70]


def test_generate_mask_non_causal_with_attention_scope():
    """Verify attention scope local_k_start / local_k_end bounds."""
    m_non_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=False,
    )
    block_cfg = configs.BlockSizes(bq_sz=64, bq_c_sz=32, bkv_sz=256, batch_size=1)
    serve_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=16,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    rpa_cfg = configs.RpaConfigs(
        block=block_cfg,
        model=m_non_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    step_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.array(0)],
        local_k_start=[jnp.array(32)],
        local_k_end=[jnp.array(128)],
    )
    stitch_meta = kernel.StitchMetadata(
        bkv_sz_frm_cache=[jnp.array(256)],
        bkv_sz_frm_new=[jnp.array(0)],
        new_kv_len_start=[jnp.array(0)],
    )

    mask = kernel.generate_mask(
        shape=(1, 8, 32, 256),
        bq_start=0,
        step_meta=step_meta,
        stitch_meta=stitch_meta,
        cfgs=rpa_cfg,
    )

    # Keys < 32 are masked out by local_k_start
    assert bool(jnp.all(~mask[0][:, :, :32]))
    # Keys in [32, 128) are valid
    assert bool(jnp.all(mask[0][:, :, 32:128]))
    # Keys >= 128 are masked out by local_k_end
    assert bool(jnp.all(~mask[0][:, :, 128:]))


class RecordingMetadataComputer(schedule.BaseMetadataComputer):
    """Test helper that records k_loop invocations from BaseMetadataComputer."""

    def __init__(self, cfgs: configs.RpaConfigs):
        self.cfgs = cfgs
        self.recorded_end_k_idx: list[jax.Array] = []

    def k_loop(self, k_idx, carry: schedule.LoopCarry, **kwargs) -> schedule.LoopCarry:
        self.recorded_end_k_idx.append(kwargs["end_k_idx"])
        return schedule.LoopCarry(carry.hbm_offset, carry.count + 1)


def test_schedule_end_k_idx_causal_vs_non_causal():
    """Verify schedule drives real q_loop pruning for causal vs non-causal."""
    m_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=True,
    )
    m_non_causal = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=False,
    )
    block_cfg = configs.BlockSizes(bq_sz=64, bq_c_sz=32, bkv_sz=256, batch_size=1)
    serve_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=16,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    rpa_causal = configs.RpaConfigs(
        block=block_cfg,
        model=m_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    rpa_non_causal = configs.RpaConfigs(
        block=block_cfg,
        model=m_non_causal,
        serve=serve_cfg,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )

    # Sequence with 1024 tokens = 4 KV blocks of size 256.
    # For query block 0 (tokens 0..63):
    # - Causal attention only schedules k_idx=0 (end_k_idx=1).
    # - Non-causal attention schedules all 4 blocks (k_idx=0, 1, 2, 3 with end_k_idx=4).
    carry0 = schedule.LoopCarry(0, 0)

    comp_causal = RecordingMetadataComputer(rpa_causal)
    carry_causal = comp_causal.q_loop(
        q_idx=0,
        carry=carry0,
        s_idx=0,
        q_start=0,
        q_end=64,
        q_offset=0,
        kv_cache_len=0,
        kv_new_len=1024,
        num_k=4,
        p_offset=0,
    )
    assert int(carry_causal.count) == 1
    assert int(comp_causal.recorded_end_k_idx[0]) == 1

    comp_non_causal = RecordingMetadataComputer(rpa_non_causal)
    carry_non_causal = comp_non_causal.q_loop(
        q_idx=0,
        carry=carry0,
        s_idx=0,
        q_start=0,
        q_end=64,
        q_offset=0,
        kv_cache_len=0,
        kv_new_len=1024,
        num_k=4,
        p_offset=0,
    )
    assert int(carry_non_causal.count) == 4
    assert int(comp_non_causal.recorded_end_k_idx[0]) == 4


def test_ref_ragged_paged_attention_non_causal_numerics():
    """Verify ref attention diverges at prefix but matches at suffix."""
    rng = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(rng, 3)

    queries = jax.random.normal(k1, (32, 4, 128), dtype=jnp.float32)
    keys = jax.random.normal(k2, (32, 2, 128), dtype=jnp.float32)
    values = jax.random.normal(k3, (32, 2, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((10, 16, 4, 1, 128), dtype=jnp.float32)
    kv_lens = jnp.array([32], dtype=jnp.int32)
    page_indices = jnp.array([0, 1], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 32], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    out_causal, _ = ref_ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=True,
    )
    out_non_causal, _ = ref_ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=False,
    )

    assert out_causal.shape == (32, 4, 128)
    assert out_non_causal.shape == (32, 4, 128)
    assert bool(jnp.all(jnp.isfinite(out_non_causal)))

    # For query token 0, causal attention attends only to key token 0,
    # while non-causal attention attends to all 32 tokens, so outputs must diverge.
    diff_q0 = jnp.abs(out_causal[0] - out_non_causal[0])
    assert bool(jnp.max(diff_q0) > 1e-3)

    # For query token 31, both causal and non-causal attend to all 32 tokens (0..31),
    # so their attention outputs are numerically identical.
    diff_q31 = jnp.abs(out_causal[-1] - out_non_causal[-1])
    assert bool(jnp.max(diff_q31) < 1e-5)


def test_wrapper_rejects_sliding_window_with_non_causal():
    """Verify wrapper rejects sliding_window with use_causal_mask=False."""
    with pytest.raises(
        ValueError,
        match="Sliding window attention is only supported with causal masking.",
    ):
        wrapper.ragged_paged_attention(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            sliding_window=64,
            use_causal_mask=False,
        )
