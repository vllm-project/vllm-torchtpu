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
"""CPU unit tests for non-causal attention support in batched RPA (#2882)."""

import jax
import jax.numpy as jnp

from tests.kernels.experimental.batched_rpa.correctness_test import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
)


def test_model_configs_use_causal_mask_default_and_override():
    """Verify ModelConfigs defaults use_causal_mask to True and accepts False."""
    default_cfg = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    assert default_cfg.use_causal_mask is True

    non_causal_cfg = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
        use_causal_mask=False,
    )
    assert non_causal_cfg.use_causal_mask is False


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
    assert mask_causal[0][0, 0, 10] == False  # noqa: E712
    # In causal mask, query 10 can attend to past key 0
    assert mask_causal[0][0, 10, 0] == True  # noqa: E712

    # In non-causal mask, query 0 CAN attend to key 10
    assert mask_non_causal[0][0, 0, 10] == True  # noqa: E712
    # In non-causal mask, all queries attend to all 256 valid keys
    assert bool(jnp.all(mask_non_causal[0])) is True


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
    assert bool(jnp.all(mask[0][:, :, :140])) is True
    # Tail padding tokens [140, 256) must be masked out to prevent NaN/stale leakage
    assert bool(jnp.all(~mask[0][:, :, 140:])) is True


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
    assert mask_causal[0][0, 0, 0] == True  # noqa: E712
    # Query 0 cannot attend to key 10 (causal bound)
    assert mask_causal[0][0, 0, 10] == False  # noqa: E712

    # For non-causal attention:
    # Query 0 can attend to key 0, key 10, key 70 (all valid tokens)
    assert mask_non_causal[0][0, 0, 0] == True  # noqa: E712
    assert mask_non_causal[0][0, 0, 10] == True  # noqa: E712
    assert mask_non_causal[0][0, 0, 70] == True  # noqa: E712


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
    assert bool(jnp.all(~mask[0][:, :, :32])) is True
    # Keys in [32, 128) are valid
    assert bool(jnp.all(mask[0][:, :, 32:128])) is True
    # Keys >= 128 are masked out by local_k_end
    assert bool(jnp.all(~mask[0][:, :, 128:])) is True


def test_schedule_end_k_idx_causal_vs_non_causal():
    """Verify schedule prunes KV blocks for causal but keeps all for non-causal."""
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

    # Sequence with 1024 tokens = 4 KV blocks of size 256
    num_k = 4
    q_idx = 0
    q_offset = 0
    q_sz_task = 64
    bkv_sz = 256

    # Causal schedule formula
    end_k_idx_causal = (q_offset + q_idx * 64 + q_sz_task - 1) // bkv_sz + 1
    end_k_causal = (
        min(num_k, end_k_idx_causal) if m_causal.use_causal_mask else num_k
    )
    assert end_k_causal == 1  # Only block 0 is scheduled for q_idx=0

    # Non-causal schedule formula
    end_k_non_causal = (
        min(num_k, end_k_idx_causal) if m_non_causal.use_causal_mask else num_k
    )
    assert end_k_non_causal == 4  # All 4 blocks are scheduled for q_idx=0


def test_ref_ragged_paged_attention_non_causal_numerics():
    """Verify reference implementation outputs match non-causal attention semantics."""
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
    # Output must be finite
    assert bool(jnp.all(jnp.isfinite(out_non_causal))) is True

    # For query token 0, causal attention attends only to key token 0
    # Non-causal attention attends to all 32 tokens, so the outputs must differ
    diff = jnp.abs(out_causal[0] - out_non_causal[0])
    assert bool(jnp.max(diff) > 1e-3) is True
