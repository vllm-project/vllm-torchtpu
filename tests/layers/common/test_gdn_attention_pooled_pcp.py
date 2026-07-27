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
"""Tests for PCP prefill with GDN state in the unified block pool."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order as _build_pcp_rank_major_token_order
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    pcp_local_token_counts as _pcp_local_token_counts
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_v3_wrapper
from vllm_torchtpu.layers.common.gdn_attention import (
    run_jax_gdn_attention_pooled_local,
    run_jax_gdn_attention_pooled_pcp_tp_prefill)
from vllm_torchtpu.layers.common.utils import inverse_reorder_for_sharding


def test_pooled_pcp_gdn_op_is_available():
    assert callable(run_jax_gdn_attention_pooled_pcp_tp_prefill)


def test_pooled_pcp_gdn_requires_pcp_mesh_axis():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape((1, )), ('model', ))
    with pytest.raises(NotImplementedError, match='pcp mesh axis'):
        run_jax_gdn_attention_pooled_pcp_tp_prefill(
            jnp.zeros((4, 8), dtype=jnp.bfloat16),
            jnp.zeros((4, 2), dtype=jnp.bfloat16),
            jnp.zeros((4, 2), dtype=jnp.bfloat16),
            jnp.zeros((2, 16, 4, 8), dtype=jnp.bfloat16),
            jnp.zeros((8, 1, 4), dtype=jnp.bfloat16),
            None,
            jnp.zeros((2, ), dtype=jnp.float32),
            jnp.zeros((2, ), dtype=jnp.float32),
            jnp.array([1], dtype=jnp.int32),
            jnp.array([0, 4], dtype=jnp.int32),
            jnp.array([0, 1, 1], dtype=jnp.int32),
            jnp.array([4], dtype=jnp.int32),
            n_kq=2,
            n_v=2,
            d_k=2,
            d_v=2,
            kernel_size=4,
            pool_block_tokens=16,
            pcp_size=2,
            interleave_size=4,
            mesh=mesh,
        )


def test_pooled_pcp_gdn_validates_mesh_size():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape((1, )), ('pcp', ))
    with pytest.raises(ValueError, match='does not match mesh axis size'):
        run_jax_gdn_attention_pooled_pcp_tp_prefill(
            jnp.zeros((4, 8), dtype=jnp.bfloat16),
            jnp.zeros((4, 2), dtype=jnp.bfloat16),
            jnp.zeros((4, 2), dtype=jnp.bfloat16),
            jnp.zeros((2, 16, 4, 8), dtype=jnp.bfloat16),
            jnp.zeros((8, 1, 4), dtype=jnp.bfloat16),
            None,
            jnp.zeros((2, ), dtype=jnp.float32),
            jnp.zeros((2, ), dtype=jnp.float32),
            jnp.array([1], dtype=jnp.int32),
            jnp.array([0, 4], dtype=jnp.int32),
            jnp.array([0, 1, 1], dtype=jnp.int32),
            jnp.array([4], dtype=jnp.int32),
            n_kq=2,
            n_v=2,
            d_k=2,
            d_v=2,
            kernel_size=4,
            pool_block_tokens=16,
            pcp_size=2,
            interleave_size=4,
            mesh=mesh,
        )


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != 'tpu':
        pytest.skip(reason)
    return devices


def _pool_region_geometry(pool_shape, pool_dtype, n_v, d_k, d_v, kernel_size,
                          conv_dim, pool_block_tokens):
    """Mirror of run_jax_gdn_attention_pooled_local's region derivation."""
    assert pool_block_tokens % pool_shape[1] == 0
    split = pool_block_tokens // pool_shape[1]
    lanes = pool_shape[-1]
    per_tok_elems = int(np.prod(pool_shape[2:]))
    tok_bytes = per_tok_elems * jnp.dtype(pool_dtype).itemsize
    ssm_bytes = n_v * d_k * d_v * 4
    assert ssm_bytes % tok_bytes == 0
    ssm_ntok = ssm_bytes // tok_bytes
    conv_bytes = (kernel_size - 1) * conv_dim * 2
    conv_data_rows = conv_bytes // (2 * lanes)
    conv_ntok = 1
    while (conv_ntok * tok_bytes < conv_bytes or ssm_ntok % conv_ntok != 0):
        conv_ntok *= 2
    return split, ssm_ntok, conv_ntok, conv_data_rows, lanes


def test_pooled_pcp_prefill_matches_split_and_pooled_baselines():
    _require_tpu_devices(
        2, 'pooled GDN PCP numerical test requires two TPU devices.')
    pcp_size = 2
    interleave_size = 16
    lengths = [96]
    n_kq = 4
    n_v = 4
    # The chunked (prefill) V3 kernel loads qkv through the large layout,
    # which slices whole 128-lane registers per head, so the head dims have
    # to be a multiple of the TPU lane count.
    d_k = 128
    d_v = 128
    kernel_size = 4
    num_tokens = sum(lengths)
    num_blocks = len(lengths) + 1
    dim = 2 * n_kq * d_k + n_v * d_v
    qkv_split_sizes = [n_kq * d_k, n_kq * d_k, n_v * d_v]
    local_n_v = n_v // pcp_size

    # Pool geometry: bf16 KV-shaped pool, one manager block per kernel
    # block (split == 1). Both the full-head baseline pool and the
    # per-rank PCP-localized pools use the same physical shape; only the
    # region extents differ. The lane count has to be a multiple of d_v so
    # the ssm region can be read back with out_lanes=d_v, and the block has
    # to hold the widest slot -- the full-head baseline's 256 ssm + 16 conv
    # tokens of 1 KB each.
    pool_heads2 = 4
    pool_lanes = 128
    pool_block_tokens = 512
    pool_dtype = jnp.bfloat16

    rng = jax.random.key(0)
    keys = jax.random.split(rng, 8)
    mixed_qkv0 = jax.random.normal(keys[0], (num_tokens, dim),
                                   dtype=jnp.bfloat16)
    b0 = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.bfloat16)
    a0 = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.bfloat16)
    conv_weight0 = jax.random.normal(keys[3], (dim, 1, kernel_size),
                                     dtype=jnp.bfloat16)
    conv_bias0 = jax.random.normal(keys[4], (dim, ), dtype=jnp.bfloat16)
    A_log0 = jax.random.normal(keys[5], (n_v, ), dtype=jnp.float32)
    dt_bias0 = jax.random.normal(keys[6], (n_v, ), dtype=jnp.float32)

    query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
    state_indices = jnp.array([1], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
    seq_lens = jnp.array(lengths, dtype=jnp.int32)

    conv_state0 = jnp.zeros((num_blocks, kernel_size - 1, dim),
                            dtype=jnp.bfloat16)
    rec_state0 = jnp.zeros((num_blocks, n_v, d_k, d_v), dtype=jnp.float32)
    (ref_conv, ref_rec), ref_output = gdn_v3_wrapper.fused_conv1d_gdn(
        mixed_qkv0,
        b0,
        a0,
        conv_state0,
        rec_state0,
        conv_weight0,
        conv_bias0,
        A_log0,
        dt_bias0,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size)

    full_pool = jnp.zeros(
        (num_blocks, pool_block_tokens, pool_heads2, pool_lanes),
        dtype=pool_dtype)
    _, full_output = run_jax_gdn_attention_pooled_local(
        mixed_qkv0,
        b0,
        a0,
        full_pool,
        conv_weight0,
        conv_bias0,
        A_log0,
        dt_bias0,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens)
    np.testing.assert_allclose(np.array(full_output),
                               np.array(ref_output),
                               rtol=5e-2,
                               atol=5e-2)

    # --- Pooled PCP under test. ---
    padded_num_tokens = int(
        _pcp_local_token_counts(lengths, pcp_size,
                                interleave_size).max()) * pcp_size
    token_order, _ = _build_pcp_rank_major_token_order(
        np.array(lengths, dtype=np.int32),
        pcp_size,
        interleave_size,
        padded_num_tokens,
    )
    valid = token_order >= 0
    pad_tokens = padded_num_tokens - num_tokens
    mixed_qkv_padded = jnp.pad(mixed_qkv0, ((0, pad_tokens), (0, 0)))
    b_padded = jnp.pad(b0, ((0, pad_tokens), (0, 0)))
    a_padded = jnp.pad(a0, ((0, pad_tokens), (0, 0)))
    valid_indices = np.where(valid)[0]
    src_indices = token_order[valid]
    packed_qkv = jnp.zeros_like(mixed_qkv_padded).at[valid_indices].set(
        mixed_qkv_padded[src_indices])
    packed_b = jnp.zeros_like(b_padded).at[valid_indices].set(
        b_padded[src_indices])
    packed_a = jnp.zeros_like(a_padded).at[valid_indices].set(
        a_padded[src_indices])

    mesh = Mesh(
        np.array(jax.devices()[:pcp_size]).reshape((pcp_size, )), ('pcp', ))

    def shard_tokens(x):
        return jax.device_put(x, NamedSharding(mesh, P('pcp', None)))

    def shard_pool(x):
        return jax.device_put(x, NamedSharding(mesh, P('pcp', None, None,
                                                       None)))

    def replicate(x):
        return jax.device_put(x, NamedSharding(mesh, P()))

    # One same-shaped pool per rank, stacked on the block axis.
    pcp_pools = jnp.zeros(
        (pcp_size * num_blocks, pool_block_tokens, pool_heads2, pool_lanes),
        dtype=pool_dtype)

    new_pools, pcp_output = run_jax_gdn_attention_pooled_pcp_tp_prefill(
        shard_tokens(packed_qkv),
        shard_tokens(packed_b),
        shard_tokens(packed_a),
        shard_pool(pcp_pools),
        replicate(conv_weight0),
        replicate(conv_bias0),
        replicate(A_log0),
        replicate(dt_bias0),
        replicate(state_indices),
        replicate(query_start_loc),
        replicate(distribution),
        replicate(seq_lens),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        mesh=mesh)

    # Output: unpack rank-major rows back to request-major and compare. The
    # op returns per-head rows (num_tokens, n_v, d_v); flatten the head dims
    # back to the reference's (num_tokens, n_v * d_v) layout.
    pcp_output_np = np.array(pcp_output).reshape(padded_num_tokens, -1)
    pcp_output_seq = np.zeros((padded_num_tokens, pcp_output_np.shape[1]),
                              dtype=pcp_output_np.dtype)
    pcp_output_seq[token_order[valid]] = pcp_output_np[valid]
    np.testing.assert_allclose(pcp_output_seq[:num_tokens],
                               np.array(ref_output),
                               rtol=5e-2,
                               atol=5e-2)

    # State regions: each rank's pool must hold its contiguous head shard
    # of the ssm state and its interleave-reordered shard of the conv
    # state — the bytes the Raiden manifest ships.
    local_conv_dim = dim // pcp_size
    split, ssm_ntok, conv_ntok, conv_data_rows, lanes = _pool_region_geometry(
        (num_blocks, pool_block_tokens, pool_heads2, pool_lanes), pool_dtype,
        local_n_v, d_k, d_v, kernel_size, local_conv_dim, pool_block_tokens)
    new_pools_np = np.array(new_pools)
    rank_ssm = []
    rank_conv = []
    for rank in range(pcp_size):
        rank_pool = jnp.asarray(new_pools_np[rank * num_blocks:(rank + 1) *
                                             num_blocks])
        ssm = pool_adapters.gather_region(rank_pool,
                                          state_indices,
                                          tok0=0,
                                          ntok=ssm_ntok,
                                          out_dtype=jnp.float32,
                                          out_lanes=d_v,
                                          split=split)
        rank_ssm.append(
            np.array(ssm).reshape(len(lengths), local_n_v, d_k, d_v))
        conv = pool_adapters.gather_region(rank_pool,
                                           state_indices,
                                           tok0=ssm_ntok,
                                           ntok=conv_ntok,
                                           out_dtype=jnp.bfloat16,
                                           split=split)
        rank_conv.append(
            np.array(conv.astype(jnp.float32))[:, :conv_data_rows, :].reshape(
                len(lengths), kernel_size - 1, local_conv_dim))
    ssm_all = np.concatenate(rank_ssm, axis=1)
    np.testing.assert_allclose(ssm_all,
                               np.array(ref_rec[1:1 + len(lengths)]),
                               rtol=5e-2,
                               atol=5e-2)
    conv_interleaved = np.concatenate(rank_conv, axis=-1)
    conv_raw = inverse_reorder_for_sharding(jnp.asarray(conv_interleaved),
                                            qkv_split_sizes, pcp_size, -1)
    np.testing.assert_allclose(np.array(conv_raw),
                               np.array(ref_conv[1:1 + len(lengths)].astype(
                                   jnp.float32)),
                               rtol=5e-2,
                               atol=5e-2)
