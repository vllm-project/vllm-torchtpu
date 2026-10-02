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
"""Raiden GDN pooled resharding vs dense reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import RegionSpec
from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.layers.core import gdn_attention

from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

try:
    from kernels.gdn.v3.gdn_pool_state_utils import (
        read_states,
        write_states,
    )
except ImportError:
    from tests.kernels.gdn.v3.gdn_pool_state_utils import (
        read_states,
        write_states,
    )

TOL = dict(rtol=1e-3, atol=1e-3)


def test_qk_pair_layout_rejected_for_pooled_gdn(monkeypatch):
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    geometry = rpm.GdnHeadGeometry(
        local_key_heads=2,
        local_value_heads=8,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_size=4,
    )
    with pytest.raises(
        rpm.ManifestError, match="TPU_GDN_CONV_QK_PAIR_LAYOUT is not readable"
    ):
        rpm._gdn_conv_regions(
            conv_shape=(1, 3, 1536),
            itemsize=4,
            geometry=geometry,
        )


def _run_reshard_prefill_then_decode(
    ssm_dtype, degree, carrier_bytes, *, pair_blocked: bool
):
    ssm_itemsize = 4 if ssm_dtype == jnp.float32 else 2

    n_kq = 16
    n_v = 64
    d_k = 128
    d_v = 128
    kernel_size = 4
    dim = n_kq * d_k * 2 + n_v * d_v
    kbs = 256
    lanes = carrier_bytes // 4
    pool_block_tokens = 4352 * (1024 // carrier_bytes)
    split = pool_block_tokens // kbs
    num_mgr = 2
    pool_shape = (num_mgr * split, kbs, 1, 4, lanes)
    block_shape = (split, kbs, 1, 4, lanes)
    block_bytes = split * kbs * 1 * 4 * lanes

    keys = iter(jax.random.split(jax.random.key(20261002), 12))

    def normal(shape, scale, dtype=jnp.bfloat16):
        val = jax.random.normal(next(keys), shape, dtype=dtype)
        return val * jnp.asarray(scale, dtype=dtype)

    qkv_prefill = normal((16, dim), 0.5)
    b_prefill = normal((16, n_v), 0.5)
    a_prefill = normal((16, n_v), 0.5)

    qkv_decode = normal((1, dim), 0.5)
    b_decode = normal((1, n_v), 0.5)
    a_decode = normal((1, n_v), 0.5)

    conv_weight = normal((dim, 1, kernel_size), 0.5)
    conv_bias = normal((dim,), 0.1)
    a_log = normal((n_v,), 0.02, dtype=jnp.float32)
    dt_bias = normal((n_v,), 0.02, dtype=jnp.float32) - 2.0

    init_conv = normal((1, kernel_size - 1, dim), 0.5, dtype=jnp.float32)
    init_recurrent = normal((1, n_v, d_k, d_v), 0.5, dtype=ssm_dtype)

    # 1. Dense tuple reference: prefill 16 tokens -> decode 1 token
    (dense_prefill_conv, dense_prefill_ssm), _ = wrapper.fused_conv1d_gdn(
        qkv=qkv_prefill,
        b=b_prefill,
        a=a_prefill,
        states=(jnp.copy(init_conv), jnp.copy(init_recurrent)),
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=jnp.array([0, 16], dtype=jnp.int32),
        state_indices=jnp.array([0], dtype=jnp.int32),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([20], dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    (dense_decode_conv, dense_decode_ssm), dense_decode_out = wrapper.fused_conv1d_gdn(
        qkv=qkv_decode,
        b=b_decode,
        a=a_decode,
        states=(jnp.copy(dense_prefill_conv), jnp.copy(dense_prefill_ssm)),
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=jnp.array([0, 1], dtype=jnp.int32),
        state_indices=jnp.array([0], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        seq_lens=jnp.array([21], dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    # 2. Producer shards: prefill 16 tokens into head-sharded local pools
    n_kq_local = n_kq // degree
    n_v_local = n_v // degree
    local_dim = 2 * n_kq_local * d_k + n_v_local * d_v
    geom_local = rpm.GdnHeadGeometry(
        local_key_heads=n_kq_local,
        local_value_heads=n_v_local,
        key_head_dim=d_k,
        value_head_dim=d_v,
        conv_kernel_size=kernel_size,
    )

    row_stride = local_dim * 4
    key_bytes = d_k * 4
    value_bytes = d_v * 4

    src_ssm_bytes = n_v_local * d_v * d_k * ssm_itemsize
    dst_ssm_bytes = n_v * d_v * d_k * ssm_itemsize

    dst_phys = np.zeros(block_bytes, dtype=np.uint8)

    for r in range(degree):
        q_slice = slice(r * n_kq_local * d_k, (r + 1) * n_kq_local * d_k)
        k_slice = slice(
            n_kq * d_k + r * n_kq_local * d_k, n_kq * d_k + (r + 1) * n_kq_local * d_k
        )
        v_slice = slice(
            2 * n_kq * d_k + r * n_v_local * d_v,
            2 * n_kq * d_k + (r + 1) * n_v_local * d_v,
        )

        qkv_r = jnp.concatenate(
            [qkv_prefill[:, q_slice], qkv_prefill[:, k_slice], qkv_prefill[:, v_slice]],
            axis=-1,
        )
        b_r = b_prefill[:, r * n_v_local : (r + 1) * n_v_local]
        a_r = a_prefill[:, r * n_v_local : (r + 1) * n_v_local]
        conv_weight_r = jnp.concatenate(
            [conv_weight[q_slice], conv_weight[k_slice], conv_weight[v_slice]], axis=0
        )
        conv_bias_r = jnp.concatenate(
            [conv_bias[q_slice], conv_bias[k_slice], conv_bias[v_slice]], axis=0
        )
        a_log_r = a_log[r * n_v_local : (r + 1) * n_v_local]
        dt_bias_r = dt_bias[r * n_v_local : (r + 1) * n_v_local]

        init_conv_r = jnp.concatenate(
            [
                init_conv[:, :, q_slice],
                init_conv[:, :, k_slice],
                init_conv[:, :, v_slice],
            ],
            axis=-1,
        )
        init_recurrent_r = init_recurrent[:, r * n_v_local : (r + 1) * n_v_local]

        pool_r = jnp.zeros(pool_shape, dtype=jnp.int8)
        pool_r = write_states(
            pool_r,
            init_recurrent_r,
            init_conv_r,
            np.array([1], dtype=np.int32),
            split=split,
            recurrent_dtype=ssm_dtype,
        )

        pool_r, _ = gdn_attention.run_jax_gdn_attention_pooled_local(
            mixed_qkv=qkv_r,
            b=b_r,
            a=a_r,
            recurrent_state=pool_r,
            conv_weight=conv_weight_r,
            conv_bias=conv_bias_r,
            A_log=a_log_r,
            dt_bias=dt_bias_r,
            query_start_loc=jnp.array([0, 16], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([20], dtype=jnp.int32),
            n_kq=n_kq_local,
            n_v=n_v_local,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pool_block_tokens=pool_block_tokens,
            recurrent_state_dtype=ssm_dtype,
        )

        # 3. Transfer: Lower spans and copy physical bytes into destination
        if pair_blocked:
            conv_regions = (
                RegionSpec(
                    "gdn_conv_qk", 0, row_stride, 2 * n_kq_local * key_bytes, 3, 1
                ),
                RegionSpec(
                    "gdn_conv_v",
                    2 * n_kq_local * key_bytes,
                    row_stride,
                    value_bytes,
                    3,
                    n_v_local,
                ),
            )
        else:
            conv_regions = rpm._gdn_conv_regions(
                conv_shape=(1, kernel_size - 1, local_dim),
                itemsize=4,
                geometry=geom_local,
            )
        ssm_regions = rpm._gdn_ssm_regions(
            ssm_shape=(1, n_v_local, d_v, d_k),
            itemsize=ssm_itemsize,
        )

        reg_ssm = sol.lower_gdn_state_shard_spans_sol(
            tag="gdn.ssm.l0",
            block_id=0,
            transfer_rank=r,
            parallelism=degree,
            regions=ssm_regions,
            dst_shards=1,
            physical_granule_bytes=carrier_bytes,
        )
        reg_conv = sol.lower_gdn_state_shard_spans_sol(
            tag="gdn.conv.l0",
            block_id=0,
            transfer_rank=r,
            parallelism=degree,
            regions=conv_regions,
            dst_shards=1,
            physical_granule_bytes=carrier_bytes,
        )

        block1_slice_r = np.asarray(pool_r[1 * split : 2 * split]).view(np.uint8)
        raw_src_r = _apply_xla_tiled_layout(
            block1_slice_r, (4, 3, 2, 1, 0), ((4, 128), (4, 1))
        )

        for span in reg_ssm.spans:
            for i in range(span.count):
                s_off = span.src_offset_bytes + i * span.src_stride_bytes
                d_off = span.dst_offset_bytes + i * span.dst_stride_bytes
                dst_phys[d_off : d_off + span.size_bytes] = raw_src_r[
                    s_off : s_off + span.size_bytes
                ]

        for span in reg_conv.spans:
            for i in range(span.count):
                s_off = (
                    src_ssm_bytes + span.src_offset_bytes + i * span.src_stride_bytes
                )
                d_off = (
                    dst_ssm_bytes + span.dst_offset_bytes + i * span.dst_stride_bytes
                )
                dst_phys[d_off : d_off + span.size_bytes] = raw_src_r[
                    s_off : s_off + span.size_bytes
                ]

    # 4. Un-tile destination block 1 back into destination pool
    idx = _apply_xla_tiled_layout(
        np.arange(block_bytes).reshape(block_shape), (4, 3, 2, 1, 0), ((4, 128), (4, 1))
    )
    dst_block1_logical = np.empty(block_bytes, dtype=np.uint8)
    dst_block1_logical[idx] = dst_phys
    dst_pool = jnp.zeros(pool_shape, dtype=jnp.int8)
    dst_pool = dst_pool.at[1 * split : 2 * split].set(
        dst_block1_logical.reshape(block_shape).view(jnp.int8)
    )

    # Read back states before decode
    _, pre_decode_conv = read_states(
        dst_pool,
        np.array([1], dtype=np.int32),
        split=split,
        recurrent_shape=(n_v, d_k, d_v),
        conv_shape=(kernel_size - 1, dim),
        recurrent_dtype=ssm_dtype,
    )

    # 5. Full-width decode on destination pool
    dst_pool, pooled_decode_out = gdn_attention.run_jax_gdn_attention_pooled_local(
        mixed_qkv=qkv_decode,
        b=b_decode,
        a=a_decode,
        recurrent_state=dst_pool,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        A_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=jnp.array([0, 1], dtype=jnp.int32),
        state_indices=jnp.array([1], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        seq_lens=jnp.array([21], dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
        recurrent_state_dtype=ssm_dtype,
    )

    final_ssm, final_conv = read_states(
        dst_pool,
        np.array([1], dtype=np.int32),
        split=split,
        recurrent_shape=(n_v, d_k, d_v),
        conv_shape=(kernel_size - 1, dim),
        recurrent_dtype=ssm_dtype,
    )

    return (
        pre_decode_conv,
        dense_prefill_conv,
        pooled_decode_out,
        dense_decode_out,
        final_conv,
        dense_decode_conv,
        final_ssm,
        dense_decode_ssm,
    )


@pytest.mark.parametrize("ssm_dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("degree", [2, 8])
@pytest.mark.parametrize("carrier_bytes", [512, 1024])
def test_pooled_gdn_state_reshard_matches_dense(ssm_dtype_name, degree, carrier_bytes):
    ssm_dtype = jnp.float32 if ssm_dtype_name == "float32" else jnp.bfloat16
    (
        pre_decode_conv,
        dense_prefill_conv,
        pooled_decode_out,
        dense_decode_out,
        final_conv,
        dense_decode_conv,
        final_ssm,
        dense_decode_ssm,
    ) = _run_reshard_prefill_then_decode(
        ssm_dtype, degree, carrier_bytes, pair_blocked=False
    )

    assert bool(jnp.allclose(pre_decode_conv[0], dense_prefill_conv[0], **TOL))
    assert bool(jnp.allclose(pooled_decode_out, dense_decode_out, **TOL))
    assert bool(jnp.allclose(final_conv[0], dense_decode_conv[0], **TOL))
    assert bool(jnp.allclose(final_ssm[0], dense_decode_ssm[0], **TOL))


def test_qk_pair_layout_scrambles_pooled_gdn_decode():
    (
        pre_decode_conv,
        dense_prefill_conv,
        pooled_decode_out,
        dense_decode_out,
        final_conv,
        dense_decode_conv,
        final_ssm,
        dense_decode_ssm,
    ) = _run_reshard_prefill_then_decode(jnp.float32, 8, 1024, pair_blocked=True)

    # Scramble checks: pair-blocked lowering scrambles conv state
    # and corrupts decode output
    assert not bool(jnp.allclose(pre_decode_conv[0], dense_prefill_conv[0], **TOL))
    assert not bool(jnp.allclose(pooled_decode_out, dense_decode_out, **TOL))
    assert not bool(jnp.allclose(final_conv[0], dense_decode_conv[0], **TOL))
    assert not bool(jnp.allclose(final_ssm[0], dense_decode_ssm[0], **TOL))
