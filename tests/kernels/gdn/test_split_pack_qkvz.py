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
"""Unit tests for GDN split_pack_gdn_qkvz kernel and fused_conv1d_gdn 3D qkv path."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.gdn.split_pack_qkvz import (
    can_split_pack_qkvz,
    legal_row_tile,
    split_pack_gdn_qkvz,
)

_TEST_VMEM_16M = 16 * 1024 * 1024
_TEST_VMEM_64M = 64 * 1024 * 1024


def test_legal_row_tile():
    assert legal_row_tile(8, 64) == 8
    assert legal_row_tile(24, 64) == 24
    assert legal_row_tile(40, 64) == 40
    assert legal_row_tile(40, 16) == 8
    assert legal_row_tile(48, 64) == 48
    assert legal_row_tile(48, 32) == 24
    assert legal_row_tile(384, 64) == 64
    assert legal_row_tile(1024, 64) == 64

    with pytest.raises(ValueError, match="rows must be a positive multiple of 8"):
        legal_row_tile(12, 64)
    with pytest.raises(ValueError, match="rows must be a positive multiple of 8"):
        legal_row_tile(4, 64)
    with pytest.raises(ValueError, match="max_tile must be a positive multiple of 8"):
        legal_row_tile(64, 12)


def test_can_split_pack_qkvz_alignment_and_vmem_guards():
    assert can_split_pack_qkvz(
        8, 12288, 64, 128, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    assert can_split_pack_qkvz(
        384, 1536, 8, 128, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_16M
    )
    assert not can_split_pack_qkvz(
        4, 12288, 64, 128, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    assert not can_split_pack_qkvz(
        12, 12288, 64, 128, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    assert not can_split_pack_qkvz(
        16, 12200, 64, 128, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    assert not can_split_pack_qkvz(
        16, 1536, 4, 128, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    assert not can_split_pack_qkvz(
        16, 1536, 8, 64, dtype_itemsize=2, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    assert not can_split_pack_qkvz(
        16, 1536, 8, 128, dtype_itemsize=1, vmem_capacity_bytes=_TEST_VMEM_64M
    )
    # 512 KiB VMEM cannot fit 8 double-buffered rows of Qwen3.5-397B TP=1.
    assert not can_split_pack_qkvz(
        16, 12288, 64, 128, dtype_itemsize=2, vmem_capacity_bytes=512 * 1024
    )


@pytest.mark.parametrize("rows", [8, 24, 40, 48, 96])
@pytest.mark.parametrize("n_v", [8, 16])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_split_pack_gdn_qkvz_cpu_interpret_bit_exact(rows, n_v, dtype):
    qkv_size = 512 + n_v * 128
    d_v = 128
    total = qkv_size + n_v * d_v
    x = jax.random.normal(jax.random.key(rows + n_v), (rows, total), dtype=dtype)

    qkv_3d, z_3d = split_pack_gdn_qkvz(
        x,
        qkv_size=qkv_size,
        local_num_v_heads=n_v,
        head_v_dim=d_v,
        max_tile=32,
        interpret=True,
    )

    expected_qkv = x[:, :qkv_size].astype(jnp.float32).reshape(rows, 1, qkv_size)
    expected_z = x[:, qkv_size:].reshape(rows, n_v, d_v)

    assert qkv_3d.shape == (rows, 1, qkv_size)
    assert qkv_3d.dtype == jnp.float32
    assert z_3d.shape == (rows, n_v, d_v)
    assert z_3d.dtype == dtype
    assert jnp.array_equal(qkv_3d, expected_qkv)
    assert jnp.array_equal(z_3d, expected_z)


def test_split_pack_gdn_qkvz_rejects_bad_inputs():
    x_3d = jnp.zeros((8, 1, 2560), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="must be 2D"):
        split_pack_gdn_qkvz(
            x_3d, qkv_size=1536, local_num_v_heads=8, head_v_dim=128, interpret=True
        )

    x_bad_cols = jnp.zeros((8, 2048), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="width"):
        split_pack_gdn_qkvz(
            x_bad_cols,
            qkv_size=1536,
            local_num_v_heads=8,
            head_v_dim=128,
            interpret=True,
        )

    x_unaligned = jnp.zeros((4, 2560), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="precondition"):
        split_pack_gdn_qkvz(
            x_unaligned,
            qkv_size=1536,
            local_num_v_heads=8,
            head_v_dim=128,
            interpret=True,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [8, 16, 24, 40, 48, 64, 128, 160])
@pytest.mark.parametrize("n_v", [8, 16, 32])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_tpu_split_pack_multi_tile_and_sublane_parity(rows, n_v, dtype):
    n_kq = max(2, n_v // 4)
    d_k, d_v = 128, 128
    qkv_size = 2 * n_kq * d_k + n_v * d_v
    total = qkv_size + n_v * d_v
    qkvz = jax.random.normal(jax.random.key(rows * 7 + n_v), (rows, total), dtype=dtype)
    qkv_3d, z_3d = split_pack_gdn_qkvz(
        qkvz,
        qkv_size=qkv_size,
        local_num_v_heads=n_v,
        head_v_dim=d_v,
    )
    assert qkv_3d.dtype == jnp.float32
    assert z_3d.dtype == dtype
    assert jnp.array_equal(
        qkv_3d, qkvz[:, :qkv_size].astype(jnp.float32).reshape(rows, 1, qkv_size)
    )
    assert jnp.array_equal(z_3d, qkvz[:, qkv_size:].reshape(rows, n_v, d_v))


@pytest.mark.tpu_test
@pytest.mark.parametrize("mode", ["prefill", "mixed"])
def test_tpu_fused_conv1d_gdn_3d_qkv_equivalence(mode):
    from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper

    rows = 64
    n_kq, n_v, d_k, d_v, kernel_size = 2, 8, 128, 128, 4
    qkv_size = 2 * n_kq * d_k + n_v * d_v
    total = qkv_size + n_v * d_v
    key = jax.random.key(17 if mode == "prefill" else 29)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    qkvz = jax.random.normal(k1, (rows, total), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (rows, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(k3, (rows, n_v), dtype=jnp.bfloat16)
    conv_w = jax.random.normal(k4, (qkv_size, 1, kernel_size), dtype=jnp.bfloat16)
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    conv_state = jnp.zeros((4, kernel_size - 1, qkv_size), dtype=jnp.float32)
    rec_state = jnp.zeros((4, n_v, d_k, d_v), dtype=jnp.float32)

    if mode == "prefill":
        q_start = jnp.array([0, rows, rows, rows], dtype=jnp.int32)
        s_idx = jnp.array([1, 0, 0], dtype=jnp.int32)
        dist = jnp.array([0, 1, 1], dtype=jnp.int32)
        s_lens = jnp.array([rows, 0, 0], dtype=jnp.int32)
    else:
        # 2 decode sequences (1 token each) + 1 prefill sequence (62 tokens).
        q_start = jnp.array([0, 1, 2, rows], dtype=jnp.int32)
        s_idx = jnp.array([1, 2, 3], dtype=jnp.int32)
        dist = jnp.array([2, 3, 3], dtype=jnp.int32)
        s_lens = jnp.array([8, 8, rows - 2], dtype=jnp.int32)

    (cs_ref, rs_ref), out_ref = gdn_wrapper.fused_conv1d_gdn(
        qkvz[:, :qkv_size],
        b,
        a,
        (jnp.array(conv_state), jnp.array(rec_state)),
        conv_w,
        None,
        a_log,
        dt_bias,
        q_start,
        s_idx,
        dist,
        s_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    qkv_3d, z_3d = split_pack_gdn_qkvz(
        qkvz, qkv_size=qkv_size, local_num_v_heads=n_v, head_v_dim=d_v
    )
    (cs_new, rs_new), out_new = gdn_wrapper.fused_conv1d_gdn(
        qkv_3d,
        b,
        a,
        (jnp.array(conv_state), jnp.array(rec_state)),
        conv_w,
        None,
        a_log,
        dt_bias,
        q_start,
        s_idx,
        dist,
        s_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    assert out_new.dtype == jnp.bfloat16
    assert jnp.array_equal(z_3d, qkvz[:, qkv_size:].reshape(rows, n_v, d_v))
    assert jnp.array_equal(out_new, out_ref)
    assert jnp.array_equal(cs_new, cs_ref)
    assert jnp.array_equal(rs_new, rs_ref)
