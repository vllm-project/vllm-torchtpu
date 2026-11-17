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
"""Tests for INT8 block-16 GDN recurrent state compression."""

from __future__ import annotations

import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.gdn_pool_layout import (
    gdn_ssm_quant_u32_rows,
    pooled_gdn_ssm_state_bytes,
)


def test_gdn_ssm_quant_u32_rows_and_bytes():
    assert gdn_ssm_quant_u32_rows(128, "") is None
    # 128 // 4 = 32 value u32 rows + 128 // 16 = 8 FP32 scale rows = 40 (5x T(8,128))
    assert gdn_ssm_quant_u32_rows(128, "int8_block16") == 40
    assert gdn_ssm_quant_u32_rows(256, "int8_block16") == 80

    fp32_bytes = pooled_gdn_ssm_state_bytes(
        num_v_heads=32, head_k_dim=128, head_v_dim=128, dtype="torch.float32"
    )
    int8_bytes = pooled_gdn_ssm_state_bytes(
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        dtype="torch.float32",
        quant_mode="int8_block16",
    )
    assert fp32_bytes == 32 * 128 * 128 * 4
    assert int8_bytes == 32 * 40 * 128 * 4
    assert fp32_bytes / int8_bytes == pytest.approx(3.2)


def test_gdn_ssm_quant_validation(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported GDN SSM state quant mode"):
        gdn_ssm_quant_u32_rows(128, "bogus")
    with pytest.raises(ValueError, match="divisible by 128"):
        gdn_ssm_quant_u32_rows(64, "int8_block16")

    monkeypatch.setenv("TPU_GDN_SSM_STATE_QUANT", "int8_block16")
    assert envs.TPU_GDN_SSM_STATE_QUANT == "int8_block16"
    monkeypatch.setenv("TPU_GDN_SSM_STATE_QUANT", "invalid_mode")
    with pytest.raises(ValueError, match="Invalid value"):
        _ = envs.TPU_GDN_SSM_STATE_QUANT


@pytest.mark.tpu_test
def test_gdn_v3_kernel_int8_block16_recurrent_state_roundtrip():
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.gdn.v3 import config, tiling, wrapper

    quant_mode = "int8_block16"
    assert tiling._rec_state_bytes(jnp.float32, quant_mode) == 1.25

    n_kq, n_v, d_k, d_v, k_sz = 2, 4, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qkv = jax.random.normal(k1, (16, dim), dtype=jnp.bfloat16) * 0.2
    b = jax.random.normal(k2, (16, n_v), dtype=jnp.bfloat16) * 0.1
    a = jax.random.normal(k3, (16, n_v), dtype=jnp.bfloat16) * 0.1
    conv_w = jax.random.normal(k4, (dim, 1, k_sz), dtype=jnp.bfloat16) * 0.1
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)

    q_start = jnp.array([0, 16], dtype=jnp.int32)
    s_idx = jnp.array([1], dtype=jnp.int32)
    dist = jnp.array([0, 0, 1], dtype=jnp.int32)
    s_lens = jnp.array([16], dtype=jnp.int32)

    conv_ref = jnp.zeros((2, k_sz - 1, dim), dtype=jnp.float32)
    rec_ref = jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)
    (conv_ref, rec_ref), _ = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        (conv_ref, rec_ref),
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
        kernel_size=k_sz,
        prefill_only=True,
        ssm_state_quant="",
    )

    u32_rows = gdn_ssm_quant_u32_rows(d_k, quant_mode)
    assert u32_rows == 40
    conv_q = jnp.zeros((2, k_sz - 1, dim), dtype=jnp.float32)
    rec_q = jnp.zeros((2, n_v, u32_rows, d_v), dtype=jnp.float32)
    (conv_q, rec_q), _ = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        (conv_q, rec_q),
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
        kernel_size=k_sz,
        prefill_only=True,
        ssm_state_quant=quant_mode,
    )

    # Single-token decode step reading back the quantized state from HBM:
    q_dec = qkv[:1]
    b_dec = b[:1]
    a_dec = a[:1]
    q_start_d = jnp.array([0, 1], dtype=jnp.int32)
    dist_d = jnp.array([1, 1, 1], dtype=jnp.int32)
    s_lens_d = jnp.array([17], dtype=jnp.int32)

    _, out_ref = wrapper.fused_conv1d_gdn(
        q_dec,
        b_dec,
        a_dec,
        (conv_ref, rec_ref),
        conv_w,
        None,
        a_log,
        dt_bias,
        q_start_d,
        s_idx,
        dist_d,
        s_lens_d,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=k_sz,
        batched_only=True,
        ssm_state_quant="",
    )
    _, out_q = wrapper.fused_conv1d_gdn(
        q_dec,
        b_dec,
        a_dec,
        (conv_q, rec_q),
        conv_w,
        None,
        a_log,
        dt_bias,
        q_start_d,
        s_idx,
        dist_d,
        s_lens_d,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=k_sz,
        batched_only=True,
        ssm_state_quant=quant_mode,
    )
    cos = float(
        jnp.sum(out_ref.astype(jnp.float32) * out_q.astype(jnp.float32))
        / (
            jnp.linalg.norm(out_ref.astype(jnp.float32))
            * jnp.linalg.norm(out_q.astype(jnp.float32))
            + 1e-12
        )
    )
    assert cos > 0.999, f"{quant_mode} cosine {cos} < 0.999"
    cfg = config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            jnp.float32, jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32
        ),
        batch_size=1,
        dim_size=dim,
        kernel_size=k_sz,
        tile_size=1,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        is_shared=True,
        ssm_state_quant=quant_mode,
    )
    assert cfg.recurrent_u32_rows == u32_rows
    assert quant_mode in cfg.get_kernel_name()
