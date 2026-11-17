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
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest
from vllm_torchtpu.gdn_pool_layout import (
    gdn_ssm_quant_u32_rows,
    pooled_gdn_ssm_state_bytes,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


def test_gdn_ssm_quant_u32_rows_and_bytes():
    assert gdn_ssm_quant_u32_rows(128, "") is None
    # 128 // 4 = 32 value u32 rows + 128 // 16 = 8 FP32 scale rows = 40 (5x T(8,128))
    assert gdn_ssm_quant_u32_rows(128, "int8_block16") == 40
    assert gdn_ssm_quant_u32_rows(256, "int8_block16") == 80

    # Symmetric d_k=128, d_v=128 and asymmetric d_k=128, d_v=256.
    for d_k, d_v in ((128, 128), (128, 256)):
        fp32_bytes = pooled_gdn_ssm_state_bytes(
            num_v_heads=32, head_k_dim=d_k, head_v_dim=d_v, dtype="torch.float32"
        )
        int8_bytes = pooled_gdn_ssm_state_bytes(
            num_v_heads=32,
            head_k_dim=d_k,
            head_v_dim=d_v,
            dtype="torch.float32",
            quant_mode="int8_block16",
        )
        assert fp32_bytes == 32 * d_k * d_v * 4
        assert int8_bytes == 32 * 40 * d_v * 4
        assert fp32_bytes / int8_bytes == pytest.approx(3.2)


def test_gdn_ssm_quant_validation_and_platform_env_var(monkeypatch):
    assert "TPU_GDN_SSM_STATE_QUANT" in TpuPlatform.additional_env_vars

    with pytest.raises(ValueError, match="Unsupported GDN SSM state quant mode"):
        gdn_ssm_quant_u32_rows(128, "bogus")
    with pytest.raises(ValueError, match="divisible by 128"):
        gdn_ssm_quant_u32_rows(64, "int8_block16")

    monkeypatch.setenv("TPU_GDN_SSM_STATE_QUANT", "int8_block16")
    assert envs.TPU_GDN_SSM_STATE_QUANT == "int8_block16"
    monkeypatch.setenv("TPU_GDN_SSM_STATE_QUANT", "invalid_mode")
    with pytest.raises(ValueError, match="Invalid value"):
        _ = envs.TPU_GDN_SSM_STATE_QUANT


def test_gdn_attention_state_shape_asymmetric_and_pcp_disable(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    layer = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.num_k_heads = 2
    layer.num_v_heads = 4
    layer.head_k_dim = 128
    layer.head_v_dim = 256
    layer.tp_size = 1
    layer._pcp_streaming_configured = False

    # Upstream gated_delta_net_state_shape returns
    # (local_v_heads, head_v_dim, head_k_dim).
    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "get_state_shape",
        lambda self: ((3, 2 * 2 * 128 + 4 * 256), (4, 256, 128)),
    )
    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "get_state_dtype",
        lambda self: (torch.bfloat16, torch.bfloat16),
    )
    monkeypatch.setattr(
        VllmGatedDeltaNetAttention,
        "_pcp_streaming_enabled",
        staticmethod(lambda: False),
    )

    monkeypatch.setenv("TPU_GDN_SSM_STATE_QUANT", "int8_block16")
    _, ssm_shape = layer.get_state_shape()
    assert ssm_shape == (4, 40, 256)
    assert layer.get_state_dtype() == (torch.float32, torch.float32)

    # Under PCP > 1, state quantization must stand down uniformly.
    layer._pcp_streaming_configured = True
    _, pcp_ssm_shape = layer.get_state_shape()
    assert pcp_ssm_shape == (4, 256, 128)
    assert layer.get_state_dtype() == (torch.float32, torch.bfloat16)


def test_raiden_rejects_quantized_gdn_state(monkeypatch):
    monkeypatch.setenv("TPU_GDN_SSM_STATE_QUANT", "int8_block16")
    with pytest.raises(NotImplementedError, match="TPU_GDN_SSM_STATE_QUANT"):
        pool_manifest._gdn_ssm_regions(ssm_shape=(2, 4, 40, 128), itemsize=4)

    geom = pool_manifest.GdnHeadGeometry(
        local_key_heads=2,
        local_value_heads=4,
        key_head_dim=128,
        value_head_dim=128,
    )
    with pytest.raises(NotImplementedError, match="TPU_GDN_SSM_STATE_QUANT"):
        pool_manifest._pooled_gdn_state_views(
            pool=torch.zeros(4),
            spec=None,
            raw_tensors=(),
            layer_name="model.layers.0.linear_attn",
            gdn_geometry=geom,
        )


def _alloc_states(
    *,
    is_shared: bool,
    num_blocks: int,
    n_v: int,
    d_k: int,
    d_v: int,
    k_sz: int,
    dim: int,
    quant_mode: str,
):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if not is_shared:
        rows = gdn_ssm_quant_u32_rows(d_k, quant_mode) or d_k
        return (
            jnp.zeros((num_blocks, k_sz - 1, dim), dtype=jnp.float32),
            jnp.zeros((num_blocks, n_v, rows, d_v), dtype=jnp.float32),
        )
    num_lanes = pltpu.get_tpu_info().num_lanes
    rows = gdn_ssm_quant_u32_rows(d_k, quant_mode) or d_k
    granules = ((n_v * rows * d_v) // num_lanes) + pl.cdiv((k_sz - 1) * dim, num_lanes)
    return jnp.zeros((num_blocks, granules, 1, num_lanes), dtype=jnp.float32)


@pytest.mark.tpu_test
@pytest.mark.parametrize("is_shared", [False, True])
@pytest.mark.parametrize(
    "seq_len,d_k,d_v",
    [
        (16, 128, 128),
        (128, 128, 128),
        (16, 256, 128),
    ],
)
def test_gdn_v3_kernel_int8_block16_recurrent_state_roundtrip(
    is_shared: bool, seq_len: int, d_k: int, d_v: int
):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.gdn.v3 import config, tiling, wrapper

    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")

    quant_mode = "int8_block16"
    assert tiling._rec_state_bytes(jnp.float32, quant_mode, d_k=d_k) == 1.25

    n_kq, n_v, k_sz = 2, 4, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qkv = jax.random.normal(k1, (seq_len, dim), dtype=jnp.bfloat16) * 0.2
    b = jax.random.normal(k2, (seq_len, n_v), dtype=jnp.bfloat16) * 0.1
    a = jax.random.normal(k3, (seq_len, n_v), dtype=jnp.bfloat16) * 0.1
    conv_w = jax.random.normal(k4, (dim, 1, k_sz), dtype=jnp.bfloat16) * 0.1
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)

    q_start = jnp.array([0, seq_len], dtype=jnp.int32)
    s_idx = jnp.array([1], dtype=jnp.int32)
    dist = jnp.array([0, 0, 1], dtype=jnp.int32)
    s_lens = jnp.array([seq_len], dtype=jnp.int32)
    shared_kw = (
        dict(conv_dtype=jnp.float32, recurrent_dtype=jnp.float32) if is_shared else {}
    )

    states_ref = _alloc_states(
        is_shared=is_shared,
        num_blocks=2,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        k_sz=k_sz,
        dim=dim,
        quant_mode="",
    )
    states_ref, prefill_ref = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        states_ref,
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
        mixed_tile_size=64,
        prefill_only=True,
        ssm_state_quant="",
        **shared_kw,
    )

    states_q = _alloc_states(
        is_shared=is_shared,
        num_blocks=2,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        k_sz=k_sz,
        dim=dim,
        quant_mode=quant_mode,
    )
    states_q, prefill_q = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        states_q,
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
        mixed_tile_size=64,
        prefill_only=True,
        ssm_state_quant=quant_mode,
        **shared_kw,
    )
    assert float(jnp.max(jnp.abs(prefill_ref - prefill_q))) < 1e-2

    # Decode step reading back the quantized state from HBM:
    q_dec, b_dec, a_dec = qkv[:1], b[:1], a[:1]
    q_start_d = jnp.array([0, 1], dtype=jnp.int32)
    dist_d = jnp.array([1, 1, 1], dtype=jnp.int32)
    s_lens_d = jnp.array([seq_len + 1], dtype=jnp.int32)

    _, out_ref = wrapper.fused_conv1d_gdn(
        q_dec,
        b_dec,
        a_dec,
        states_ref,
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
        **shared_kw,
    )
    _, out_q = wrapper.fused_conv1d_gdn(
        q_dec,
        b_dec,
        a_dec,
        states_q,
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
        **shared_kw,
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
        is_shared=is_shared,
        ssm_state_quant=quant_mode,
    )
    assert cfg.recurrent_u32_rows == gdn_ssm_quant_u32_rows(d_k, quant_mode)
    assert quant_mode in cfg.get_kernel_name()


@pytest.mark.tpu_test
def test_gdn_v3_kernel_int8_block16_spec_decode_window():
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")

    quant_mode = "int8_block16"
    n_kq, n_v, d_k, d_v, k_sz = 2, 4, 128, 128, 4
    num_spec_tokens = 2
    win = num_spec_tokens + 1
    dim = 2 * n_kq * d_k + n_v * d_v
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qkv = jax.random.normal(k1, (win, dim), dtype=jnp.bfloat16) * 0.2
    b = jax.random.normal(k2, (win, n_v), dtype=jnp.bfloat16) * 0.1
    a = jax.random.normal(k3, (win, n_v), dtype=jnp.bfloat16) * 0.1
    conv_w = jax.random.normal(k4, (dim, 1, k_sz), dtype=jnp.bfloat16) * 0.1
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)

    q_start = jnp.array([0, win], dtype=jnp.int32)
    s_idx = jnp.array([0], dtype=jnp.int32)
    read_off = jnp.array([0], dtype=jnp.int32)
    ckpt_idx = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    dist = jnp.array([1, 1, 1], dtype=jnp.int32)
    s_lens = jnp.array([win], dtype=jnp.int32)

    states_ref = _alloc_states(
        is_shared=True,
        num_blocks=4,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        k_sz=k_sz,
        dim=dim,
        quant_mode="",
    )
    states_q = _alloc_states(
        is_shared=True,
        num_blocks=4,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        k_sz=k_sz,
        dim=dim,
        quant_mode=quant_mode,
    )
    states_ref, out_ref = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        states_ref,
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
        num_spec_tokens=num_spec_tokens,
        read_offsets=read_off,
        ckpt_indices=ckpt_idx,
        batched_only=True,
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.float32,
        ssm_state_quant="",
    )
    states_q, out_q = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        states_q,
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
        num_spec_tokens=num_spec_tokens,
        read_offsets=read_off,
        ckpt_indices=ckpt_idx,
        batched_only=True,
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.float32,
        ssm_state_quant=quant_mode,
    )
    assert float(jnp.max(jnp.abs(out_ref - out_q))) < 1e-2

    # Rollback verify step reading checkpoint index 2 (draft token 1 accepted):
    s_idx_rb = jnp.array([2], dtype=jnp.int32)
    ckpt_idx_rb = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    _, next_ref = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        states_ref,
        conv_w,
        None,
        a_log,
        dt_bias,
        q_start,
        s_idx_rb,
        dist,
        jnp.array([win + 2], dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=k_sz,
        num_spec_tokens=num_spec_tokens,
        read_offsets=read_off,
        ckpt_indices=ckpt_idx_rb,
        batched_only=True,
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.float32,
        ssm_state_quant="",
    )
    _, next_q = wrapper.fused_conv1d_gdn(
        qkv,
        b,
        a,
        states_q,
        conv_w,
        None,
        a_log,
        dt_bias,
        q_start,
        s_idx_rb,
        dist,
        jnp.array([win + 2], dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=k_sz,
        num_spec_tokens=num_spec_tokens,
        read_offsets=read_off,
        ckpt_indices=ckpt_idx_rb,
        batched_only=True,
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.float32,
        ssm_state_quant=quant_mode,
    )
    cos = float(
        jnp.sum(next_ref.astype(jnp.float32) * next_q.astype(jnp.float32))
        / (
            jnp.linalg.norm(next_ref.astype(jnp.float32))
            * jnp.linalg.norm(next_q.astype(jnp.float32))
            + 1e-12
        )
    )
    assert cos > 0.999, f"spec rollback cosine {cos} < 0.999"
