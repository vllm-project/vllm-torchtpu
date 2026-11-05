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
"""Tests for batched RPA opt-in FP8 QK/PV prefill compute."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    flash_attention,
    schedule,
    wrapper,
)

FP8 = jnp.float8_e4m3fn
_VMEM_64M = 1 << 26
_VMEM_32M = 1 << 25


def _cfgs(num_q_heads: int, num_kv_heads: int, dtype_q: jnp.dtype = FP8):
    model = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=512,
        mask_value=-1e30,
    )
    serve = configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=16384,
        num_page_indices=8 * 640,
        dtype_q=dtype_q,
        dtype_kv=FP8,
        dtype_out=jnp.bfloat16,
        fp8_compute=True,
    )
    return model, serve


@pytest.mark.parametrize(
    "heads,expected",
    [
        ((4, 1), configs.BlockSizes(2048, 64, 512, 1)),
        ((16, 2), configs.BlockSizes(512, 32, 512, 1)),
        ((32, 4), configs.BlockSizes(256, 32, 512, 1)),
        ((8, 1), configs.BlockSizes(1024, 32, 512, 1)),
    ],
)
def test_fp8_prefill_block_sizes(heads, expected):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(*_cfgs(*heads), _VMEM_64M)
    assert blocks == expected


@pytest.mark.parametrize("sliding_window", [512, None])
def test_fp8_prefill_block_sizes_sliding_window_cap(sliding_window):
    model, serve = _cfgs(4, 1)
    model = dataclasses.replace(model, sliding_window=sliding_window)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(model, serve, _VMEM_64M)
    if sliding_window:
        assert blocks == configs.BlockSizes(256, 64, 256, 1)
    else:
        assert blocks == configs.BlockSizes(2048, 64, 512, 1)


def test_fp8_prefill_block_sizes_shrink_to_fit_vmem():
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(*_cfgs(4, 1), _VMEM_32M)
        cfgs = configs.RpaConfigs(
            block=blocks,
            model=_cfgs(4, 1)[0],
            serve=_cfgs(4, 1)[1],
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=_VMEM_32M,
        )
        used = wrapper.fp8_prefill_vmem_bytes(cfgs)
    assert blocks.bq_sz < 2048
    assert used <= wrapper.FP8_PREFILL_BUFFER_VMEM_FRACTION * _VMEM_32M


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(kv_dtype=jnp.bfloat16), "fp8 KV cache"),
        (dict(num_q_heads=4, num_kv_heads=2), "multiple of 4"),
        (dict(q_scale=0.5), "q_scale"),
        (dict(return_lse=True), "return_lse"),
        (dict(use_per_token_scale=True), "per-token"),
    ],
)
def test_validate_fp8_compute_rejects(kwargs, match):
    args = dict(
        kv_dtype=FP8, num_q_heads=32, num_kv_heads=4, q_scale=None, return_lse=False
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        wrapper.validate_fp8_compute(**args)


def test_quantize_queries_keeps_head_order_and_saturates():
    kv_heads, tokens, g, d = 2, 3, 8, 128
    q = jnp.arange(kv_heads * tokens * g * d, dtype=jnp.float32)
    q = (q.reshape(kv_heads, tokens, g, d) % 97 - 48) / 8.0
    q = q.at[0, 0, 0, 0].set(1000.0).at[1, 2, 7, 5].set(-1000.0)
    q_hbm = q.astype(jnp.bfloat16).reshape(kv_heads, tokens, g // 2, 2, d)

    q8 = wrapper.quantize_queries(q_hbm, FP8)

    assert q8.shape == (kv_heads, tokens, g // 4, 4, d)
    assert q8.dtype == FP8
    got = np.asarray(q8.astype(jnp.float32)).reshape(kv_heads, tokens, g, d)
    want = np.array(q.astype(jnp.bfloat16).astype(FP8).astype(jnp.float32))
    want[0, 0, 0, 0] = 448.0
    want[1, 2, 7, 5] = -448.0
    np.testing.assert_array_equal(got, want)


def test_fp8_shifted_p_rowsum_accumulates_finitely_in_bfloat16():
    """Shifted FP8 probabilities stay below e4m3 max (448) and sum finitely in bf16."""
    bkv = 512
    tq = 4
    head_dim = 128
    model, serve = _cfgs(4, 1)
    model = dataclasses.replace(model, head_dim=head_dim, sm_scale=1.0)
    serve = dataclasses.replace(serve, page_size=16, total_q_tokens=64)
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv, batch_size=1),
        model=model,
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=_VMEM_64M,
    )
    q = jnp.zeros((1, 1, tq, head_dim), dtype=FP8)
    k = jnp.zeros((1, 1, bkv, head_dim), dtype=FP8)
    m_prev = jnp.full((1, tq, 128), -jnp.inf, dtype=jnp.bfloat16)
    l_prev = jnp.zeros((1, tq, 128), dtype=jnp.bfloat16)
    is_last_k = schedule.SmemWrapper(
        data=jnp.asarray([0], dtype=jnp.int32), shape=(1, 1)
    )
    mask = [jnp.ones((1, tq, bkv), dtype=jnp.bool_)]

    for _ in range(5):
        p, _, _, l_next, m_prev, l_prev = flash_attention.flash_attention_qk_softmax(
            jnp.int32(0),
            q,
            k,
            m_prev,
            l_prev,
            is_last_k,
            custom_mask=mask,
            cfgs=cfgs,
            bq_start=0,
        )
        assert p.dtype == FP8
    assert l_prev.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(l_prev))
    expected_per_key = float(jnp.exp(jnp.bfloat16(3.5)).astype(FP8).astype(jnp.float32))
    np.testing.assert_allclose(
        np.asarray(l_prev[..., 0], dtype=np.float32),
        5 * bkv * expected_per_key,
        rtol=2e-2,
    )


def test_fp8_mixed_kernel_lowers_for_tpu7x():
    tokens, num_q_heads, num_kv_heads, head_dim, page_size = 1024, 4, 1, 512, 256
    max_seqs, pages_per_seq = 8, 32
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        cache_shape = wrapper.get_kv_cache_shape(
            max_seqs * pages_per_seq, page_size, num_kv_heads, head_dim, FP8
        )
        args = (
            jax.ShapeDtypeStruct((tokens, num_q_heads, head_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), FP8),
            jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), FP8),
            jax.ShapeDtypeStruct(cache_shape, FP8),
            jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )

        def call(*xs):
            return wrapper.ragged_paged_attention(
                *xs, sm_scale=1.0, k_scale=1.0, v_scale=1.0, fp8_compute=True
            )

        lowered = jax.jit(call).trace(*args).lower(lowering_platforms=("tpu",))
    text = lowered.as_text()
    assert "RPAm-snh-p256-b1-q2048-k512" in text
    assert "RPAd-snh-p256" in text


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,num_q_heads,num_kv_heads,head_dim,page_size",
    [
        (256, 4, 1, 128, 16),
        (512, 16, 2, 512, 256),  # Gemma 4 full-attention RPAm-snh-p256-b1-q512-k512
    ],
)
def test_tpu_fp8_compute_prefill_accuracy(
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
):
    max_seqs, pages_per_seq = 4, max(32, (tokens // page_size) + 4)
    key = jax.random.key(7)
    kq, kk, kv = jax.random.split(key, 3)
    queries = jax.random.normal(kq, (tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    keys = jax.random.normal(
        kk, (tokens, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(FP8)
    values = jax.random.normal(
        kv, (tokens, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(FP8)
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq, page_size, num_kv_heads, head_dim, FP8
    )
    kv_cache_init = jnp.zeros(cache_shape, dtype=FP8)
    half = tokens // 2
    cu_q_lens = jnp.asarray([0, half, tokens, tokens, tokens], dtype=jnp.int32)
    kv_lens = jnp.asarray([half, half, 0, 0], dtype=jnp.int32)
    page_indices = jnp.arange(max_seqs * pages_per_seq, dtype=jnp.int32)
    dist = jnp.asarray([0, 0, 2], dtype=jnp.int32)
    sm_scale = head_dim**-0.5

    out_ref, cache_ref = wrapper.ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        sm_scale=sm_scale,
        k_scale=1.0,
        v_scale=1.0,
        fp8_compute=False,
    )
    out_fp8, cache_fp8 = wrapper.ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        sm_scale=sm_scale,
        k_scale=1.0,
        v_scale=1.0,
        fp8_compute=True,
    )
    ref_f32 = np.asarray(out_ref, dtype=np.float32)
    fp8_f32 = np.asarray(out_fp8, dtype=np.float32)
    cos_sim = float(
        np.sum(ref_f32 * fp8_f32) / (np.linalg.norm(ref_f32) * np.linalg.norm(fp8_f32))
    )
    assert cos_sim > 0.995
    np.testing.assert_array_equal(
        np.asarray(cache_fp8.view(jnp.uint8)),
        np.asarray(cache_ref.view(jnp.uint8)),
    )
