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
"""Unit and TPU tests for RPA v3 FP32 online-softmax accumulation."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_v3

_MIB = 1024 * 1024


def _build_multi_tile_workload(
    kv_len: int = 256,
    q_len: int = 8,
    num_q_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 16,
    seed: int = 42,
) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    total_pages = kv_len // page_size
    q = jnp.asarray(
        rng.standard_normal((q_len, num_q_heads, head_dim), dtype=np.float32) * 0.5,
        dtype=jnp.bfloat16,
    )
    k_hist = jnp.asarray(
        rng.standard_normal((kv_len, num_kv_heads, head_dim), dtype=np.float32) * 0.5,
        dtype=jnp.bfloat16,
    )
    # Add structured token-dependent bias to V so multi-tile softmax rounding is
    # directly observable against exact FP32 attention.
    token_ramp = np.linspace(-2.0, 2.0, kv_len, dtype=np.float32)[:, None, None]
    v_hist = jnp.asarray(
        rng.standard_normal((kv_len, num_kv_heads, head_dim), dtype=np.float32) * 0.25
        + token_ramp,
        dtype=jnp.bfloat16,
    )
    cache_shape = rpa_v3.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, jnp.bfloat16
    )
    kv_cache = rpa_v3.merge_kv(k_hist, v_hist).reshape(cache_shape)
    return {
        "q": q,
        "k_new": k_hist[kv_len - q_len :],
        "v_new": v_hist[kv_len - q_len :],
        "k_hist": k_hist,
        "v_hist": v_hist,
        "kv_cache": kv_cache,
        "kv_lens": jnp.array([kv_len], dtype=jnp.int32),
        "page_indices": jnp.arange(total_pages, dtype=jnp.int32),
        "cu_q_lens": jnp.array([0, q_len], dtype=jnp.int32),
        "distribution": jnp.array([0, 0, 1], dtype=jnp.int32),
    }


def _tiled_online_softmax_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    sm_scale: float,
    tile_size: int,
    state_dtype: jnp.dtype,
    mask_value: float,
) -> jax.Array:
    """Mirror `_ragged_paged_attention_kernel_loop` multi-KV-tile accumulation."""
    q_len, num_q_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    num_tiles = kv_len // tile_size
    gqa = num_q_heads // num_kv_heads
    k_tiles = jnp.repeat(k, gqa, axis=1).reshape(
        num_tiles, tile_size, num_q_heads, head_dim
    )
    v_tiles = jnp.repeat(v, gqa, axis=1).reshape(
        num_tiles, tile_size, num_q_heads, head_dim
    )
    q_pos = (kv_len - q_len) + jnp.arange(q_len, dtype=jnp.int32)[None, :, None]
    tile_starts = jnp.arange(0, kv_len, tile_size, dtype=jnp.int32)

    def step(carry, inputs):
        m, l_sum, acc = carry
        start, k_tile, v_tile = inputs
        s = jnp.einsum(
            "qhd,khd->hqk", q, k_tile, preferred_element_type=jnp.float32
        ).astype(state_dtype)
        s = s * sm_scale
        k_pos = (start + jnp.arange(tile_size, dtype=jnp.int32))[None, None, :]
        s = jnp.where(q_pos >= k_pos, s, mask_value)
        m_curr = jnp.maximum(m, jnp.max(s, axis=-1, keepdims=True))
        p = jnp.exp(s - m_curr)
        exp_m_diff = jnp.exp(m - m_curr)
        l_next = exp_m_diff * l_sum + jnp.sum(
            p, axis=-1, keepdims=True, dtype=state_dtype
        )
        pv = jnp.einsum(
            "hqk,khd->hqd", p, v_tile, preferred_element_type=jnp.float32
        ).astype(state_dtype)
        return (m_curr, l_next, exp_m_diff * acc + pv), None

    init = (
        jnp.full((num_q_heads, q_len, 1), -jnp.inf, dtype=state_dtype),
        jnp.zeros((num_q_heads, q_len, 1), dtype=state_dtype),
        jnp.zeros((num_q_heads, q_len, head_dim), dtype=state_dtype),
    )
    (_, l_sum, acc), _ = jax.lax.scan(step, init, (tile_starts, k_tiles, v_tiles))
    return jnp.swapaxes((acc / l_sum).astype(q.dtype), 0, 1)


def _legacy_bf16_ref_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, sm_scale: float
) -> jax.Array:
    """Reproduce the pre-fix bf16-truncated softmax and PV accumulation."""
    q_len, num_q_heads, _ = q.shape
    kv_len, num_kv_heads, _ = k.shape
    k_rep = jnp.repeat(k, num_q_heads // num_kv_heads, axis=1)
    v_rep = jnp.repeat(v, num_q_heads // num_kv_heads, axis=1)
    attn = jnp.einsum(
        "qhd,khd->hqk", q, k_rep, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    attn = attn * sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    attn = jnp.where(q_span >= kv_span, attn, float(jnp.finfo(jnp.bfloat16).min))
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    return jnp.einsum("hqk,khd->qhd", attn, v_rep).astype(jnp.bfloat16)


def test_ref_fp32_softmax_beats_legacy_bf16_and_preserves_dtypes() -> None:
    wl = _build_multi_tile_workload(kv_len=128, q_len=8, page_size=16)
    sm_scale = 128**-0.5

    f32_cache_shape = rpa_v3.get_kv_cache_shape(8, 16, 4, 128, jnp.float32)
    f32_kv_cache = rpa_v3.merge_kv(
        wl["k_hist"].astype(jnp.float32), wl["v_hist"].astype(jnp.float32)
    ).reshape(f32_cache_shape)
    exact_f32, _ = rpa_v3.ref_ragged_paged_attention(
        wl["q"].astype(jnp.float32),
        wl["k_new"].astype(jnp.float32),
        wl["v_new"].astype(jnp.float32),
        f32_kv_cache,
        wl["kv_lens"],
        wl["page_indices"],
        wl["cu_q_lens"],
        wl["distribution"],
        sm_scale=sm_scale,
    )
    ref_default, _ = rpa_v3.ref_ragged_paged_attention(
        wl["q"],
        wl["k_new"],
        wl["v_new"],
        wl["kv_cache"],
        wl["kv_lens"],
        wl["page_indices"],
        wl["cu_q_lens"],
        wl["distribution"],
        sm_scale=sm_scale,
    )
    ref_explicit_f32, _ = rpa_v3.ref_ragged_paged_attention(
        wl["q"],
        wl["k_new"],
        wl["v_new"],
        wl["kv_cache"],
        wl["kv_lens"],
        wl["page_indices"],
        wl["cu_q_lens"],
        wl["distribution"],
        sm_scale=sm_scale,
        out_dtype=jnp.float32,
    )
    assert ref_default.dtype == jnp.bfloat16
    assert ref_explicit_f32.dtype == jnp.float32

    tiled_fp32 = _tiled_online_softmax_attention(
        wl["q"],
        wl["k_hist"],
        wl["v_hist"],
        sm_scale=sm_scale,
        tile_size=16,
        state_dtype=jnp.float32,
        mask_value=rpa_v3.DEFAULT_MASK_VALUE,
    )
    tiled_legacy_bf16 = _tiled_online_softmax_attention(
        wl["q"],
        wl["k_hist"],
        wl["v_hist"],
        sm_scale=sm_scale,
        tile_size=16,
        state_dtype=jnp.bfloat16,
        mask_value=float(jnp.finfo(jnp.bfloat16).min),
    )
    legacy_ref_bf16 = _legacy_bf16_ref_attention(
        wl["q"], wl["k_hist"], wl["v_hist"], sm_scale=sm_scale
    )

    exact_np = np.asarray(exact_f32)
    err_ref_default = np.mean(np.abs(np.asarray(ref_default, np.float32) - exact_np))
    err_ref_f32 = np.mean(np.abs(np.asarray(ref_explicit_f32) - exact_np))
    err_legacy_ref = np.mean(np.abs(np.asarray(legacy_ref_bf16, np.float32) - exact_np))
    err_tiled_fp32 = np.mean(np.abs(np.asarray(tiled_fp32, np.float32) - exact_np))
    err_tiled_bf16 = np.mean(
        np.abs(np.asarray(tiled_legacy_bf16, np.float32) - exact_np)
    )

    assert err_ref_f32 < 1e-6
    assert err_ref_default < err_legacy_ref * 0.4
    assert err_tiled_fp32 < err_tiled_bf16 * 0.25
    np.testing.assert_allclose(
        np.asarray(tiled_fp32, np.float32),
        np.asarray(ref_default, np.float32),
        atol=1e-3,
        rtol=1e-2,
    )


def test_rpa_kernel_plumbing_fp32_scratch_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = jnp.zeros((8, 32, 256), dtype=jnp.bfloat16)
    kv = jnp.zeros((8, 8, 256), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros(
        rpa_v3.get_kv_cache_shape(256, 16, 8, 256, jnp.bfloat16), dtype=jnp.bfloat16
    )
    kv_lens = jnp.array([128], dtype=jnp.int32)
    page_indices = jnp.arange(256, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 8], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
    captured_calls: list[dict[str, object]] = []

    def fake_pallas_call(kernel_fn, *, grid_spec, compiler_params, out_shape, **_kw):
        captured_calls.append(
            {
                "keywords": kernel_fn.keywords,
                "scratch_shapes": grid_spec.scratch_shapes,
                "vmem_limit_bytes": compiler_params.vmem_limit_bytes,
                "out_shape": out_shape,
            }
        )
        return lambda *_args: (
            jnp.zeros(out_shape[0].shape, dtype=out_shape[0].dtype),
            jnp.zeros(out_shape[1].shape, dtype=out_shape[1].dtype),
        )

    monkeypatch.setattr(rpa_v3, "get_tpu_version", lambda: 6)
    monkeypatch.setattr(
        rpa_v3.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=32 * _MIB),
    )
    monkeypatch.setattr(rpa_v3.pl, "pallas_call", fake_pallas_call)

    out_bf16, _ = rpa_v3.ragged_paged_attention.__wrapped__(
        q,
        kv,
        kv,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=256**-0.5,
        chunk_prefill_size=64,
    )
    assert out_bf16.dtype == jnp.bfloat16
    assert len(captured_calls) == 3

    for call in captured_calls:
        scratch = call["scratch_shapes"]
        # l_scratch, m_scratch, acc_scratch must always accumulate in float32.
        assert scratch[4].dtype == jnp.float32
        assert scratch[5].dtype == jnp.float32
        assert scratch[6].dtype == jnp.float32
        # Inner HBM output stays q.dtype so packed int32 bitcast store succeeds.
        assert call["out_shape"][0].dtype == jnp.bfloat16
        kw = call["keywords"]
        assert kw["mask_value"] == rpa_v3.DEFAULT_MASK_VALUE
        # CompilerParams still receives full VMEM capacity while tiles stay within budget.
        assert call["vmem_limit_bytes"] == 32 * _MIB
        vmem = rpa_v3.get_vmem_estimate_bytes(
            8, 4, 256, kw["bq_sz"], kw["bkv_sz"], jnp.bfloat16, jnp.bfloat16
        )
        assert vmem <= int(32 * _MIB * rpa_v3._VMEM_CLAMP_UTILIZATION)

    # Decode case uses bq_sz=1; prefill/mixed with 8 tokens use bq_sz=4, bq_csz=4.
    assert captured_calls[0]["keywords"]["bq_sz"] == 1
    assert captured_calls[1]["keywords"]["bq_sz"] == 4
    assert captured_calls[1]["keywords"]["bq_csz"] == 4
    assert captured_calls[2]["keywords"]["bq_sz"] == 4
    assert captured_calls[2]["keywords"]["bq_csz"] == 4

    out_f32, _ = rpa_v3.ragged_paged_attention.__wrapped__(
        q,
        kv,
        kv,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=256**-0.5,
        out_dtype=jnp.float32,
    )
    assert out_f32.dtype == jnp.float32


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("case_name", "q_len", "distribution"),
    [
        ("decode", 1, (1, 1, 1)),
        ("prefill", 16, (0, 1, 1)),
        ("mixed", 8, (0, 0, 1)),
    ],
)
@pytest.mark.parametrize("out_dtype", [None, jnp.float32])
def test_tpu_rpa_v3_multi_tile_fp32_softmax_accuracy(
    case_name: str,
    q_len: int,
    distribution: tuple[int, int, int],
    out_dtype: jnp.dtype | None,
) -> None:
    """Run the real RPA v3 Pallas kernel on multi-tile BF16 attention (kv_len > bkv_sz)
    and verify accuracy against FP32 reference across decode, prefill, and mixed modes."""
    kv_len = 512
    page_size = 16
    wl = _build_multi_tile_workload(
        kv_len=kv_len,
        q_len=q_len,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        page_size=page_size,
        seed=42,
    )
    sm_scale = 128**-0.5
    dist = jnp.array(distribution, dtype=jnp.int32)
    chunk_prefill_size = q_len if case_name == "prefill" else None
    # Force 8 KV tiles (bkv_sz=64, bkv_csz=32) so online-softmax rescaling executes
    # across many tiles — old BF16 scratch accumulates >0.01 MAE here.
    block_sizes = (1, 64, 1, 32) if q_len == 1 else (min(q_len, 8), 64, min(q_len, 8), 32)

    f32_cache_shape = rpa_v3.get_kv_cache_shape(
        kv_len // page_size, page_size, 4, 128, jnp.float32
    )
    f32_kv_cache = rpa_v3.merge_kv(
        wl["k_hist"].astype(jnp.float32), wl["v_hist"].astype(jnp.float32)
    ).reshape(f32_cache_shape)
    # Compute reference before calling ragged_paged_attention, which donates q/k/v/kv_cache.
    ref_f32, _ = rpa_v3.ref_ragged_paged_attention(
        wl["q"].astype(jnp.float32),
        wl["k_new"].astype(jnp.float32),
        wl["v_new"].astype(jnp.float32),
        f32_kv_cache,
        wl["kv_lens"],
        wl["page_indices"],
        wl["cu_q_lens"],
        dist,
        sm_scale=sm_scale,
    )

    out, _ = rpa_v3.ragged_paged_attention(
        wl["q"],
        wl["k_new"],
        wl["v_new"],
        wl["kv_cache"],
        wl["kv_lens"],
        wl["page_indices"],
        wl["cu_q_lens"],
        dist,
        sm_scale=sm_scale,
        out_dtype=out_dtype,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=block_sizes,
        p_block_sizes=block_sizes,
        m_block_sizes=block_sizes,
    )
    expected_dtype = jnp.float32 if out_dtype == jnp.float32 else jnp.bfloat16
    assert out.dtype == expected_dtype

    err = np.abs(np.asarray(out, dtype=np.float32) - np.asarray(ref_f32, dtype=np.float32))
    # Tight enough that pre-fix BF16 online-softmax scratch (>2e-3 MAE, >1e-2 max) fails.
    assert float(np.mean(err)) < 5e-4
    assert float(np.max(err)) < 3e-3

