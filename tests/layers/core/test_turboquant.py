# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""CPU tests for TurboQuant `(num_blocks, num_kv_heads, page_words, 128)` VREG ops."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.layers.core import turboquant as tq

PRESETS = sorted(TQ_PRESETS)


def _bf16_round_np(x: np.ndarray) -> np.ndarray:
    u32 = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) << 16).view(np.float32)


def _layout_and_tables(preset: str, head_dim: int):
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
    layout = tq.TQLayout.from_tq_config(cfg)
    centroids = get_centroids(head_dim, cfg.centroid_bits).numpy()
    return layout, tq.make_tables(layout, centroids)


def test_tq_layout_validation():
    """TQLayout rejects unsupported head_dim and page_size values with ValueError."""
    fp8_layout, _ = _layout_and_tables("turboquant_k8v4", 128)
    mse_layout, _ = _layout_and_tables("turboquant_4bit_nc", 128)

    assert fp8_layout.supported_page_sizes == (16, 32, 64)
    assert mse_layout.supported_page_sizes == (16, 64)

    for bad_page_size in (8, 24, 128, 256):
        with pytest.raises(ValueError, match="supported page sizes"):
            fp8_layout.page_words_128(bad_page_size)
    with pytest.raises(ValueError, match="supported page sizes"):
        mse_layout.page_words_128(32)
    with pytest.raises(ValueError, match="Cannot recover supported page_size"):
        mse_layout.page_size_from_words_128(999)

    with pytest.raises(ValueError, match="positive multiple of 128"):
        tq.TQLayout(
            head_dim=64,
            key_bits=4,
            value_bits=4,
            norm_correction=True,
            slot_size_aligned=70,
        )
    with pytest.raises(ValueError, match="power-of-two head_dim"):
        tq.TQLayout(
            head_dim=384,
            key_bits=4,
            value_bits=4,
            norm_correction=True,
            slot_size_aligned=390,
        )


@pytest.mark.parametrize(
    ("preset", "page_size", "head_dim"),
    [(p, 16, 128) for p in PRESETS]
    + [
        ("turboquant_k8v4", 32, 128),
        ("turboquant_k8v4", 64, 256),
        ("turboquant_4bit_nc", 64, 256),
    ],
)
def test_vreg_pack_update_dequant_roundtrip(preset, page_size, head_dim):
    """Quantize -> pack -> VREG page update -> VREG dequant approximates Q.K and V."""
    layout, tables = _layout_and_tables(preset, head_dim)
    page_words = layout.page_words_128(page_size)
    assert page_words % 8 == 0
    assert layout.page_size_from_words_128(page_words) == page_size

    rng = np.random.default_rng(3)
    k = rng.standard_normal((page_size, 1, head_dim)).astype(np.float32)
    v = rng.standard_normal((page_size, 1, head_dim)).astype(np.float32)
    q = rng.standard_normal((head_dim,)).astype(np.float32)

    nk_u, nv_i, n_meta = tq._quantize_kv_components(
        jnp.asarray(k), jnp.asarray(v), layout, tables
    )
    packed_toks = tq._pack_new_tq_tokens_128(nk_u, nv_i, n_meta, page_size, layout)
    empty_page = jnp.zeros((page_words, 128), jnp.uint8)
    valid_mask = jnp.ones((page_size, 1), dtype=jnp.bool_)
    page = tq._update_tq_page_vreg(
        empty_page, packed_toks[:, :, 0, :], valid_mask, layout=layout
    )
    dk, dv = tq._dequant_tq_page_vreg(page, jnp.float32, layout=layout, tables=tables)
    dk_unscaled, dv_4, kn_row = tq._dequant_tq_page_vreg(
        [page, page, page, page],
        jnp.float32,
        layout=layout,
        tables=tables,
        return_k_norm=True,
    )
    np.testing.assert_allclose(
        np.asarray(dv_4), np.tile(np.asarray(dv), (4, 1)), rtol=0, atol=0
    )
    dk_reconstructed = (
        np.asarray(dk_unscaled) * np.asarray(kn_row).T
        if kn_row is not None
        else np.asarray(dk_unscaled)
    )
    np.testing.assert_allclose(
        dk_reconstructed, np.tile(np.asarray(dk), (4, 1)), rtol=1e-3, atol=1e-3
    )

    # Bit-for-bit parity against a NumPy transcription of upstream store/dequant math.
    levels = (1 << layout.value_bits) - 1
    v0 = v[:, 0]
    v_min = v0.min(axis=-1, keepdims=True)
    v_max = v0.max(axis=-1, keepdims=True)
    v_scale = np.maximum((v_max - v_min) / levels, 1e-8)
    ref_v_idx = np.clip(np.floor((v0 - v_min) / v_scale + 0.5), 0, levels).astype(
        np.uint8
    )
    np.testing.assert_array_equal(np.asarray(nv_i[:, 0]), ref_v_idx)
    ref_dv = ref_v_idx.astype(np.float32) * _bf16_round_np(v_scale) + _bf16_round_np(
        v_min
    )
    np.testing.assert_array_equal(np.asarray(dv), ref_dv)

    k0 = k[:, 0]
    if layout.is_key_fp8:
        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
        ref_dk = np.asarray(
            jnp.clip(k0, -fp8_max, fp8_max)
            .astype(jnp.float8_e4m3fn)
            .astype(jnp.float32)
        )
        np.testing.assert_array_equal(np.asarray(dk), ref_dk)
    else:
        k_norm = np.sqrt(np.sum(k0 * k0, axis=-1, keepdims=True))
        rot_k = (k0 / (k_norm + 1e-8)) @ tables.hadamard
        ref_k_idx = np.searchsorted(tables.midpoints, rot_k, side="right")
        half_levels = 1 << (layout.key_bits - 1)
        ref_sign_mag = np.where(
            ref_k_idx < half_levels,
            (half_levels - 1 - ref_k_idx) | half_levels,
            ref_k_idx - half_levels,
        ).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(nk_u[:, 0]), ref_sign_mag)
        c_vec = tables.centroids[ref_k_idx]
        if layout.norm_correction:
            k_norm = k_norm / np.sqrt(
                np.sum(c_vec * c_vec, axis=-1, keepdims=True) + 1e-16
            )
        ref_dk = _bf16_round_np(c_vec) * _bf16_round_np(k_norm)
        np.testing.assert_array_equal(np.asarray(dk), ref_dk)

    q_in = q if layout.is_key_fp8 else np.asarray(tq.rotate(q, tables.hadamard))
    approx_scores = np.asarray(dk) @ q_in
    exact_scores = k[:, 0] @ q
    rel_k = np.linalg.norm(approx_scores - exact_scores) / np.linalg.norm(exact_scores)
    assert rel_k < {8: 0.05, 4: 0.16, 3: 0.28}[layout.key_bits], rel_k

    rel_v = np.linalg.norm(np.asarray(dv) - v[:, 0]) / np.linalg.norm(v[:, 0])
    assert rel_v < {4: 0.12, 3: 0.24}[layout.value_bits], rel_v


@pytest.mark.parametrize(
    ("preset", "head_dim"),
    [(p, 128) for p in PRESETS] + [("turboquant_4bit_nc", 256)],
)
def test_tq_ragged_paged_attention_and_skip_update(preset, head_dim):
    """Prefill + decode attention approximates exact RPA and respects skip_kv_update."""
    page_size, num_pages = 16, 12
    num_q_heads, num_kv_heads = 4, 2
    layout, tables = _layout_and_tables(preset, head_dim)
    rng = np.random.default_rng(7)
    sm_scale = head_dim**-0.5

    # Prefill 2 seqs of lengths [20, 12], then step [1, 4] tokens -> kv_lens [21, 16].
    hist_lens = jnp.array([20, 12], jnp.int32)
    hist_cu = jnp.array([0, 20, 32], jnp.int32)
    kv_lens = jnp.array([21, 16], jnp.int32)
    cu_q_lens = jnp.array([0, 1, 5], jnp.int32)
    dist_prefill = jnp.array([0, 2, 2], jnp.int32)
    dist_mixed = jnp.array([1, 2, 2], jnp.int32)
    page_indices = jnp.array([2, 5, 8, 1, 4, 9], jnp.int32)

    hk = jnp.asarray(rng.standard_normal((32, num_kv_heads, head_dim)), jnp.float32)
    hv = jnp.asarray(rng.standard_normal((32, num_kv_heads, head_dim)), jnp.float32)
    hq = jnp.zeros((32, num_q_heads, head_dim), jnp.float32)
    q = jnp.asarray(rng.standard_normal((5, num_q_heads, head_dim)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((5, num_kv_heads, head_dim)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((5, num_kv_heads, head_dim)), jnp.float32)

    page_words = layout.page_words_128(page_size)
    tq_cache = jnp.zeros((num_pages, num_kv_heads, page_words, 128), jnp.uint8)
    with jax.default_matmul_precision("float32"):
        tq_cache, _ = tq.tq_ragged_paged_attention(
            hq,
            hk,
            hv,
            tq_cache,
            hist_lens,
            page_indices,
            hist_cu,
            dist_prefill,
            layout=layout,
            tables=tables,
            rpa_fn=ref_ragged_paged_attention,
            sm_scale=sm_scale,
        )
        # Verify skip_kv_update=True leaves cache bitwise identical.
        unchanged_cache, _ = tq.tq_ragged_paged_attention(
            q,
            k,
            v,
            tq_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            dist_mixed,
            layout=layout,
            tables=tables,
            rpa_fn=ref_ragged_paged_attention,
            sm_scale=sm_scale,
            skip_kv_update=True,
        )
        np.testing.assert_array_equal(np.asarray(unchanged_cache), np.asarray(tq_cache))

        tq_cache, tq_out = tq.tq_ragged_paged_attention(
            q,
            k,
            v,
            tq_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            dist_mixed,
            layout=layout,
            tables=tables,
            rpa_fn=ref_ragged_paged_attention,
            sm_scale=sm_scale,
        )

        # Unquantized f32 reference over the same history + current step.
        ref_pool = jnp.zeros(
            (num_pages, page_size, num_kv_heads * 2, 1, head_dim), jnp.float32
        )
        _, ref_pool = ref_ragged_paged_attention(
            hq, hk, hv, ref_pool, hist_lens, page_indices, hist_cu, dist_prefill
        )
        ref_out, _ = ref_ragged_paged_attention(
            q,
            k,
            v,
            ref_pool,
            kv_lens,
            page_indices,
            cu_q_lens,
            dist_mixed,
            sm_scale=sm_scale,
        )

    rel_err = np.linalg.norm(np.asarray(tq_out - ref_out)) / np.linalg.norm(
        np.asarray(ref_out)
    )
    assert rel_err < {8: 0.12, 4: 0.20, 3: 0.32}[layout.key_bits], rel_err

    # Unallocated pages in the pool must remain all-zero.
    untouched = [p for p in range(num_pages) if p not in {2, 5, 1, 4}]
    assert not np.any(np.asarray(tq_cache)[untouched])
