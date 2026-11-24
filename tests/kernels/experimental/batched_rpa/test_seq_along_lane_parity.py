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
"""Numeric parity for the SEQ_ALONG_LANE KV-cache layout.

Selected with `VLLM_KV_CACHE_LAYOUT=HND`, which drives both this fork and the
mainline batched-RPA kernel.
"""

import dataclasses
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.tpu_test

_DECODE_BLOCKS = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8)
_PREFILL_BLOCKS = configs.BlockSizes(bq_sz=1792, bq_c_sz=28, bkv_sz=256, batch_size=2)


def _build_cache(
    kv_layout,
    target_k,
    target_v,
    kv_len,
    total_pages,
    page_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_indices,
    sm_scale,
):
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype, kv_layout=kv_layout
    )
    _, cache = wrapper.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), dtype),
        jnp.asarray(target_k, dtype),
        jnp.asarray(target_v, dtype),
        jnp.zeros(cache_shape, dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        kv_layout=kv_layout,
    )
    return cache


@pytest.mark.parametrize(
    "kv_len, q_len, distribution",
    [
        (32, 16, (0, 0, 1)),
        (33, 1, (1, 1, 1)),
        (4096, 1, (1, 1, 1)),
    ],
)
def test_seq_along_lane_matches_head_along_sublane(kv_len, q_len, distribution):
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    head_dim = 128
    num_kv_heads, num_q_heads = 2, 4
    page_size, total_pages = 128, max(8, -(-kv_len // 16) + 1)
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k, target_v = (
        r(kv_len, num_kv_heads, head_dim),
        r(kv_len, num_kv_heads, head_dim),
    )
    query = r(q_len, num_q_heads, head_dim)
    new_k, new_v = target_k[kv_len - q_len : kv_len], target_v[kv_len - q_len : kv_len]

    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    distribution = jnp.asarray(distribution, jnp.int32)

    def run(kv_layout):
        cache = _build_cache(
            kv_layout,
            target_k,
            target_v,
            kv_len,
            total_pages,
            page_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype,
            page_indices,
            sm_scale,
        )
        out, _ = wrapper.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(new_k, dtype),
            jnp.asarray(new_v, dtype),
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            decode_block_sizes=_DECODE_BLOCKS,
            prefill_block_sizes=_PREFILL_BLOCKS,
            kv_layout=kv_layout,
        )
        return np.asarray(out.astype(jnp.float32))

    out_sublane = run(configs.KVLayout.HEAD_ALONG_SUBLANE)
    out_lane = run(configs.KVLayout.SEQ_ALONG_LANE)

    np.testing.assert_allclose(out_lane, out_sublane, atol=2e-2, rtol=0)
    assert np.isfinite(out_lane).all()


@dataclasses.dataclass(frozen=True)
class _Batch:
    """Three requests (q lens 1, 3, 2) with fresh KV, and how to run them."""

    query: jnp.ndarray
    key: jnp.ndarray
    value: jnp.ndarray
    q_lens: tuple[int, ...]
    starts: np.ndarray
    head_dim: int
    page_size: int
    scales: tuple[jnp.ndarray | None, jnp.ndarray | None]
    scale_flags: dict[str, object]
    run: Callable[..., tuple[jnp.ndarray, ...]]


def _quantize_per_token(x, scale_dtype):
    """fp8 with one scale per token and head: (x_fp8 [T, H, D], scale [T, H])."""
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / float(
        jnp.finfo(jnp.float8_e4m3fn).max
    )
    scale = jnp.where(scale == 0, 1.0, scale)
    return (x / scale).astype(jnp.float8_e4m3fn), scale[..., 0].astype(scale_dtype)


def _gqa_batch(
    kv_layout, q_heads_per_kv, *, seed, head_dim=128, per_token_scale=False
) -> _Batch:
    """Multiple KV heads and unequal request lengths, through decode and prefill."""
    rng = np.random.default_rng(seed)
    q_lens = (1, 3, 2)
    starts = np.cumsum((0, *q_lens)).astype(np.int32)
    num_kv_heads, page_size = 2, 128
    num_q_heads = num_kv_heads * q_heads_per_kv
    num_tokens = sum(q_lens)

    def random_array(shape, dtype):
        return jnp.asarray(rng.normal(0, 0.5, shape), dtype)

    query = random_array((num_tokens, num_q_heads, head_dim), jnp.bfloat16)
    kv_shape = (num_tokens, num_kv_heads, head_dim)
    scales, scale_flags, scale_dtype = (None, None), {}, None
    if per_token_scale:
        scale_dtype = jnp.bfloat16
        key, k_scale = _quantize_per_token(
            random_array(kv_shape, jnp.float32), scale_dtype
        )
        value, v_scale = _quantize_per_token(
            random_array(kv_shape, jnp.float32), scale_dtype
        )
        scales = (k_scale, v_scale)
        scale_flags = {
            "use_per_token_scale": True,
            "per_token_scale_dtype": scale_dtype,
        }
    else:
        key = random_array(kv_shape, jnp.float8_e4m3fn)
        value = random_array(kv_shape, jnp.float8_e4m3fn)
    # Reserve two pages per request for the default 256-token KV block.
    cache_shape = wrapper.get_kv_cache_shape(
        6,
        page_size,
        num_kv_heads,
        head_dim,
        key.dtype,
        kv_layout=kv_layout,
        use_per_token_scale=per_token_scale,
        scale_dtype=scale_dtype,
    )

    def run(queries, keys, values, **kwargs):
        # LONGCTX routes prefill through the MIXED segment, as in the
        # existing parity cases above; its standalone PREFILL pass is unused.
        return wrapper.ragged_paged_attention(
            queries,
            keys,
            values,
            jnp.zeros(cache_shape, key.dtype),
            jnp.asarray(q_lens, jnp.int32),
            jnp.array([4, 0, 2, 5, 1, 3], jnp.int32),
            jnp.asarray(starts),
            jnp.array([1, 1, 3], jnp.int32),
            sm_scale=head_dim**-0.5,
            kv_layout=kv_layout,
            **kwargs,
        )

    return _Batch(
        query, key, value, q_lens, starts, head_dim, page_size, scales, scale_flags, run
    )


@pytest.mark.parametrize("kv_layout", list(configs.KVLayout))
@pytest.mark.parametrize("q_heads_per_kv", [3, 10])
def test_return_lse_preserves_gqa_heads(kv_layout, q_heads_per_kv):
    """LSE writeback preserves GQA head mapping and logical output shapes.

    Exercise multiple KV heads, small and non-power-of-two Q groups, and
    unequal request lengths through both decode and prefill scheduling.
    """
    batch = _gqa_batch(kv_layout, q_heads_per_kv, seed=42)
    query, key, value = batch.query, batch.key, batch.value
    starts, head_dim = batch.starts, batch.head_dim

    def run(return_lse):
        return batch.run(query, key, value, return_lse=return_lse)

    out_without_lse, cache_without_lse = run(False)
    output, cache, lse = run(True)
    np.testing.assert_allclose(
        np.asarray(output, np.float32),
        np.asarray(out_without_lse, np.float32),
        atol=3e-3,
        rtol=1e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(cache, np.float32), np.asarray(cache_without_lse, np.float32)
    )

    queries = np.asarray(query, np.float32)
    keys = np.repeat(np.asarray(key, np.float32), q_heads_per_kv, axis=1)
    values = np.repeat(np.asarray(value, np.float32), q_heads_per_kv, axis=1)
    expected_out, expected_lse = [], []
    for start, end in zip(starts[:-1], starts[1:]):
        for pos in range(start, end):
            scores = (
                np.einsum("hd,thd->ht", queries[pos], keys[start : pos + 1])
                * head_dim**-0.5
            )
            maximum = scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores - maximum)
            denominator = weights.sum(axis=-1, keepdims=True)
            expected_out.append(
                np.einsum("ht,thd->hd", weights / denominator, values[start : pos + 1])
            )
            expected_lse.append((maximum + np.log(denominator))[:, 0])
    assert output.shape == query.shape
    assert lse.shape == query.shape[:2]
    np.testing.assert_allclose(
        np.asarray(output, np.float32), expected_out, atol=3e-3, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(lse, np.float32), expected_lse, atol=4e-2, rtol=1e-2
    )


_SAL = configs.KVLayout.SEQ_ALONG_LANE
_PARITY_CASES = [
    # kv_layout, q_heads_per_kv, mode, head_dim, return_lse, per_token_scale
    *(
        (layout, q, mode, 128, False, False)
        for layout in configs.KVLayout
        for q in (3, 10)
        for mode in ("prepared", "raw", "both")
    ),
    *((layout, 3, "both", 64, False, False) for layout in configs.KVLayout),
    *((layout, 3, "both", 128, True, False) for layout in configs.KVLayout),
    *((_SAL, 3, mode, 128, False, True) for mode in ("prepared", "raw", "both")),
]


@pytest.mark.parametrize(
    "kv_layout,q_heads_per_kv,mode,head_dim,return_lse,per_token_scale", _PARITY_CASES
)
def test_prepared_inputs_and_raw_output_match_default(  # noqa: PLR0913 - the case's fields
    kv_layout, q_heads_per_kv, mode, head_dim, return_lse, per_token_scale
):
    """prepared_inputs and return_raw_output, alone or together, match the default.

    With prepared_inputs the placeholders are zeros, so the test fails if the
    wrapper packs queries/keys/values instead of using the prepared buffers.
    """
    b = _gqa_batch(
        kv_layout,
        q_heads_per_kv,
        seed=0,
        head_dim=head_dim,
        per_token_scale=per_token_scale,
    )
    k_scale, v_scale = b.scales
    dynamic = {"dynamic_k_scale": k_scale, "dynamic_v_scale": v_scale}
    dynamic = {k: v for k, v in dynamic.items() if v is not None}
    want = b.run(
        b.query, b.key, b.value, return_lse=return_lse, **b.scale_flags, **dynamic
    )

    kwargs: dict[str, object] = {"return_lse": return_lse, **b.scale_flags}
    queries, keys, values = b.query, b.key, b.value
    if mode in ("prepared", "both"):
        kwargs["prepared_inputs"] = wrapper.prepare_inputs(
            b.query,
            b.key,
            b.value,
            b.query.dtype,
            b.key.dtype,
            k_scale,
            v_scale,
            kv_layout=kv_layout,
            page_size=b.page_size,
        )
        queries, keys, values = (jnp.zeros_like(x) for x in (b.query, b.key, b.value))
    else:
        kwargs.update(dynamic)
    if mode in ("raw", "both"):
        kwargs["return_raw_output"] = True
    got = b.run(queries, keys, values, **kwargs)

    out = got[0]
    if mode in ("raw", "both"):
        q_packing = 2  # bf16
        aligned_heads = -(-q_heads_per_kv // q_packing) * q_packing
        aligned_dim = -(-head_dim // 128) * 128
        num_kv_heads = b.key.shape[1]
        assert out.shape == (
            num_kv_heads,
            b.query.shape[0],
            aligned_heads // q_packing,
            q_packing,
            aligned_dim,
        )
        out = wrapper.prepare_outputs(out)[:, :, :q_heads_per_kv, :head_dim]
        out = out.swapaxes(0, 1).reshape(b.query.shape)
    assert len(got) == len(want)
    for g, w in zip((out, *got[1:]), want, strict=True):
        np.testing.assert_array_equal(
            np.asarray(g, np.float32), np.asarray(w, np.float32)
        )


@pytest.mark.parametrize("kv_layout", list(configs.KVLayout))
@pytest.mark.parametrize("head_dim", [80, 128])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_skip_kv_update_ignores_garbage_kv_and_preserves_cache(
    kv_layout, head_dim, dtype
):
    """KV-shared calls ignore NaN K/V inputs and leave the target cache untouched."""
    rng = np.random.default_rng(7)
    kv_len, q_len = 32, 4
    num_kv_heads, num_q_heads = 2, 6
    page_size, total_pages = 128, 8
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)

    target_k = jnp.asarray(
        rng.standard_normal((kv_len, num_kv_heads, head_dim)) * 0.5, dtype
    )
    target_v = jnp.asarray(
        rng.standard_normal((kv_len, num_kv_heads, head_dim)) * 0.5, dtype
    )
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype, kv_layout=kv_layout
    )
    _, cache = wrapper.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), jnp.bfloat16),
        target_k,
        target_v,
        jnp.zeros(cache_shape, dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sm_scale=sm_scale,
        kv_layout=kv_layout,
    )
    cache_before = np.asarray(cache.astype(jnp.float32)).copy()

    query = jnp.asarray(
        rng.standard_normal((q_len, num_q_heads, head_dim)) * 0.5, jnp.bfloat16
    )
    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    dist = jnp.array([0, 0, 1], jnp.int32)

    finite_kv = jnp.ones((q_len, num_kv_heads, head_dim), dtype)
    nan_kv = jnp.full((q_len, num_kv_heads, head_dim), jnp.nan, dtype)
    common = dict(sm_scale=sm_scale, kv_layout=kv_layout, skip_kv_update=True)
    out_finite, cache_finite = wrapper.ragged_paged_attention(
        query,
        finite_kv,
        finite_kv,
        jnp.copy(cache),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        **common,
    )
    out_nan, cache_nan = wrapper.ragged_paged_attention(
        query,
        nan_kv,
        nan_kv,
        jnp.copy(cache),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        **common,
    )
    prepared = wrapper.prepare_inputs(
        query,
        nan_kv,
        nan_kv,
        query.dtype,
        cache.dtype,
        kv_layout=kv_layout,
        page_size=page_size,
        skip_kv_update=True,
    )
    out_prep, cache_prep = wrapper.ragged_paged_attention(
        query,
        nan_kv,
        nan_kv,
        jnp.copy(cache),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        prepared_inputs=prepared,
        **common,
    )
    assert np.isfinite(np.asarray(out_nan, np.float32)).all()
    np.testing.assert_array_equal(
        np.asarray(out_nan, np.float32), np.asarray(out_finite, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(out_prep, np.float32), np.asarray(out_finite, np.float32)
    )
    np.testing.assert_array_equal(np.asarray(cache_finite, np.float32), cache_before)
    np.testing.assert_array_equal(np.asarray(cache_nan, np.float32), cache_before)
    np.testing.assert_array_equal(np.asarray(cache_prep, np.float32), cache_before)
