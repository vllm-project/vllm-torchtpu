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
"""Tests for the `skip_kv_update` path of ragged paged attention.

`skip_kv_update=True` implements cross-layer KV cache sharing (e.g. gemma4
E2B/E4B): a shared layer must attend against the target layer's K/V that is
already present in the (aliased) cache, WITHOUT writing its own K/V into it.

The invariant under test: running with `skip_kv_update=True` and arbitrary
("garbage") keys/values must be identical to running with `skip_kv_update=False`
where the keys/values passed are exactly the ones already cached at the current
positions (i.e. a no-op rewrite) -- and must leave the cache byte-for-byte
unchanged. Covered for the v3 default kernel (head_dim != 64), the hd64 kernel
(head_dim == 64), and the experimental batched kernel (--attention-backend
CUSTOM). The batched kernel is additionally checked for numerical parity against
the v3 kernel on the same logical K/V.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel_hd64 as K64

_BATCHED_DECODE_BLOCKS = KBC.BlockSizes(
    bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8, n_buffer=3
)
_BATCHED_PREFILL_BLOCKS = KBC.BlockSizes(
    bq_sz=1792, bq_c_sz=28, bkv_sz=256, batch_size=2, n_buffer=3
)

# One entry per kernel family. Each kernel family ships its own cache layout
# (`get_kv_cache_shape`/`merge_kv`) and reference + Pallas implementations, but
# all share the same public call signature, so the test below is layout-aware
# but otherwise identical across families.
V3 = dict(
    name="v3",
    head_dim=128,
    get_shape=K.get_kv_cache_shape,
    merge_kv=K.merge_kv,
    pallas=K.ragged_paged_attention,
    ref=K.ref_ragged_paged_attention,
)
HD64 = dict(
    name="hd64",
    head_dim=64,
    get_shape=K64.get_kv_cache_shape,
    merge_kv=K64.merge_kv,
    pallas=K64.ragged_paged_attention_hd64,
    ref=K64.ref_ragged_paged_attention_hd64,
)


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}.")


def _build_packed_cache(
    cfg, target_k, target_v, kv_len, total_pages, page_size, num_kv_heads, dtype
):
    """Pack `target_k/v` for positions [0, kv_len) into a paged kv cache.

    Page i holds positions [i*page_size, (i+1)*page_size), so the single
    sequence uses pages 0, 1, ... in order.
    """
    cache_shape = cfg["get_shape"](
        total_pages, page_size, num_kv_heads, cfg["head_dim"], dtype
    )
    merged = np.asarray(
        cfg["merge_kv"](
            jnp.asarray(target_k, dtype), jnp.asarray(target_v, dtype)
        ).astype(jnp.float32)
    )
    cache = np.zeros(cache_shape, np.float32)
    flat = cache.reshape(total_pages * page_size, *cache_shape[2:])
    flat[:kv_len] = merged
    return flat.reshape(cache_shape)


@pytest.mark.parametrize("cfg", [V3, HD64], ids=lambda c: c["name"])
@pytest.mark.parametrize("impl", ["pallas", "ref"])
@pytest.mark.parametrize(
    "kv_len, q_len, distribution",
    [
        (48, 16, (0, 0, 1)),  # mixed: history + a chunk of current tokens
        (33, 1, (1, 1, 1)),  # decode: a single current token
    ],
)
def test_skip_kv_update_reuses_cache_without_writing(
    cfg, impl, kv_len, q_len, distribution
):
    _require_tpu()
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    head_dim = cfg["head_dim"]
    num_kv_heads, num_q_heads = 2, 4
    page_size, total_pages = 16, 8
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    fn = cfg[impl]

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    # The K/V the target layer placed into the shared cache for every position.
    target_k, target_v = (
        r(kv_len, num_kv_heads, head_dim),
        r(kv_len, num_kv_heads, head_dim),
    )
    query = r(q_len, num_q_heads, head_dim)
    # The shared layer's own (un-normed / un-RoPE'd) K/V -- must be ignored.
    garbage_k, garbage_v = (
        r(q_len, num_kv_heads, head_dim),
        r(q_len, num_kv_heads, head_dim),
    )
    # What the target wrote at the current positions == what is already cached.
    cached_k, cached_v = (
        target_k[kv_len - q_len : kv_len],
        target_v[kv_len - q_len : kv_len],
    )

    cache_np = _build_packed_cache(
        cfg, target_k, target_v, kv_len, total_pages, page_size, num_kv_heads, dtype
    )

    kv_lens = jnp.array([kv_len], jnp.int32)
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    distribution = jnp.asarray(distribution, jnp.int32)

    def run(keys, values, skip):
        # Inputs are donated by the kernel, so materialize fresh copies.
        out, cache = fn(
            jnp.asarray(query, dtype),
            jnp.asarray(keys, dtype),
            jnp.asarray(values, dtype),
            jnp.asarray(cache_np, dtype),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            skip_kv_update=skip,
        )
        return (
            np.asarray(out.astype(jnp.float32)),
            np.asarray(cache.astype(jnp.float32)),
        )

    # Oracle: write the already-cached values back (a no-op) and attend.
    oracle_out, _ = run(cached_k, cached_v, skip=False)
    # Skip: ignore garbage K/V, read the cache, write nothing.
    skip_out, skip_cache = run(garbage_k, garbage_v, skip=True)
    # Negative control: without skipping, garbage corrupts output and cache.
    bad_out, bad_cache = run(garbage_k, garbage_v, skip=False)

    tol = 1e-2  # bf16
    np.testing.assert_allclose(skip_out, oracle_out, atol=tol, rtol=0)
    # The shared cache must be untouched by the skip path.
    np.testing.assert_array_equal(skip_cache, cache_np)
    # Sanity: the garbage really would have mattered, so the test is meaningful.
    assert np.abs(bad_out - oracle_out).max() > tol
    assert np.abs(bad_cache - cache_np).max() > tol


def test_batched_wrapper_accepts_skip_kv_update():
    """Cheap, hardware-free signature guard (mirrors upstream's
    `test_batched_rpa_wrapper_accepts_update_kv_cache`).

    `sharded_ragged_paged_attention` forwards `skip_kv_update` unconditionally
    on the non-hd64 path, so the batched wrapper must declare it as a kwarg with
    a no-op default of False -- otherwise the forwarding crashes at trace time
    with `TypeError: ... got an unexpected keyword argument 'skip_kv_update'`.
    """
    import inspect

    params = inspect.signature(KB.ragged_paged_attention).parameters
    assert "skip_kv_update" in params, list(params)
    assert params["skip_kv_update"].default is False


def _build_batched_cache(
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
    """Populate a batched-layout kv cache with target K/V for [0, kv_len).

    Rather than hand-replicate the batched kernel's packed cache layout
    (k/v interleaved per head, folded into the dtype-packing axis), build the
    cache by running the kernel's own write path: a full prefill
    (`skip_kv_update=False`, q_len == kv_len) writes every position's K/V into
    an initially-zero cache. The attention output is discarded.
    """
    cache_shape = KB.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype
    )
    _, cache = KB.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), dtype),
        jnp.asarray(target_k, dtype),
        jnp.asarray(target_v, dtype),
        jnp.zeros(cache_shape, dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),  # one MIXED (prefill) sequence
        sm_scale=sm_scale,
        decode_block_sizes=_BATCHED_DECODE_BLOCKS,
        prefill_block_sizes=_BATCHED_PREFILL_BLOCKS,
        skip_kv_update=False,
    )
    return np.asarray(cache.astype(jnp.float32))


@pytest.mark.parametrize(
    "kv_len, q_len, distribution",
    [
        # Smaller than the v3/hd64 cases above: the batched kernel's schedule
        # metadata lives in SMEM and a larger full-prefill cache build OOMs it.
        (32, 16, (0, 0, 1)),  # mixed: history + a chunk of current tokens
        (33, 1, (1, 1, 1)),  # decode: a single current token
    ],
)
def test_skip_kv_update_batched(kv_len, q_len, distribution):
    """The experimental batched kernel's `skip_kv_update` path must (a) match a
    no-op rewrite of the already-cached values, (b) leave the cache untouched,
    and (c) numerically agree with the v3 kernel on the same logical K/V.

    The batched kernel has no reference implementation, so correctness is pinned
    two ways: a self-consistency oracle (as for v3/hd64 above) and cross-kernel
    parity against the trusted v3 `skip_kv_update` path.
    """
    _require_tpu()
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    head_dim = 128  # batched kernel has no hd64 variant; gemma4 uses 256/512
    num_kv_heads, num_q_heads = 2, 4
    page_size, total_pages = 16, 8
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    # The K/V the target layer placed into the shared cache for every position.
    target_k, target_v = (
        r(kv_len, num_kv_heads, head_dim),
        r(kv_len, num_kv_heads, head_dim),
    )
    query = r(q_len, num_q_heads, head_dim)
    # The shared layer's own (un-normed / un-RoPE'd) K/V -- must be ignored.
    garbage_k, garbage_v = (
        r(q_len, num_kv_heads, head_dim),
        r(q_len, num_kv_heads, head_dim),
    )
    # What the target wrote at the current positions == what is already cached.
    cached_k, cached_v = (
        target_k[kv_len - q_len : kv_len],
        target_v[kv_len - q_len : kv_len],
    )

    cache_np = _build_batched_cache(
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

    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    distribution = jnp.asarray(distribution, jnp.int32)

    def run(keys, values, skip):
        # Inputs are donated by the kernel, so materialize fresh copies.
        out, cache = KB.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(keys, dtype),
            jnp.asarray(values, dtype),
            jnp.asarray(cache_np, dtype),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            decode_block_sizes=_BATCHED_DECODE_BLOCKS,
            prefill_block_sizes=_BATCHED_PREFILL_BLOCKS,
            skip_kv_update=skip,
        )
        return (
            np.asarray(out.astype(jnp.float32)),
            np.asarray(cache.astype(jnp.float32)),
        )

    # Oracle: write the already-cached values back (a no-op) and attend.
    oracle_out, _ = run(cached_k, cached_v, skip=False)
    # Skip: ignore garbage K/V, read the cache, write nothing.
    skip_out, skip_cache = run(garbage_k, garbage_v, skip=True)
    # Negative control: without skipping, garbage corrupts output and cache.
    bad_out, bad_cache = run(garbage_k, garbage_v, skip=False)

    tol = 1e-2  # bf16
    np.testing.assert_allclose(skip_out, oracle_out, atol=tol, rtol=0)
    # The shared cache must be untouched by the skip path.
    np.testing.assert_array_equal(skip_cache, cache_np)
    # Sanity: the garbage really would have mattered, so the test is meaningful.
    assert np.abs(bad_out - oracle_out).max() > tol
    assert np.abs(bad_cache - cache_np).max() > tol

    # Cross-kernel parity: the trusted v3 skip path over the same logical K/V
    # must produce the same attention output (different reduction order -> a
    # looser bf16 tolerance than the self-consistency check above).
    v3_cache = _build_packed_cache(
        V3, target_k, target_v, kv_len, total_pages, page_size, num_kv_heads, dtype
    )
    v3_out, _ = K.ragged_paged_attention(
        jnp.asarray(query, dtype),
        jnp.asarray(garbage_k, dtype),
        jnp.asarray(garbage_v, dtype),
        jnp.asarray(v3_cache, dtype),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        skip_kv_update=True,
    )
    np.testing.assert_allclose(
        skip_out, np.asarray(v3_out.astype(jnp.float32)), atol=2e-2, rtol=0
    )


def test_batched_seq_along_lane_matches_head_along_sublane():
    """`kv_layout=SEQ_ALONG_LANE` must be numerically equivalent to the default
    `HEAD_ALONG_SUBLANE` layout for identical logical Q/K/V: layout only
    changes how the cache is packed in memory, not the attention math.
    """
    _require_tpu()
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    head_dim = 128
    num_kv_heads, num_q_heads = 2, 4
    # SEQ_ALONG_LANE requires page_size == 128 (RpaConfigs.validate_inputs).
    page_size, total_pages = 128, 1
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    kv_len = q_len = 32  # single MIXED sequence, full prefill from a zero cache
    distribution = jnp.asarray((0, 0, 1), jnp.int32)
    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    key = r(kv_len, num_kv_heads, head_dim)
    value = r(kv_len, num_kv_heads, head_dim)
    query = r(q_len, num_q_heads, head_dim)

    def run(kv_layout):
        cache_shape = KB.get_kv_cache_shape(
            total_pages, page_size, num_kv_heads, head_dim, dtype, kv_layout=kv_layout
        )
        out, _ = KB.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(key, dtype),
            jnp.asarray(value, dtype),
            jnp.zeros(cache_shape, dtype),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            decode_block_sizes=_BATCHED_DECODE_BLOCKS,
            prefill_block_sizes=_BATCHED_PREFILL_BLOCKS,
            kv_layout=kv_layout,
        )
        return np.asarray(out.astype(jnp.float32))

    head_along_sublane_out = run(KBC.KVLayout.HEAD_ALONG_SUBLANE)
    seq_along_lane_out = run(KBC.KVLayout.SEQ_ALONG_LANE)

    tol = 2e-2  # bf16, cross-layout reduction order differs
    np.testing.assert_allclose(
        seq_along_lane_out, head_along_sublane_out, atol=tol, rtol=0
    )
