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
"""Qwen4Exp StreamIndex top-k: BisectTopK against the lax.top_k reference."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.tpu_info import SparseCoreInfo, TpuInfo

from vllm_torchtpu.models.vllm.qwen4_exp.kernels.streamindex_topk import (
    BisectTopK,
    SparseCoreTopK,
    TensorCoreTopK,
    default_topk_backend,
    streamindex_topk,
    topk_backend,
)

_VARIANTS = {
    "default": BisectTopK(),
    # Bucketed whatever the row count, or flat.
    "radix2_buckets": BisectTopK(bucket_min_rows=1),
    "radix1_buckets": BisectTopK(radix_bits=1, bucket_min_rows=1),
    "radix4_buckets": BisectTopK(radix_bits=4, bucket_min_rows=1),
    "radix2_flat": BisectTopK(width_buckets=False),
}


def _select(backend, scores, k, lengths, return_scores):
    """Runs ``backend`` on f32 ``scores`` as StreamIndex hands them over."""
    fn = jax.jit(
        lambda bits, lens: backend.select(
            bits, k, lens, scheduling_group_id=None, return_scores=return_scores
        )
    )
    out = fn(jnp.asarray(scores.view(np.int32)), jnp.asarray(lengths, jnp.int32))
    return jax.tree.map(np.asarray, out)


def _expected_ids(scores, k, lengths):
    """lax.top_k semantics in NumPy: highest scores, lowest column on ties.

    Signed zeros compare equal, -inf never wins, columns at or past the row
    length are ineligible. Returns one sorted ID array per row.
    """
    expected = []
    for row, length in zip(scores, lengths):
        cols = np.arange(min(int(length), row.shape[0]))
        cols = cols[row[cols] > -np.inf]
        # Stable sort on -score keeps the lower column first among ties.
        order = np.argsort(-row[cols].astype(np.float64), kind="stable")
        expected.append(np.sort(cols[order[:k]]))
    return expected


def _assert_exact(ids, scores, k, lengths, got_scores=None):
    assert ids.shape == (scores.shape[0], k)
    for r, want in enumerate(_expected_ids(scores, k, lengths)):
        row = ids[r]
        n = len(want)
        np.testing.assert_array_equal(np.sort(row[:n]), want, f"row {r}")
        assert np.all(row[n:] == -1), f"row {r}: -1 padding is not a suffix"
        if got_scores is not None:
            got = got_scores[r].view(np.float32)
            np.testing.assert_array_equal(got[:n], scores[r, row[:n]])
            assert np.all(np.isneginf(got[n:]))


def _cases():
    rng = np.random.default_rng(7)
    random = rng.standard_normal((8, 4096)).astype(np.float32)
    # Few distinct values: the threshold value ties far beyond k.
    ties = rng.integers(-3, 4, (8, 4096)).astype(np.float32)
    # -inf inside the row, signed zeros and extreme values.
    padded = rng.standard_normal((8, 4096)).astype(np.float32)
    padded[:, ::3] = -np.inf
    padded[:, 1::7] = -0.0
    padded[:, 2::11] = 0.0
    padded[0, :5] = [np.inf, -np.inf, 3e38, -3e38, 1e-45]
    padded[1] = -np.inf  # No valid score at all.
    all_tied = np.full((4, 4096), 1.5, np.float32)
    # A ReLU-scored indexer: fewer than k positive scores, the rest exact
    # (+/-) zeros, so the threshold is 0.0 with thousands of ties.
    relu = np.maximum(rng.standard_normal((6, 8192)).astype(np.float32) - 3.0, 0.0)
    relu[1, ::2] = -0.0
    relu[2] = 0.0
    # Winners fill whole 256-column pack blocks, then straddle a boundary.
    dense = rng.standard_normal((4, 65536)).astype(np.float32)
    dense[0, 512:1024] += 100
    dense[1, 1000:1600] += 100
    dense[2, 65536 - 512 :] += 100
    dense[3, ::128] += 100
    return [
        ("random", random, 128, [4096] * 8),
        ("ties", ties, 128, [4096, 4095, 1000, 129, 128, 127, 1, 0]),
        ("neg_inf_padding", padded, 128, [4096, 4096, 300, 200, 100, 64, 7, 0]),
        ("all_tied", all_tied, 128, [4096, 128, 50, 0]),
        ("k_exceeds_width", ties[:, :96], 128, [96, 95, 60, 0, 96, 1, 2, 3]),
        ("relu_zero_ties", relu, 512, [8192, 8192, 8192, 600, 513, 512]),
        ("dense_blocks", dense, 512, [65536, 65536, 65536, 65000]),
    ]


_CASES = _cases()


@pytest.mark.parametrize("return_scores", [False, True])
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize(
    ("scores", "k", "lengths"),
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
def test_bisect_matches_lax_top_k(scores, k, lengths, variant, return_scores):
    out = _select(_VARIANTS[variant], scores, k, lengths, return_scores)
    ids, got_scores = out if return_scores else (out, None)
    _assert_exact(ids, scores, k, lengths, got_scores)
    # The reference selector agrees with the NumPy model too.
    ref = _select(TensorCoreTopK(), scores, k, lengths, False)
    _assert_exact(ref, scores, k, lengths)


@pytest.mark.parametrize(
    "longest", [1, 511, 512, 513, 2048, 2049, 4097, 9000, 16385, 20000, 40960]
)
@pytest.mark.parametrize("variant", ["radix2_buckets", "radix2_flat"])
def test_every_width_bucket(longest, variant):
    # The switch picks one bucket per call from the longest row, so each
    # longest length lands in a different bucket (the first ones all-win).
    rng = np.random.default_rng(longest)
    scores = rng.standard_normal((6, 40960)).astype(np.float32)
    scores[:, 7::13] = 0.0  # Some exact ties.
    lengths = [longest, longest // 2, longest - 1, 0, 1, min(longest, 512)]
    ids, got = _select(_VARIANTS[variant], scores, 512, lengths, True)
    _assert_exact(ids, scores, 512, lengths, got)


@pytest.mark.parametrize(
    "lengths", [[512] * 4, [0, 1, 300, 512]], ids=["exactly_k", "mixed_short"]
)
def test_every_short_row_wins_whole(lengths):
    # No row longer than k: the all-win bucket, no threshold search.
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((4, 8192)).astype(np.float32)
    ids = _select(_VARIANTS["radix2_buckets"], scores, 512, lengths, False)
    for row, length in zip(ids, lengths):
        np.testing.assert_array_equal(row[:length], np.arange(length))
        assert np.all(row[length:] == -1)


def test_ties_take_lowest_columns_in_column_order():
    scores = np.array([[1, 5, 1, 1, 5, 1, 0, 1]], np.float32)
    for name, backend in _VARIANTS.items():
        ids = _select(backend, scores, 4, [8], False)
        np.testing.assert_array_equal(ids, [[0, 1, 2, 4]], name)


def test_all_equal_rows_at_serving_shape():
    # Worst-case ties at a serving-sized indexer shape: every pack block is
    # full of ties, so in-block ranks reach 256 (bf16's last exact integer).
    rows, width, k = 512, 65536, 512
    lengths = np.full(rows, width, np.int32)
    lengths[1::4] = np.arange(rows // 4) * 257 % width + 1
    scores = np.full((rows, width), 0.25, np.float32)
    ids, got = _select(BisectTopK(), scores, k, lengths, True)
    counts = np.minimum(lengths, k)
    expected = np.where(np.arange(k)[None, :] < counts[:, None], np.arange(k), -1)
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_array_equal(
        got.view(np.float32), np.where(expected >= 0, np.float32(0.25), -np.inf)
    )


@pytest.mark.parametrize(
    ("width", "k", "expected"),
    [
        (40960, 512, (2048, 4096, 8192, 16384, 40960)),
        (65536, 512, (2048, 4096, 8192, 16384, 32768, 65536)),
        (2048, 512, (2048,)),
        (3000, 512, (3000,)),
        (8192, 128, (512, 1024, 2048, 4096, 8192)),
    ],
)
def test_bucket_widths(width, k, expected):
    assert topk_backend._bucket_widths(width, k) == expected


def test_backends_are_hashable_static_arguments():
    assert BisectTopK() == BisectTopK(
        radix_bits=2, width_buckets=True, bucket_min_rows=128
    )
    assert hash(BisectTopK()) != hash(BisectTopK(radix_bits=4))
    for radix_bits in (3, 8):
        with pytest.raises(ValueError, match="radix_bits"):
            BisectTopK(radix_bits=radix_bits)


@pytest.mark.parametrize(
    ("usable", "expected"), [(True, SparseCoreTopK), (False, BisectTopK)]
)
def test_default_backend(monkeypatch, usable, expected):
    monkeypatch.setattr(topk_backend, "_has_usable_sparse_core", lambda: usable)
    assert isinstance(default_topk_backend(), expected)


def test_jax_sparse_core_info_has_the_fields_the_gate_reads():
    # The gate reads JAX's own SparseCoreInfo, as `get_tpu_info().sparse_core`
    # returns it on the pinned JAX version.
    assert {f.name for f in dataclasses.fields(TpuInfo)} >= {"sparse_core"}
    assert {f.name for f in dataclasses.fields(SparseCoreInfo)} >= {
        "num_cores",
        "num_subcores",
        "num_lanes",
        "dma_granule_size_bytes",
    }


def _sc(num_cores=2, num_subcores=16, num_lanes=16, granule=32):
    return SparseCoreInfo(
        num_cores=num_cores,
        num_subcores=num_subcores,
        num_lanes=num_lanes,
        vmem_capacity_bytes=1 << 20,
        dma_granule_size_bytes=granule,
    )


@pytest.mark.parametrize(
    ("sc_info", "usable"),
    [
        (None, False),
        (_sc(), True),
        (_sc(granule=16), True),
        (_sc(granule=64), False),
        (_sc(num_cores=1, num_subcores=4, granule=64), False),
        (_sc(num_cores=4), False),
        (_sc(num_subcores=8), False),
        (_sc(num_lanes=8), False),
    ],
    ids=[
        "no_sparse_core",
        "2x16_granule_32",
        "2x16_granule_16",
        "granule_64",
        "1x4_granule_64",
        "4_cores",
        "8_subcores",
        "8_lanes",
    ],
)
def test_usable_sparse_core_matches_the_kernel_layout(monkeypatch, sc_info, usable):
    monkeypatch.setattr(topk_backend, "get_sparse_core_info", lambda: sc_info)
    assert topk_backend._has_usable_sparse_core() is usable


def _fp8_key_cache(keys):
    """FP8 + UE8M0-scale key records, ``[pages, page_size // 4, 4, 256]`` bytes."""
    num_pages, page_size, head_dim = keys.shape
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.clip(jnp.max(jnp.abs(keys), axis=-1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    fp8 = (keys / scale).astype(jnp.float8_e4m3fn)
    record = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(fp8, jnp.uint8),
            jax.lax.bitcast_convert_type(scale.astype(jnp.float8_e8m0fnu), jnp.uint8),
        ],
        axis=-1,
    )
    record = jnp.pad(record, ((0, 0), (0, 0), (0, 256 - record.shape[-1])))
    return record.reshape(num_pages, page_size // 4, 4, 256)


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_tokens", [None, 64])
@pytest.mark.parametrize(
    ("q_lens", "seq_lens"),
    [
        ((1, 1, 1, 1, 1, 1, 1, 1), (4096, 300, 2049, 64, 4000, 1, 3000, 2500)),
        ((1, 1, 126), (1500, 4096, 3000)),
        ((128,), (4096,)),
    ],
    ids=["decode", "mixed", "prefill"],
)
def test_streamindex_bisect_matches_xla_selector(q_lens, seq_lens, chunk_tokens):
    # Real StreamIndex scores (random FP8 keys, compression 4) through both
    # TensorCore selectors: the same top-k sets, row by row.
    rng = np.random.default_rng(sum(seq_lens))
    page_size, pages_per_seq, heads, dim, k = 128, 32, 4, 128, 128
    num_seqs, num_tokens = len(q_lens), sum(q_lens)
    num_pages = num_seqs * pages_per_seq
    table = rng.permutation(num_pages).astype(np.int32)
    keys = jnp.asarray(rng.standard_normal((num_pages, page_size, dim), np.float32))
    num_decodes = 0  # `distribution` wants the one-token sequences first.
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
        num_decodes += 1
    args = (
        jnp.asarray(rng.standard_normal((num_tokens, heads, dim), np.float32)),
        jnp.asarray(rng.uniform(0.25, 1.75, (num_tokens, heads)).astype(np.float32)),
        _fp8_key_cache(keys),
        jnp.asarray(seq_lens, jnp.int32),
        jnp.asarray(table),
        jnp.asarray(np.r_[0, np.cumsum(q_lens)], jnp.int32),
        jnp.asarray([num_decodes, num_decodes, num_seqs], jnp.int32),
    )
    kwargs = dict(
        k=k,
        compression_ratio=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=(1, 64, 64),
        chunk_tokens=chunk_tokens if num_tokens % 64 == 0 else None,
    )
    want = np.asarray(streamindex_topk(*args, topk_backend=TensorCoreTopK(), **kwargs))
    got = np.asarray(streamindex_topk(*args, topk_backend=BisectTopK(), **kwargs))
    assert got.shape == want.shape == (num_tokens, k)
    for row, (g, w) in enumerate(zip(got, want)):
        n = int((w >= 0).sum())
        np.testing.assert_array_equal(np.sort(g[:n]), np.sort(w[:n]), f"row {row}")
        assert np.all(g[n:] == -1), f"row {row}: -1 padding is not a suffix"


@pytest.mark.tpu_test
@pytest.mark.parametrize("return_scores", [False, True])
@pytest.mark.parametrize("chunk_tokens", [None, 64])
def test_sparsecore_and_lax_top_k_select_the_same_blocks(chunk_tokens, return_scores):
    # The SparseCore selector against the lax.top_k reference through the same
    # scoring passes, on chips where the SparseCore selector can run.
    if not topk_backend._has_usable_sparse_core():
        pytest.skip("needs a usable SparseCore")
    rng = np.random.default_rng(7)
    page_size, pages_per_seq, heads, dim, k = 128, 32, 4, 128, 512
    q_lens, seq_lens = (1, 1, 126), (1500, 4096, 3000)
    num_seqs, num_tokens = len(q_lens), sum(q_lens)
    num_pages = num_seqs * pages_per_seq
    keys = jnp.asarray(rng.standard_normal((num_pages, page_size, dim), np.float32))
    args = (
        jnp.asarray(rng.standard_normal((num_tokens, heads, dim), np.float32)),
        jnp.asarray(rng.uniform(0.25, 1.75, (num_tokens, heads)).astype(np.float32)),
        _fp8_key_cache(keys),
        jnp.asarray(seq_lens, jnp.int32),
        jnp.asarray(rng.permutation(num_pages).astype(np.int32)),
        jnp.asarray(np.r_[0, np.cumsum(q_lens)], jnp.int32),
        jnp.asarray([2, 2, num_seqs], jnp.int32),
    )
    kwargs = dict(
        k=k,
        compression_ratio=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=(1, 64, 64),
        chunk_tokens=chunk_tokens,
        return_scores=return_scores,
    )
    got = streamindex_topk(*args, topk_backend=SparseCoreTopK(), **kwargs)
    want = streamindex_topk(*args, topk_backend=TensorCoreTopK(), **kwargs)
    got_ids, want_ids = (np.asarray(jax.tree.leaves(x)[0]) for x in (got, want))
    for row, (g, w) in enumerate(zip(got_ids, want_ids, strict=True)):
        n = int((w >= 0).sum())
        np.testing.assert_array_equal(np.sort(g[:n]), np.sort(w[:n]), f"row {row}")
        assert np.all(g[n:] == -1), f"row {row}: -1 padding is not a suffix"
