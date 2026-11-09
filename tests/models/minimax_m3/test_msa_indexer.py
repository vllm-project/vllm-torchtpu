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
"""MSA indexer block scores (interpret mode) against a numpy reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels.msa_indexer import block_scores

PAGE = 128
HEAD_DIM = 128
PAGES_PER_SEQ = 8
MAX_SEQS = 4
NUM_PAGES = 40


@pytest.mark.parametrize("kernel", ["v1", "v2", "v3"])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize(
    "q_lens,ctx_lens,block_q,large_block_q,pages_per_step",
    [
        # small (decode, 3 tokens) and large (130 tokens: 3 large tiles)
        ([1, 130, 3], [300, 200, 0], 64, 48, 4),
        ([1, 1], [1000, 5], 128, 512, 2),  # small class only, page groups
    ],
)
def test_matches_reference(
    q_lens, ctx_lens, block_q, large_block_q, pages_per_step, dtype, kernel
):
    rng = np.random.default_rng(len(q_lens))
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + 5
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    keys = rounded(rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)), dtype)
    cache = pack(keys, dtype)
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )

    scores = block_scores(
        index_q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        block_q=block_q,
        large_block_q=large_block_q,
        pages_per_step=pages_per_step,
        small_pages_per_step=pages_per_step,
        kernel=kernel,
        large_kernel=kernel,
        interpret=True,
    )

    queries = np.asarray(index_q.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        for t in range(cu_q[r], cu_q[r + 1]):
            for b in range(num_pages):
                expected = (keys[tables[r, b]] @ queries[t]).max()
                np.testing.assert_allclose(scores[t, b], expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("kernel", ["v1", "v2", "v3"])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lowers_for_tpu(dtype, kernel):
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, max_seqs, pages_per_seq = 256, 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct(
            (4096, PAGE // PACKING[dtype], PACKING[dtype], HEAD_DIM), dtype
        ),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(block_scores, kernel=kernel, large_kernel=kernel))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("layout", ["folded", "scale_page"])
@pytest.mark.parametrize("kernel", ["v2", "v3"])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,block_q,large_block_q,pages_per_step",
    [([1, 130, 3], [300, 200, 0], 64, 48, 4), ([1, 1], [1000, 5], 128, 512, 2)],
)
def test_fp4_matches_dequantized_reference(
    q_lens, ctx_lens, block_q, large_block_q, pages_per_step, kernel, layout
):
    """4-bit index cache: scores equal rotated queries against the dequantized
    rotated keys. "folded": the scales sit in each page's last word row (the
    served layout); "scale_page": a separate scale page row (NaN elsewhere)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    rng = np.random.default_rng(7 + len(q_lens))
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + 5
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    raw = rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)) * rng.uniform(
        0.1, 5, (1, 1, HEAD_DIM)
    )
    nib, scale = msa_fp4.quantize(msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32)))
    row = 2
    scale_cache = np.full((NUM_PAGES, 8, msa_fp4.SCALE_LANES), np.nan, np.float32)
    scale_cache[:, row] = np.asarray(scale)
    if layout == "folded":
        cache = msa_fp4.pack_rows(nib, scales=scale)
        scale_kwargs = {}
    else:
        cache = msa_fp4.pack_rows(nib)
        scale_kwargs = dict(scale_cache=jnp.asarray(scale_cache), scale_row=row)
    keys = np.asarray(msa_fp4.dequantize(nib, scale))
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    q_rot = msa_fp4.rotate_queries(index_q)

    scores = block_scores(
        q_rot,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        block_q=block_q,
        large_block_q=large_block_q,
        pages_per_step=pages_per_step,
        small_pages_per_step=pages_per_step,
        kernel=kernel,
        large_kernel=kernel,
        interpret=True,
        **scale_kwargs,
    )

    queries = np.asarray(q_rot.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        for t in range(cu_q[r], cu_q[r + 1]):
            for b in range(num_pages):
                expected = (keys[tables[r, b]] @ queries[t]).max()
                # Folded pages scale the keys in bf16 (msa_indexer.SCALE_KEYS):
                # the scale and the scaled key are each rounded to bf16.
                rtol = 1e-2 if layout == "folded" else 1e-4
                np.testing.assert_allclose(scores[t, b], expected, rtol=rtol, atol=1e-3)


@pytest.mark.parametrize("kernel", ["v2", "v3"])
def test_fp4_folded_lowers_for_tpu(kernel):
    tokens, max_seqs, pages_per_seq = 256, 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, PAGE // 8 + 1, 4, HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    fn = functools.partial(block_scores, kernel=kernel, large_kernel=kernel)
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("kernel", ["v2", "v3"])
def test_fp4_lowers_for_tpu(kernel):
    tokens, max_seqs, pages_per_seq = 256, 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, PAGE // 8, 4, HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((4096, 8, 128), jnp.float32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))

    def f(q, cache, seq_lens, tables, cu_q, n, scales):
        return block_scores(
            q,
            cache,
            seq_lens,
            tables,
            cu_q,
            n,
            kernel=kernel,
            large_kernel=kernel,
            scale_cache=scales,
            scale_row=2,
        )

    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("layout", ["bf16", "fp8", "fp4_folded"])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [
        # decode / verify only: no large request (the large class is skipped)
        ([1, 8, 8, 1], [0, 129, 600, 1015]),
        # a large request too: both classes run
        ([8, 40, 1], [250, 300, 777]),
        ([1], [0]),  # one page, every other group of the step dead
    ],
)
def test_skip_dead_matches_reference_arm(q_lens, ctx_lens, layout, monkeypatch):
    """`skip_dead` (v3 skips dot groups past a request's last page; the large
    class runs only with a large request) gives the reference arm's scores
    bit for bit on every page of every request."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_indexer

    # Two pages per dot, so a step of 8 pages has 4 groups to skip.
    block_q = 16
    monkeypatch.setattr(msa_indexer, "_MAX_SCORE_ELEMS", 2 * PAGE * block_q)
    jax.clear_caches()
    rng = np.random.default_rng(sum(ctx_lens) + len(layout))
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = max(num_real + 3, block_q + 8)
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    raw = rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM))
    if layout == "fp4_folded":
        nib, scale = msa_fp4.quantize(
            msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32))
        )
        cache = msa_fp4.pack_rows(nib, scales=scale)
    else:
        dtype = jnp.bfloat16 if layout == "bf16" else jnp.float8_e4m3fn
        cache = pack(rounded(raw, dtype), dtype)
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    args = (
        index_q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
    )
    kw = dict(
        block_q=block_q,
        large_block_q=32,
        pages_per_step=4,
        small_pages_per_step=PAGES_PER_SEQ,
        kernel="v3",
        large_kernel="v2",
        interpret=True,
    )
    got = np.asarray(block_scores(*args, skip_dead=True, **kw))[:, 0]
    ref = np.asarray(block_scores(*args, skip_dead=False, **kw))[:, 0]
    assert got.shape == ref.shape
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        rows = slice(cu_q[r], cu_q[r + 1])
        np.testing.assert_array_equal(
            got[rows, :num_pages].view(np.uint32), ref[rows, :num_pages].view(np.uint32)
        )


def _packed_case(q_lens, ctx_lens, layout, heads, seed):
    """Folded-row batch (`heads` index heads per token) on a 16-page table."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    pages_per_seq, num_pages, max_seqs = 16, 72, 6
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    rows = [heads * q for q in q_lens]
    num_real = sum(rows)
    num_tokens = num_real + 5  # padding rows after the last request
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(rows)])
    tables = np.zeros((max_seqs, pages_per_seq), np.int32)
    tables[:num_seqs] = rng.integers(1, num_pages, (num_seqs, pages_per_seq))
    raw = rng.standard_normal((num_pages, PAGE, HEAD_DIM))
    raw[5] = raw[5, :1]  # every token of a page ties
    raw[7, :, :] = raw[9]  # two pages tie
    if layout == "fp4_folded":
        nib, scale = msa_fp4.quantize(
            msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32))
        )
        scale = scale.at[3, 7].set(jnp.inf).at[4, 9].set(jnp.nan)  # unwritten slots
        cache = msa_fp4.pack_rows(nib, scales=scale)
    else:
        dtype = jnp.bfloat16 if layout == "bf16" else jnp.float8_e4m3fn
        cache = pack(rounded(raw, dtype), dtype)
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    index_q = index_q.at[2].set(index_q[1])  # tied query rows
    args = (
        index_q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
    )
    return args, seq_lens, cu_q, num_seqs


def _packed_scores(args, packed_rows, dma_mode="group", dma_group=4):
    """`block_scores` with 16-page small steps and DMA groups of `dma_group`
    pages (4: dot groups span DMA groups; 16: served alignment)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

    saved = msa_indexer.DMA_MODE, msa_indexer.DMA_GROUP
    msa_indexer.DMA_MODE, msa_indexer.DMA_GROUP = dma_mode, dma_group
    jax.clear_caches()
    try:
        return np.asarray(
            block_scores(
                *args,
                large_block_q=256,
                pages_per_step=4,
                small_pages_per_step=16,
                packed_rows=packed_rows,
                interpret=True,
            )
        )[:, 0]
    finally:
        msa_indexer.DMA_MODE, msa_indexer.DMA_GROUP = saved
        jax.clear_caches()


def _assert_same_visible(got, ref, seq_lens, cu_q, num_seqs):
    assert got.shape == ref.shape
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        rows = slice(cu_q[r], cu_q[r + 1])
        np.testing.assert_array_equal(
            got[rows, :num_pages].view(np.uint32), ref[rows, :num_pages].view(np.uint32)
        )


# (q_lens, ctx_lens, heads, packed_rows)
PACKED_CASES = [
    # DCP2-style decode (1 head x 8 tokens, the served packing): 16 pages
    # per dot; 1..16 pages, dead DMA groups and dead pages inside live packs
    ([8, 8, 1, 8], [0, 2000, 5, 1200], 1, 8),
    # unaligned folded starts, every request packable (4 heads, 32-row tiles)
    ([1, 5, 2, 8, 3], [129, 600, 300, 1015, 40], 4, 32),
    # a block_q (40 rows) and a large (260 rows) request: the v29 classes
    ([1, 5, 10, 8, 65], [129, 600, 300, 1015, 40], 4, 32),
    ([8, 3, 8], [500, 70, 1990], 2, 16),
    ([30, 8], [800, 64], 2, 64),
]


@pytest.mark.parametrize("layout", ["bf16", "fp8", "fp4_folded"])
@pytest.mark.parametrize("q_lens,ctx_lens,heads,packed_rows", PACKED_CASES)
def test_packed_matches_unpacked(q_lens, ctx_lens, heads, packed_rows, layout):
    """The packed tile class (`packed_rows`: 128 / packed_rows pages share
    each dot against a block-diagonal query) and the grouped page DMAs give
    v29's scores (per-page DMAs, 128-row tiles) bit for bit on every page of
    every request, including tied pages, tied rows, -inf dead groups and
    non-finite 4-bit scales."""
    args, seq_lens, cu_q, n = _packed_case(
        q_lens, ctx_lens, layout, heads, sum(ctx_lens) + heads
    )
    ref = _packed_scores(args, None, dma_mode="v29")
    _assert_same_visible(_packed_scores(args, None), ref, seq_lens, cu_q, n)
    _assert_same_visible(_packed_scores(args, packed_rows), ref, seq_lens, cu_q, n)
    _assert_same_visible(
        _packed_scores(args, packed_rows, dma_group=16), ref, seq_lens, cu_q, n
    )
    _assert_same_visible(
        _packed_scores(args, packed_rows, dma_mode="v29"), ref, seq_lens, cu_q, n
    )


@pytest.mark.parametrize(
    "fault,case",
    [
        ("blocks_swapped", 0),  # every packed request
        ("rows_shifted", 1),  # every packed request
        ("dma_last_page", 0),  # requests with a page past the first group
    ],
)
def test_packed_fault_controls(fault, case, monkeypatch):
    """Broken packing / DMAs must fail the parity check: query blocks paired
    with the wrong pages, packed tiles reading their query rows one off, or
    the last DMA group of a request not fetched."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

    q_lens, ctx_lens, heads, packed_rows = PACKED_CASES[case]
    args, seq_lens, cu_q, n = _packed_case(
        q_lens, ctx_lens, "fp4_folded", heads, sum(ctx_lens) + heads
    )
    ref = _packed_scores(args, None)
    if fault == "blocks_swapped":
        orig = msa_indexer._block_diagonal

        def swapped(q, pack):
            bd = orig(q, pack)
            rows = q.shape[0]
            return jnp.concatenate([bd[rows:], bd[:rows]], axis=0)

        monkeypatch.setattr(msa_indexer, "_block_diagonal", swapped)
    elif fault == "rows_shifted":
        orig = msa_indexer._packed_query_rows
        monkeypatch.setattr(
            msa_indexer, "_packed_query_rows", lambda r, n: orig(r, n) + 1
        )
    else:  # grouped DMAs fetch one page short
        orig = msa_indexer._per_group
        monkeypatch.setattr(
            msa_indexer,
            "_per_group",
            lambda fn, pages, live: orig(fn, pages, jnp.maximum(live - 4, 1)),
        )
    got = _packed_scores(args, packed_rows)
    with pytest.raises(AssertionError):
        _assert_same_visible(got, ref, seq_lens, cu_q, n)


@pytest.mark.parametrize(
    "dtype,packed_rows",
    [(jnp.uint8, 32), (jnp.uint8, 8), (jnp.float8_e4m3fn, 8), (jnp.float8_e4m3fn, 32)],
)
def test_packed_lowers_for_tpu(dtype, packed_rows):
    """Packed tiles at served shapes (DCP8 c64 with 4 folded heads, or 1
    head) lower for TPU7x."""
    rows_per_page = 17 if dtype == jnp.uint8 else PAGE // PACKING[dtype]
    tokens, max_seqs, pages_per_seq = 2048, 64, 256
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, rows_per_page, 4, HEAD_DIM), dtype),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(block_scores, packed_rows=packed_rows))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    assert lowered.as_text().count("tpu_custom_call") >= 2
