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

from tests.models.minimax_m3.packing import (
    CACHE_DTYPES,
    PACKING,
    pack_index,
    pack_index_fp4,
    rounded,
)
from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_indexer
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
        ([1, 3], [1000, 5], 128, 512, 128),  # small class only, token-major (v3)
        ([1, 3], [1000, 5], 128, 512, 256),  # 256-page groups: copies over 8 chunks
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
    cache = pack_index(keys, dtype)
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


def _served_case(q_lens, ctx_lens, cache_kind, seed):
    """Batch metadata, cache and dequantized keys at the served page group:
    384 pages per request, so the v3 ring runs several 128-page groups."""
    rng = np.random.default_rng(seed)
    pages_per_seq, num_pages = 384, 1200
    num_seqs, max_seqs = len(q_lens), 4
    num_real = sum(q_lens)
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((max_seqs, pages_per_seq), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, num_pages))[
        : num_seqs * pages_per_seq
    ].reshape(num_seqs, pages_per_seq)
    raw = rng.standard_normal((num_pages, PAGE, HEAD_DIM))
    if cache_kind == "fp4":
        nib, scale = msa_fp4.quantize(jnp.asarray(raw, jnp.float32))
        cache = pack_index_fp4(nib, scale)
        keys = np.asarray(msa_fp4.dequantize(nib, scale))
    else:
        keys = rounded(raw, cache_kind)
        cache = pack_index(keys, cache_kind)
    index_q = jnp.asarray(rng.standard_normal((num_real + 5, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    return index_q, cache, keys, seq_lens, tables, cu_q, num_seqs


@pytest.mark.parametrize("cache_kind", [*CACHE_DTYPES, "fp4"])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [
        # Last groups of 44, 80 and 1 live pages: 2, 3 and 1 of 4 chunks scored.
        ([8, 8, 3], [300 * PAGE - 8, 207 * PAGE + 5, 7]),
        # A 100-row request over two 64-row tiles, and a large (v2) request.
        ([100, 8, 200], [290 * PAGE, 128 * PAGE + 3, 40 * PAGE]),
        # A large request in a decode-sized step: the conditional's large path.
        ([8, 150, 8], [200 * PAGE, 60 * PAGE, 3]),
    ],
)
def test_served_page_groups_match_reference(cache_kind, q_lens, ctx_lens):
    """The default kernels at the served page group (128 pages per step)."""
    index_q, cache, keys, seq_lens, tables, cu_q, num_seqs = _served_case(
        q_lens, ctx_lens, cache_kind, len(q_lens) + q_lens[0]
    )
    scores = block_scores(
        index_q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        interpret=True,
    )
    queries = np.asarray(index_q.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    # The large class (v2) scales 4-bit keys in bf16 (msa_indexer.SCALE_KEYS).
    tol = (
        {"rtol": 2e-2, "atol": 5e-2}
        if cache_kind == "fp4"
        else {"rtol": 1e-5, "atol": 1e-4}
    )
    for r in range(num_seqs):
        num_pages_r = -(-seq_lens[r] // PAGE)
        q_r = queries[cu_q[r] : cu_q[r + 1]]
        large = cu_q[r + 1] - cu_q[r] > msa_indexer.DEFAULT_BLOCK_Q
        keys_r = keys[tables[r, :num_pages_r]]
        if large and msa_indexer.FP8_MXU_QUERIES and cache_kind != jnp.bfloat16:
            # The prefill tiles score e4m3 queries (msa_indexer.FP8_MXU_QUERIES).
            q_r = np.asarray(jnp.asarray(q_r).astype(jnp.float8_e4m3fn), np.float32)
        expected = np.einsum("psd,td->tps", keys_r, q_r).max(-1)
        got = scores[cu_q[r] : cu_q[r + 1], :num_pages_r]
        if large and cache_kind == "fp4":
            # 4-bit keys times their scales round to e4m3 too: bounded as in
            # `test_fp4_folded_prefill_tiles_close_to_reference`.
            err = np.abs(got - expected) / np.abs(expected).mean()
            assert err.max() < 0.12 and err.mean() < 0.025, (err.max(), err.mean())
            continue
        np.testing.assert_allclose(got, expected, **tol)


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


@pytest.mark.parametrize(
    "q_lens,ctx_lens,pages_per_step",
    [
        ([1, 3], [1000, 5], 128),  # token-major (decode)
        ([8, 8], [700, 90], 128),  # token-major (spec-decode verify)
        ([8, 8], [700, 90], 256),  # 256-page groups (`msa._wide_decode_scores`)
        ([8, 8], [700, 90], 4),  # page groups
        ([1, 130, 3], [300, 200, 0], 4),  # small and large classes
    ],
)
@pytest.mark.parametrize("token_major", [False, True])
def test_fp4_folded_matches_reference(
    q_lens, ctx_lens, pages_per_step, token_major, monkeypatch
):
    """4-bit index pages with their scales folded in (`msa_fp4`), kernel v3:
    transposed scores (tiles up to `_TRANSPOSED_MAX_ROWS` rows) and
    token-major ones with the scales folded into the bf16 keys."""
    if token_major:
        monkeypatch.setattr(msa_indexer, "_TRANSPOSED_MAX_ROWS", 0)
    rng = np.random.default_rng(len(q_lens) + pages_per_step)
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
    nib, scale = msa_fp4.quantize(
        jnp.asarray(rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)), jnp.float32)
    )
    cache = pack_index_fp4(nib, scale)
    keys = np.asarray(msa_fp4.dequantize(nib, scale))
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
        block_q=64,
        large_block_q=48,
        pages_per_step=pages_per_step,
        small_pages_per_step=pages_per_step,
        kernel="v3",
        large_kernel="v3",
        interpret=True,
    )

    queries = np.asarray(index_q.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        for t in range(cu_q[r], cu_q[r + 1]):
            for b in range(num_pages):
                expected = (keys[tables[r, b]] @ queries[t]).max()
                # Transposed scores are scaled in f32; token-major keys carry
                # bf16-rounded scales.
                tol = dict(rtol=1e-2, atol=5e-2) if token_major else {}
                np.testing.assert_allclose(
                    scores[t, b], expected, **(tol or dict(rtol=1e-4, atol=1e-3))
                )


@pytest.mark.parametrize("pages_per_step", [4, 32])
def test_fp4_folded_prefill_tiles_close_to_reference(pages_per_step):
    """Prefill-sized tiles (> `DEFAULT_BLOCK_Q` rows) score e4m3 queries
    against e4m3 keys times their scales (`FP8_MXU_QUERIES`): within a few
    percent of the exact block maxima (e4m3 queries alone, exact keys: max
    3.9 %, mean 0.84 % of the mean score; with e4m3 keys 9.1 %, 1.6 %)."""
    rng = np.random.default_rng(pages_per_step)
    q_lens, ctx_lens = [1, 300], [700, 200]
    num_seqs, num_real = len(q_lens), sum(q_lens)
    num_tokens = num_real + 5
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    nib, scale = msa_fp4.quantize(
        jnp.asarray(rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)), jnp.float32)
    )
    keys = np.asarray(msa_fp4.dequantize(nib, scale))
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    scores = block_scores(
        index_q,
        pack_index_fp4(nib, scale),
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        block_q=128,
        large_block_q=256,
        pages_per_step=pages_per_step,
        small_pages_per_step=128,
        kernel="v3",
        large_kernel="v3",
        interpret=True,
    )
    queries = np.asarray(index_q.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    r = 1  # the large request
    num_pages = -(-seq_lens[r] // PAGE)
    got = scores[cu_q[r] : cu_q[r + 1], :num_pages]
    want = np.stack(
        [
            (keys[tables[r, :num_pages]] @ queries[t]).max(axis=1)
            for t in range(cu_q[r], cu_q[r + 1])
        ]
    )
    err = np.abs(got - want) / np.abs(want).mean()
    assert err.max() < 0.12 and err.mean() < 0.025, (err.max(), err.mean())


@pytest.mark.parametrize("tokens", [8, 256])
def test_fp4_folded_lowers_for_tpu(tokens):
    max_seqs, pages_per_seq = 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, PAGE // 8 + 1, 4, HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(block_scores, kernel="v3", large_kernel="v3"))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("block_q", [8, 64, 512])
@pytest.mark.parametrize("seed", range(4))
def test_class_work_lists_every_group_of_every_tile(block_q, seed):
    """`class_work`'s items up to `group_ends[-1]`: each live tile's page
    groups in order, tiles in order."""
    rng = np.random.default_rng(seed)
    max_seqs, pages_per_seq, pages_per_step = 12, 40, 8
    num_seqs = int(rng.integers(1, max_seqs + 1))
    q_lens = rng.integers(0, 3 * block_q, num_seqs)
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = q_lens + rng.integers(
        0, pages_per_seq * PAGE - 3 * block_q, num_seqs
    )
    cu = np.full(max_seqs + 1, q_lens.sum(), np.int32)
    cu[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    in_class = np.arange(max_seqs) < num_seqs
    in_class[rng.permutation(num_seqs)[: num_seqs // 3]] = False
    num_tokens = int(cu[-1]) + 5
    max_tiles = num_tokens // block_q + 2 * max_seqs
    work = msa_indexer.class_work(
        jnp.zeros((num_tokens, 1, HEAD_DIM), jnp.bfloat16),
        jnp.asarray(seq_lens),
        jnp.zeros(max_seqs * pages_per_seq, jnp.int32),
        jnp.asarray(cu),
        jnp.asarray(in_class),
        block_q=block_q,
        max_tiles=max_tiles,
        pages_per_step=pages_per_step,
        page_size=PAGE,
    )
    tile_seq = np.asarray(work.tile_seq)
    want = []
    for t in range(int(work.num_tiles)):
        pages = -(-seq_lens[tile_seq[t]] // PAGE)
        want += [(t, g) for g in range(max(-(-pages // pages_per_step), 1))]
    total = int(work.group_ends[-1])
    assert total == len(want)
    got = np.asarray(work.work)[:total]
    np.testing.assert_array_equal(got >> 16, [t for t, _ in want])
    np.testing.assert_array_equal(got & 0xFFFF, [g for _, g in want])
