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
"""The opt-in r6 MSA kernel variants (`models.minimax_m3.kernel_toggles`) in
interpret mode: each gives bit for bit what the default path gives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_dcp

PAGE = 128
TOPK = 16


@pytest.fixture(autouse=True)
def _fresh_traces():
    # The flags are read at trace time; jitted kernels must retrace.
    jax.clear_caches()
    yield
    jax.clear_caches()


def _bits(x):
    x = np.asarray(x)
    return x.view(np.int32) if x.dtype == np.float32 else x


def test_unstable_union_sort_same_unions(monkeypatch):
    rng = np.random.default_rng(0)
    per_tile = jnp.asarray(rng.integers(-1, 40, (16, 256)).astype(np.int32))
    want = np.asarray(msa_attention._sorted_unions(per_tile))
    monkeypatch.setattr(msa_attention, "UNSTABLE_UNION_SORT", True)
    got = np.asarray(msa_attention._sorted_unions(per_tile))
    np.testing.assert_array_equal(want, got)


@pytest.mark.parametrize(
    "num_cols,pretransposed",
    [(146, True), (700, True), (1300, True), (384, False)],
)
def test_fine_scan_widths_same_picks(monkeypatch, num_cols, pretransposed):
    """Prefill-sized steps (> `_MIN_CANDIDATE_TOKENS` rows) with rows at
    every context length up to the full width."""
    rng = np.random.default_rng(num_cols)
    num_tokens, dcp, rank = 1280, 2, 1
    max_len = num_cols * dcp * PAGE
    pos = np.sort(rng.integers(0, max_len, num_tokens)).astype(np.int32)
    s = rng.integers(-3, 4, (num_tokens, num_cols)).astype(np.float32)
    s[rng.random(s.shape) < 0.01] = -np.inf
    scores = jnp.asarray(s.T if pretransposed else s[:, None, :])

    def picks():
        return msa_dcp.local_candidates(
            scores,
            jnp.asarray(pos),
            block_size=PAGE,
            topk=TOPK,
            init_blocks=1,
            local_blocks=2,
            rank=rank,
            dcp=dcp,
            interpret=True,
        )

    want = picks()
    monkeypatch.setattr(msa_dcp, "FINE_SCAN_WIDTHS", True)
    jax.clear_caches()
    assert len(msa_dcp._scan_widths(num_cols, pretransposed)) > 1
    got = picks()
    for w, g in zip(want, got):
        np.testing.assert_array_equal(_bits(w), _bits(g))


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_])
def test_onehot_lookup_same_values(monkeypatch, dtype):
    from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

    rng = np.random.default_rng(1)
    table = jnp.asarray(rng.integers(0, 2 if dtype == jnp.bool_ else 1 << 20, 64)).astype(dtype)
    index = jnp.asarray(rng.integers(0, 64, (3, 100)).astype(np.int32))
    want = np.asarray(msa_indexer.lookup(table, index))
    monkeypatch.setattr(msa_indexer, "ONEHOT_LOOKUP", True)
    got = np.asarray(msa_indexer.lookup(table, index))
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(want, got)


def _indexer_case(rng, dtype, q_lens, ctx_lens):
    from tests.models.minimax_m3.packing import pack, rounded

    page, head_dim, pages_per_seq, max_seqs, num_pages = PAGE, 128, 8, 4, 40
    num_seqs, num_real = len(q_lens), sum(q_lens)
    num_tokens = num_real + 2
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((max_seqs, pages_per_seq), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, num_pages))[
        : num_seqs * pages_per_seq
    ].reshape(num_seqs, pages_per_seq)
    if dtype == "fp4":
        from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

        raw = rng.standard_normal((num_pages, page, head_dim))
        nib, scale = msa_fp4.quantize(
            msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32))
        )
        cache = msa_fp4.pack_rows(nib, scales=scale)
    else:
        keys = rounded(rng.standard_normal((num_pages, page, head_dim)), dtype)
        cache = pack(keys, dtype)
    q = jnp.asarray(rng.standard_normal((num_tokens, 1, head_dim))).astype(
        jnp.bfloat16
    )
    args = (
        q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
    )
    live = [(slice(cu_q[r], cu_q[r + 1]), -(-seq_lens[r] // page)) for r in range(num_seqs)]
    return args, live


def _indexer_scores(args, **kw):
    from vllm_torchtpu.models.minimax_m3.kernels.msa_indexer import block_scores

    return np.asarray(block_scores.__wrapped__(*args, interpret=True, **kw))[:, 0]


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.bfloat16, "fp4"])
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([4, 4, 2], [600, 250, 900]), ([1, 130, 3], [300, 200, 0])]
)
def test_indexer_dma_ring3_same_scores(monkeypatch, dtype, q_lens, ctx_lens):
    """Several page groups per request (pages_per_step 2), so the ring wraps."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

    args, live = _indexer_case(np.random.default_rng(3), dtype, q_lens, ctx_lens)
    kw = dict(
        block_q=64, large_block_q=48, pages_per_step=2, small_pages_per_step=2
    )
    want = _indexer_scores(args, **kw)
    monkeypatch.setattr(msa_indexer, "DMA_RING3", True)
    got = _indexer_scores(args, **kw)
    for rows, n in live:
        np.testing.assert_array_equal(
            _bits(want[rows, :n]), _bits(got[rows, :n])
        )


def test_indexer_fp8_transposed_close(monkeypatch):
    """fp8 caches scored transposed: the same products; checked to f32
    rounding here (the MXU's summation order may differ: check bitwise on
    TPU)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

    args, live = _indexer_case(
        np.random.default_rng(5), jnp.float8_e4m3fn, [4, 4, 2], [600, 250, 900]
    )
    kw = dict(pages_per_step=4, small_pages_per_step=4, folded_tile_q=8)
    want = _indexer_scores(args, **kw)
    monkeypatch.setattr(msa_indexer, "FP8_TRANSPOSED", True)
    assert msa_indexer._fp8_transposed(8, jnp.float8_e4m3fn)
    got = _indexer_scores(args, **kw)
    for rows, n in live:
        np.testing.assert_allclose(got[rows, :n], want[rows, :n], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("block_q", [8, 16])
def test_lse_r6_kernel_matches_default(monkeypatch, dtype, block_q):
    """`msa_attention_lse.R6_KERNEL` dispatches to r6's kernel, which attends
    the same selection: equal outputs (bit for bit on decode-sized tiles,
    which keep the row-major scores; to bf16 rounding on full tiles, whose
    8/16-bit scores run transposed)."""
    from tests.models.minimax_m3.packing import pack, rounded
    from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse

    rng = np.random.default_rng(17)
    num_tokens, pages, topk, heads, hd, q_per_head = 64, 10, 4, 2, 128, 2
    q = jnp.asarray(rng.standard_normal((num_tokens, heads * q_per_head, hd)))
    q = q.astype(jnp.bfloat16)
    rows = pack(
        rounded(rng.standard_normal((pages, PAGE, heads * 2 * hd)), dtype), dtype
    )
    selected = np.full((num_tokens, heads, topk), -1, np.int32)
    for t in range(num_tokens):
        for h in range(heads):
            chosen = rng.choice(pages, rng.integers(0, topk + 1), replace=False)
            selected[t, h, : len(chosen)] = chosen
    args = (
        q,
        rows,
        jnp.asarray(selected),
        jnp.asarray(selected[:, 0, 0]),
        jnp.asarray(rng.integers(1, PAGE + 1, num_tokens), jnp.int32),
        jnp.arange(pages, dtype=jnp.int32),
    )
    kw = dict(sm_scale=hd**-0.5, block_q=block_q, pages_per_step=4, interpret=True)
    want = msa_attention_lse.msa_sparse_attention_lse_heads(*args, **kw)
    monkeypatch.setattr(msa_attention_lse, "R6_KERNEL", True)
    jax.clear_caches()
    got = msa_attention_lse.msa_sparse_attention_lse_heads(*args, **kw)
    np.testing.assert_allclose(
        np.asarray(got[0].astype(jnp.float32)),
        np.asarray(want[0].astype(jnp.float32)),
        atol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[1]), atol=1e-3)


def test_dcp_heads_slot_layout():
    """`msa_dcp_heads` reads v3's `to_slots` candidates: per head and per
    row they are the head's packed candidates (bit patterns)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_dcp_heads

    rng = np.random.default_rng(23)
    num_tokens, heads, topk = 12, 2, 4
    scores = rng.standard_normal((num_tokens, heads, topk)).astype(np.float32)
    blocks = rng.integers(-1, 50, (num_tokens, heads, topk)).astype(np.float32)
    packed = np.concatenate([scores, blocks], axis=-1).reshape(num_tokens, -1)
    slots = msa_dcp.to_slots(jnp.asarray(packed), topk)
    per_head = np.asarray(msa_dcp_heads.slots_per_head(slots, heads))
    assert per_head.shape == (heads, num_tokens, 2 * topk)
    np.testing.assert_array_equal(
        per_head[..., :topk], scores.transpose(1, 0, 2).view(np.int32)
    )
    np.testing.assert_array_equal(
        per_head[..., topk:], blocks.transpose(1, 0, 2).astype(np.int32)
    )
    rows = jnp.asarray([3, 0, 7], jnp.int32)
    picked = msa_dcp_heads.slot_rows(slots, rows, heads)
    want = msa_dcp.to_slots(jnp.asarray(packed[np.asarray(rows)]), topk)
    np.testing.assert_array_equal(np.asarray(picked), np.asarray(want))


def test_dcp_heads_merge_own_matches_merge_candidates():
    """`merge_own` on per-token int32 slots selects what `merge_candidates`
    selects from the same packed f32 candidates."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_dcp_heads

    rng = np.random.default_rng(29)
    dcp, num_tokens, topk = 4, 16, 8
    scores = rng.integers(-3, 4, (dcp, num_tokens, topk)).astype(np.float32)
    blocks = np.stack(
        [
            np.stack([rng.permutation(64)[:topk] * dcp + r for _ in range(num_tokens)])
            for r in range(dcp)
        ]
    ).astype(np.float32)
    # Each rank's list in pick order: score descending, ties to the lower block.
    order = np.lexsort((blocks, -scores), axis=-1)
    scores = np.take_along_axis(scores, order, -1)
    blocks = np.take_along_axis(blocks, order, -1)
    packed = np.concatenate([scores, blocks], axis=-1)  # [dcp, T, 2K]
    want = np.asarray(msa_dcp.merge_candidates(jnp.asarray(packed), topk))
    own = jnp.stack([msa_dcp.to_slots(jnp.asarray(p), topk).T for p in packed])
    got = np.asarray(msa_dcp_heads.merge_own(own, topk, interpret=True))
    np.testing.assert_array_equal(got, want)
