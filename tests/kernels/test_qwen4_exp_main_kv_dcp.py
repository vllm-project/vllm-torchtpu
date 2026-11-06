# SPDX-License-Identifier: Apache-2.0
"""Main-KV DCP-2 (VLLM_TPU_QSA_MAIN_KV_DCP): owned-row writes and the pair combine.

Writes (interpret mode): with a block-table entry covering 2B tokens, the
rank owning token t ((t // B) % 2) stores exactly the replicated layout's row
for t at (table[t // 2B], t % B); the other rank never touches it.
Combine: merging two ranks' partial softmax attention (out, lse) equals the
full softmax attention over the union of their keys.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_qwen4_exp_fused_kv_write import (  # noqa: F401 (fixture)
    PAGE_ROWS,
    PAGES,
    interpret,
    raw,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa

RATIO, ROTARY, WIDTH = 4, 64, 256


def _writes(query_lens, contexts, dcp, seed=0):
    """Pool after qsa_cache_writes: replicated tables (dcp None) or 2B tables."""
    rng = np.random.default_rng(seed)
    requests = len(contexts)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    tokens = max(8, -(-int(starts[-1]) // 8) * 8)
    span = PAGE_ROWS * (dcp[1] if dcp else 1)
    columns = 4
    pages = np.arange(1, 1 + requests * columns, dtype=np.int32).reshape(requests, columns)
    main = pages
    comp = pages + requests * columns
    ring = np.arange(PAGES - requests, PAGES, dtype=np.int32)
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(query_lens[r])
    data = np.random.default_rng(99)
    k = jnp.asarray(data.normal(0, 1, (tokens, 1, WIDTH)), jnp.bfloat16)
    v = jnp.asarray(data.normal(0, 1, (tokens, 1, WIDTH)), jnp.bfloat16)
    raw_keys = jnp.asarray(data.normal(0, 1, (tokens, 128)), jnp.bfloat16)
    pool = jnp.zeros((PAGES, PAGE_ROWS, 1, 4, WIDTH), jnp.float8_e4m3fn)
    trig = jnp.zeros((4096, 128), jnp.bfloat16)
    del span
    out, *_ = pooled_qsa.qsa_cache_writes(
        pool, k, v, raw_keys, jnp.asarray(positions), jnp.ones((128,), jnp.bfloat16), trig,
        jnp.asarray(starts), jnp.asarray(contexts, jnp.int32), jnp.asarray(ring),
        jnp.asarray(main), jnp.asarray(comp), compress_ratio=RATIO, norm_eps=1e-6,
        rotary_dim=ROTARY, dcp=dcp,
    )
    return raw(out), starts, positions, main


@pytest.mark.parametrize(
    "query_lens,contexts",
    [([700, 4, 4], [100, 509, 1030]), ([4, 4, 4, 4], [37, 511, 512, 1500]), ([1000], [0])],
)
def test_owned_rows_match_the_replicated_layout(interpret, query_lens, contexts):
    want, starts, positions, main = _writes(query_lens, contexts, None)
    got = [
        _writes(query_lens, contexts, (jnp.asarray([r], jnp.int32), 2))[0] for r in (0, 1)
    ]
    for r, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        for t in range(begin, end):
            pos = int(positions[t])
            owner = (pos // PAGE_ROWS) % 2
            ref = want[main[r, pos // PAGE_ROWS], pos % PAGE_ROWS]
            row = got[owner][main[r, pos // (2 * PAGE_ROWS)], pos % PAGE_ROWS]
            np.testing.assert_array_equal(row, ref, f"request {r} token {pos}")
            other = got[1 - owner][main[r, pos // (2 * PAGE_ROWS)], pos % PAGE_ROWS]
            # The non-owner never wrote this row: still the zero pool, unless
            # it owns another token mapped there (it cannot: rows are disjoint).
            assert not other.any() or (other != row).any(), (r, pos)


def _attend(q, k, v, mask):
    s = np.einsum("thd,khd->thk", q, k) / np.sqrt(q.shape[-1])
    s = np.where(mask[:, None, :], s, -np.inf)
    m = s.max(-1, keepdims=True)
    p = np.where(np.isfinite(m), np.exp(s - np.where(np.isfinite(m), m, 0)), 0)
    den = p.sum(-1, keepdims=True)
    out = np.einsum("thk,khd->thd", p, v) / np.where(den > 0, den, 1)
    lse = np.where(den[..., 0] > 0, np.log(np.where(den[..., 0] > 0, den[..., 0], 1)) + m[..., 0], -np.inf)
    return out, lse


def test_pair_combine_equals_full_attention():
    rng = np.random.default_rng(0)
    tokens, heads, keys, dim = 16, 6, 64, 32
    q = rng.normal(size=(tokens, heads, dim))
    k = rng.normal(size=(keys, 1, dim)).repeat(heads, 1)
    v = rng.normal(size=(keys, 1, dim)).repeat(heads, 1)
    visible = rng.random((tokens, keys)) < 0.5
    visible[3] = False  # a row with no key anywhere
    owner = (np.arange(keys) // 8) % 2
    visible[5] &= owner == 0  # a row only rank 0 holds keys of
    full, _ = _attend(q, k, v, visible)
    parts = [_attend(q, k, v, visible & (owner == r)[None]) for r in (0, 1)]
    out_pair = jnp.asarray(np.stack([p[0] for p in parts]), jnp.float32)
    lse_pair = jnp.asarray(np.stack([p[1] for p in parts]), jnp.float32)
    gate = jnp.full((tokens, heads // 2, dim), 30.0, jnp.bfloat16)  # sigmoid ~ 1
    for rank in (0, 1):
        got = np.asarray(
            qsa_rpa.dcp_combine(out_pair, lse_pair, gate, jnp.asarray([rank], jnp.int32)),
            np.float32,
        )
        want = full[:, rank * 3 : rank * 3 + 3]
        assert np.all(got[3] == 0)
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


# --- interleaved ownership (PCP: units of I tokens alternate over the ranks) ---


@pytest.mark.parametrize("world,interleave", [(2, 128), (2, 256), (4, 128), (2, PAGE_ROWS)])
def test_owner_row_is_a_bijection_per_entry(world, interleave):
    span = world * PAGE_ROWS
    t = np.arange(3 * span)
    owner, row = qsa_rpa.dcp_owner_row(t, PAGE_ROWS, world, interleave)
    owner, row = np.asarray(owner), np.asarray(row)
    assert np.all((0 <= row) & (row < PAGE_ROWS))
    np.testing.assert_array_equal(owner, (t // interleave) % world)
    for entry in range(3):
        sel = (t // span) == entry
        for r in range(world):
            rows = row[sel & (owner == r)]
            # Every rank fills each row of its page of the entry exactly once.
            np.testing.assert_array_equal(np.sort(rows), np.arange(PAGE_ROWS))
    if interleave == PAGE_ROWS:
        # The decode's layout: whole pages alternate.
        np.testing.assert_array_equal(owner, (t // PAGE_ROWS) % world)
        np.testing.assert_array_equal(row, t % PAGE_ROWS)


@pytest.mark.parametrize(
    "query_lens,contexts",
    [([700, 4, 4], [100, 509, 1030]), ([4, 4, 4, 4], [37, 511, 512, 1500]), ([1000], [0])],
)
def test_interleaved_owned_rows_match_the_replicated_layout(interpret, query_lens, contexts):
    interleave = 128
    want, starts, positions, main = _writes(query_lens, contexts, None)
    got = [
        _writes(query_lens, contexts, (jnp.asarray([r], jnp.int32), 2, interleave))[0]
        for r in (0, 1)
    ]
    for r, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        for t in range(begin, end):
            pos = int(positions[t])
            owner, row = (int(x) for x in qsa_rpa.dcp_owner_row(pos, PAGE_ROWS, 2, interleave))
            ref = want[main[r, pos // PAGE_ROWS], pos % PAGE_ROWS]
            entry = main[r, pos // (2 * PAGE_ROWS)]
            np.testing.assert_array_equal(got[owner][entry, row], ref, f"request {r} token {pos}")


def test_pcp_combine_merges_every_head():
    rng = np.random.default_rng(1)
    tokens, heads, keys, dim = 16, 6, 64, 32
    q = rng.normal(size=(tokens, heads, dim))
    k = rng.normal(size=(keys, 1, dim)).repeat(heads, 1)
    v = rng.normal(size=(keys, 1, dim)).repeat(heads, 1)
    visible = rng.random((tokens, keys)) < 0.5
    visible[2] = False
    owner = (np.arange(keys) // 4) % 2
    full, _ = _attend(q, k, v, visible)
    parts = [_attend(q, k, v, visible & (owner == r)[None]) for r in (0, 1)]
    out_pair = jnp.asarray(np.stack([p[0] for p in parts]), jnp.float32)
    lse_pair = jnp.asarray(np.stack([p[1] for p in parts]), jnp.float32)
    gate = jnp.full((tokens, heads, dim), 30.0, jnp.bfloat16)
    got = np.asarray(qsa_rpa.pcp_combine(out_pair, lse_pair, gate), np.float32)
    assert np.all(got[2] == 0)
    np.testing.assert_allclose(got, full, rtol=2e-2, atol=2e-2)


def _plain_page_ids(monkeypatch):
    """Without torch_tpu (CPU), stand in a plain table lookup for the SparseCore
    page-id gather the metadata uses; on TPU the real one runs."""
    try:
        import vllm_torchtpu.kernels.mla.sparse.kernel  # noqa: F401
    except ImportError:
        import sys
        import types

        fake = types.ModuleType("vllm_torchtpu.kernels.mla.sparse.kernel")
        fake.gather_page_ids = lambda table, ids, seqs, n, **_: table.reshape(n, -1)[
            seqs[:, None], jnp.clip(ids, 0, table.shape[0] // n - 1)
        ]
        monkeypatch.setitem(sys.modules, fake.__name__, fake)


@pytest.mark.parametrize("interleave", [128, PAGE_ROWS])
def test_interleaved_decode_metadata_points_at_owned_rows(monkeypatch, interleave):
    """The decode path's micro-pages land on the rows the writes used."""
    _plain_page_ids(monkeypatch)
    world, ratio = 2, RATIO
    page_groups = PAGE_ROWS // ratio
    lengths = jnp.asarray([3 * world * PAGE_ROWS - 2], jnp.int32)
    groups = np.arange(0, 3 * world * page_groups - 1, 7, dtype=np.int32)
    budget = len(groups)
    blocks = jnp.asarray(groups[None, :], jnp.int32)
    table = jnp.asarray([[11, 22, 33]], jnp.int32)
    for rank in (0, 1):
        pages, kv_lens, _, _ = qsa_rpa.virtual_page_metadata(
            blocks, lengths, table, ratio=ratio, page_size=PAGE_ROWS,
            dcp=(jnp.asarray([rank], jnp.int32), world, interleave),
        )
        pages = np.asarray(pages)[0]
        owned = [g for g in groups if ((g * ratio) // interleave) % world == rank]
        for i, g in enumerate(owned):
            owner, row = qsa_rpa.dcp_owner_row(g * ratio, PAGE_ROWS, world, interleave)
            entry = int(np.asarray(table)[0, (g * ratio) // (world * PAGE_ROWS)])
            assert int(owner) == rank
            assert pages[i] == entry * page_groups + int(row) // ratio, (rank, g)
        assert budget >= len(owned)
