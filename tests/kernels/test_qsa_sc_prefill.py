# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of the SparseCore QSA prefill planning (no kernel runs).

The page plan, the context-parallel selection rewrite and the slab fields
decide which cache row every gathered slot reads; these tests check that,
composed with the SparseCore gather's address rule, every live slot of a query
reads exactly the cache row of one of its selected tokens (owned tokens only
under DCP) and every selected token is read once.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_gqa_prefill import owner_window as window
from vllm_torchtpu.kernels.sparse_gqa_prefill import slab_gather
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


def inputs(starts, lengths, columns=5, page_size=128, budget=8):
    starts = np.asarray(starts, np.int32)
    lengths = np.asarray(lengths, np.int32)
    table = np.full((len(lengths), columns), np.iinfo(np.int32).max, np.int32)
    blocks = np.full((len(lengths), budget), -1, np.int32)
    for request, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        prefix = (max(lengths[a:b], default=0) + page_size - 1) // page_size
        # Repeated physical pages deliberately preserve aliasing.
        table[a:b, :prefix] = (np.arange(prefix) + request) % 3
    for row, length in enumerate(lengths):
        count = min(length // 4, budget)
        blocks[row, :count] = np.arange(count)
    return blocks, lengths, table, starts


def plan(xs, data_pages, dcp_world=1):
    return window.plan_pages(
        *map(jnp.asarray, xs),
        page_size=128,
        owner_pages=9,
        ratio=4,
        data_pages=data_pages,
        dcp_world=dcp_world,
    )


def address_oracle(xs, entry_tokens=128):
    blocks, lengths, table, starts = xs
    ids = []
    remapped = np.zeros_like(table)
    for a, b in zip(starts[:-1], starts[1:]):
        count = (max(lengths[a:b], default=0) + entry_tokens - 1) // entry_tokens
        offset = len(ids)
        if count:
            ids.extend(table[a, :count])
        for row in range(a, b):
            if lengths[row] > 0:
                remapped[row, :count] = np.arange(count) + offset + 1
    return np.asarray(ids, np.int32), remapped


@pytest.mark.parametrize(
    "starts,lengths",
    [
        ([0, 4], [1, 127, 128, 129]),
        ([0, 0, 2, 2, 4, 4], [129, 257, 1, 128]),
        ([0, 1, 4], [257, 127, 128, 129]),
        ([0, 2, 2, 4], [0, 0, 127, 129]),
        ([0, 4, 4, 4], [0, 0, 0, 0]),
    ],
)
def test_prefix_addresses_and_aliases(starts, lengths):
    xs = inputs(starts, lengths)
    ids, remapped = address_oracle(xs)
    p = plan(xs, max(1, len(ids)))
    assert bool(p.valid)
    assert int(p.count) == len(ids)
    np.testing.assert_array_equal(p.physical[: len(ids)], ids)
    np.testing.assert_array_equal(p.table, remapped)


@pytest.mark.parametrize(
    "mutation",
    [
        "capacity",
        "start_first",
        "start_last",
        "start_order",
        "negative_length",
        "large_length",
        "physical_negative",
        "physical_oob",
        "table_mismatch",
        "selected_negative",
        "selected_oob",
    ],
)
def test_failed_certificate(mutation):
    xs = list(inputs([0, 2, 4], [129, 257, 127, 129]))
    capacity = 5
    if mutation == "capacity":
        capacity = 4
    elif mutation == "start_first":
        xs[3][0] = 1
    elif mutation == "start_last":
        xs[3][-1] = 3
    elif mutation == "start_order":
        xs[3] = np.array([0, 3, 2, 4], np.int32)
    elif mutation == "negative_length":
        xs[1][0] = -1
    elif mutation == "large_length":
        xs[1][0] = np.iinfo(np.int32).max
    elif mutation == "physical_negative":
        xs[2][:2, 0] = -1
    elif mutation == "physical_oob":
        xs[2][:2, 0] = 9
    elif mutation == "table_mismatch":
        xs[2][1, 0] = 8
    elif mutation == "selected_negative":
        xs[0][0, 0] = -1
    elif mutation == "selected_oob":
        xs[0][0, 0] = xs[1][0] // 4
    assert not bool(plan(xs, capacity).valid)


def test_dcp_plan_counts_table_entries():
    # One entry covers 2 * 128 tokens: 257 tokens need two entries.
    xs = inputs([0, 4], [1, 200, 256, 257], page_size=256)
    ids, remapped = address_oracle(xs, entry_tokens=256)
    p = plan(xs, 5, dcp_world=2)
    assert bool(p.valid)
    assert int(p.count) == len(ids) == 2
    np.testing.assert_array_equal(p.table, remapped)
    # 5 entries of 256 tokens bound the lengths, not 5 pages of 128.
    xs[1][0] = 5 * 256 + 1
    assert not bool(plan(xs, 5, dcp_world=2).valid)


def test_window_pages_bound():
    import jax

    cache = jax.ShapeDtypeStruct((15561, 512, 1, 4, 256), jnp.float8_e4m3fn)
    table = jax.ShapeDtypeStruct((8192, 512), jnp.int32)
    page_bytes = 512 * 4 * 256
    narrowed = jax.ShapeDtypeStruct((2,), jnp.int32)
    assert window.window_data_pages(cache, narrowed, table, None) == 512
    wide = jax.ShapeDtypeStruct((65,), jnp.int32)
    assert window.window_data_pages(cache, wide, table, None) == min(
        64 * 512, SC_GATHER_MAX_CACHE_BYTES // page_bytes - 1
    )


@pytest.mark.parametrize(
    "starts,expected,single",
    [
        ([0, 0, 0, 37, 37], [0, 37], True),
        ([0, 5, 5, 9], [0, 5], False),
        ([0, 0, 0], [0, 0], False),
        ([3, 3, 11], [3, 11], True),
    ],
)
def test_single_request_starts(starts, expected, single):
    narrowed, ok = window.single_request_starts(
        jnp.asarray(starts, jnp.int32), jnp.zeros((16,), jnp.int32)
    )
    if single:
        np.testing.assert_array_equal(narrowed, expected)
    assert bool(ok) == single


def gathered_tokens(blocks, fields, table_row, *, ratio, page_size, budget):
    """The SparseCore gather's address rule (slab_gather.address_words) for one
    query: slot tokens in list order -> (table entry, row in page)."""
    count, selected_len, tail = (int(f) for f in fields)
    out = []
    for token in range(selected_len):
        slot = token // ratio
        group = blocks[slot] if slot < budget else 0
        if slot == count:
            group = tail
        entry = group // (page_size // ratio)
        row = group % (page_size // ratio) * ratio + token % ratio
        out.append((int(table_row[entry]), row))
    return out


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_reads_each_owned_token_once(world, seed):
    rng = np.random.default_rng(seed)
    ratio, page, budget, rows = 4, 32, 12, 24
    lengths = rng.integers(0, 400, rows).astype(np.int32)
    lengths[:3] = [0, 3, 401]
    blocks = np.full((rows, budget), -1, np.int32)
    for r, n in enumerate(lengths):
        count = min(n // ratio, budget)
        blocks[r, :count] = rng.choice(n // ratio, count, replace=False)
    entries = 16
    # Rank-local table: entry e holds global page e * world + rank.
    physical = rng.permutation(64)[:entries]
    for rank in range(world):
        if world == 1:
            local, fields = blocks, slab_gather.selection_fields(
                jnp.asarray(lengths), ratio=ratio, budget=budget
            )
        else:
            local, fields = window.dcp_selection(
                jnp.asarray(blocks),
                jnp.asarray(lengths),
                ratio=ratio,
                page_size=page,
                world=world,
                rank=jnp.asarray([rank], jnp.int32),
            )
        local = np.asarray(local)
        fields = [np.asarray(f) for f in fields]
        for r, n in enumerate(lengths):
            count = min(n // ratio, budget)
            expect = set()
            groups = list(blocks[r, :count]) + ([n // ratio] if n % ratio else [])
            for g in groups:
                for t in range(g * ratio, min(g * ratio + ratio, n)):
                    gpage = t // page
                    if gpage % world == rank:
                        expect.add((int(physical[gpage // world]), t % page))
            got = gathered_tokens(
                local[r],
                [f[r] for f in fields],
                physical,
                ratio=ratio,
                page_size=page,
                budget=budget,
            )
            assert len(got) == len(set(got))
            assert set(got) == expect, (rank, r)


def test_context_threshold():
    lengths = jnp.asarray([0, 10, 99, 50, 300], jnp.int32)
    starts = jnp.asarray([1, 4], jnp.int32)
    assert bool(window.context_reaches(lengths, starts, 0))
    assert bool(window.context_reaches(lengths, starts, 99))
    assert not bool(window.context_reaches(lengths, starts, 100))


def test_flag_off_keeps_tensorcore_path(monkeypatch):
    """With VLLM_TPU_QSA_PREFILL_SC unset the prefill call never consults the
    SparseCore module (default path unchanged)."""
    from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
    from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles, qsa_rpa

    monkeypatch.delenv("VLLM_TPU_QSA_PREFILL_SC", raising=False)
    calls = []
    monkeypatch.setattr(
        bsp, "block_sparse_gqa_attention", lambda *a, **k: calls.append(k) or ("tc", a[5])
    )
    monkeypatch.setattr(
        prefill_tiles, "get_sparse_prefill_tiles", lambda *a, **k: (256, 512, None)
    )
    monkeypatch.setattr(
        window, "is_supported", lambda *a, **k: pytest.fail("SC consulted")
    )
    cache = object()
    out = qsa_rpa._prefill_attention(
        None, None, None, None, None, cache, None, kv_heads=1, ratio=4
    )
    assert out == ("tc", cache) and len(calls) == 1


def test_flag_on_unsupported_keeps_tensorcore_path(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
    from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles, qsa_rpa

    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_SC", "1")
    monkeypatch.setattr(
        bsp, "block_sparse_gqa_attention", lambda *a, **k: ("tc", a[5])
    )
    monkeypatch.setattr(
        prefill_tiles, "get_sparse_prefill_tiles", lambda *a, **k: (256, 512, None)
    )
    seen = []
    monkeypatch.setattr(window, "is_supported", lambda *a, **k: seen.append(k) or False)
    cache = object()
    out = qsa_rpa._prefill_attention(
        None, None, None, None, None, cache, None, kv_heads=1, ratio=4,
        kv_head_offset=1,
    )
    assert out == ("tc", cache)
    assert seen and seen[0]["kv_head_offset"] == 1
