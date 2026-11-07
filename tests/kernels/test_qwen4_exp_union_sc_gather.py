# SPDX-License-Identifier: Apache-2.0
"""SparseCore union gather kernels (``qsa_union_sc_gather``) against XLA references.

``union_sc_list`` must return ``qsa_union_dense.union_gather_list_reference``'s
``(rows, ids)`` and ``union_sc_gather`` its ``union_gather_reference``'s
``(gathered_kv, lines)``, bit for bit. For the run-table walk,
``union_sc_run_tables`` must return ``union_run_tables_xla``'s ``(counts,
entries)`` and ``union_sc_lines`` ``union_lines_reference``'s lines.

Two kinds of test:

- CPU (always run). The kernels' index math lives in pure helpers that the
  kernels call. Two numpy emulations replay each kernel's loop structure (the
  list step's carried-rank compressed stores and padding fill; the gather
  step's items, streams, windows, KV rows, bitset rows, lines and dead-tile
  skip) through those same helpers and must equal the references. The
  run-table emulation goes further: it runs the kernel's own tile code
  (``run_table_tile``) on numpy stand-ins for the SparseCore primitives, whose
  compressed store writes garbage past the popcount. A lowering test traces
  every kernel variant and lowers it to Mosaic for an abstract TPU7x device.
  These check the index math and the Pallas-level lowering, not the SparseCore
  compile or the primitives' hardware semantics.
- TPU (skip off a SparseCore TPU). The real kernels against the references on
  synthetic and ``union_bits_plan`` inputs, one realistic size (50 query
  groups, cap 10,624, kv_tile 1024), and end to end: the attention kernel on
  the SparseCore-gathered tiles is bitwise the walk. The run tables and lines
  likewise, on ``union_bits_plan`` inputs at small sizes and at 50 query groups
  with a cap near 10K (G64 and G128), and end to end: the run-table walk on the
  SparseCore list, tables and lines is bitwise the walk.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_sc_gather as sg

RATIO = 4
WORKERS = 32  # v7x: 2 SparseCores x 16 vector subcores
LANES = sg.LANES
LIST_CASES = ["exact", "per_group_rows", "short", "long", "truncated", "empty"]
GATHER_CASES = [
    (True, None, False),
    (False, None, False),
    (True, 1, True),
    (False, 1, True),
]


def _on_sparsecore():
    if jax.default_backend() != "tpu":
        return False
    try:
        sc = pltpu.get_tpu_info().sparse_core
    except (ValueError, RuntimeError):
        return False
    return sc is not None and sc.num_lanes == LANES


@pytest.fixture
def sparsecore():
    # checked when a test runs, not in a module-level skipif: collection must not initialize a JAX backend
    # (tests/conftest.py refuses that)
    if not _on_sparsecore():
        pytest.skip("needs a TPU with SparseCores (v7x)")


_skip_off_sparsecore = pytest.mark.usefixtures("sparsecore")


def tpu_only(test):
    return pytest.mark.tpu_test(_skip_off_sparsecore(test))


# Inputs.


def _pack(bools):
    """bool[..., 32] -> int32[...]: bit i of the word is entry i."""
    weights = np.uint64(1) << np.arange(32, dtype=np.uint64)
    words = (bools.astype(np.uint64) * weights).sum(-1)
    return words.astype(np.uint32).view(np.int32)


def _synthetic(
    seed,
    *,
    ngroups,
    words,
    group=64,
    density=0.35,
    page_size=256,
    per_group_rows=False,
    pool_pages=None,
    dense_group=None,
):
    """Union bitsets as ``union_bits_plan`` returns them, a page table and a pool.

    Each query group's union has about ``density`` of its ``words * 32`` ids
    set; each row of the group selects a random subset, and every union id is
    selected by at least one row, so ``ubits`` is the OR of the rows' bits.
    The page table is a permutation; the fp8 pool is random bytes.
    """
    rng = np.random.default_rng(seed)
    union = rng.random((ngroups, words, 32)) < density
    union[0, 0, 31] = True  # a sign-bit word
    union[-1, -1] = True  # an all-ones word
    if dense_group is not None:  # a union past the cap
        union[dense_group] |= rng.random((words, 32)) < 0.05
    union_w = _pack(union).view(np.uint32)
    size = (ngroups, words, group)
    rows_sel = rng.integers(0, 2**32, size=size, dtype=np.uint32)
    rows_sel &= rng.integers(0, 2**32, size=size, dtype=np.uint32)
    rows_sel &= union_w[..., None]
    rows_sel[..., 0] |= union_w & ~np.bitwise_or.reduce(rows_sel, axis=-1)
    bits = np.zeros((ngroups, words, 128), np.uint32)
    bits[..., :group] = rows_sel
    ubits = np.bitwise_or.reduce(bits, axis=-1)
    usize = np.unpackbits(ubits.view(np.uint8), axis=-1).sum(-1).astype(np.int32)
    pages_per_row = -(-words * 32 * RATIO // page_size)
    pages = pool_pages or pages_per_row + 2
    if per_group_rows:
        perms = [rng.permutation(pages)[:pages_per_row] for _ in range(ngroups)]
        page_rows = np.stack(perms)
    else:
        page_rows = rng.permutation(pages)[:pages_per_row]
    raw = rng.integers(0, 256, size=(pages, page_size, 1, 4, 256), dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return (
        jnp.asarray(bits.view(np.int32)),
        jnp.asarray(ubits.view(np.int32)),
        jnp.asarray(usize),
        jnp.asarray(page_rows.astype(np.int32)),
        pool,
    )


def _physw(ubits, page_rows, page_size):
    return ud._word_bases(
        page_rows, words=ubits.shape[1], page_size=page_size, word_tokens=32 * RATIO
    )


def _u8(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def _reference_list(ubits, usize, page_rows, *, cap, kv_tile, page_size=256):
    return ud.union_gather_list_reference(
        ubits,
        usize,
        page_rows,
        ratio=RATIO,
        page_size=page_size,
        union_cap=cap,
        kv_tile=kv_tile,
    )


def _read_ranks(usize, *, cap, kv_tile, cap_ranks):
    """bool[ngroups * cap_ranks]: the ranks of the tiles the attention kernel reads.

    ``union_walk_kernel``: tiles = min(cdiv(usize * ratio, kv_tile), cap_tiles).
    """
    cap_tiles = -(-cap * RATIO // kv_tile)
    tiles = np.minimum(-(-np.asarray(usize) * RATIO // kv_tile), cap_tiles)
    live = tiles * (kv_tile // RATIO)
    return (np.arange(cap_ranks)[None, :] < live[:, None]).reshape(-1)


# CPU emulations: each kernel's loop structure through the module's helpers.


def _emulate_list(ubits, usize, physw, *, cap_ranks):
    """``union_sc_list``'s per-group scan and padding fill, one half-word at a time."""
    ubits, usize = np.asarray(ubits), np.asarray(usize)
    ngroups, words = ubits.shape
    physw = np.broadcast_to(np.asarray(physw), (ngroups, words))
    shape = (words, LANES)  # all words at once; the kernel does one word's (16,)
    lane = jnp.broadcast_to(jnp.arange(LANES, dtype=jnp.int32), shape)
    w = jnp.broadcast_to(jnp.arange(words, dtype=jnp.int32)[:, None], shape)
    rows_out = np.empty((ngroups, cap_ranks), np.int32)
    ids_out = np.empty((ngroups, cap_ranks), np.int32)
    for g in range(ngroups):
        word = jnp.broadcast_to(jnp.asarray(ubits[g])[:, None], shape)
        base = jnp.broadcast_to(jnp.asarray(physw[g])[:, None], shape)
        halves = [
            [np.asarray(x) for x in sg.expand_half_word(word, base, w, pos, RATIO)]
            for pos in sg.half_positions(lane)
        ]
        # Unwritten slots keep a sentinel no output can hold.
        ids_v = np.full(cap_ranks + 2 * LANES, -7, np.int32)
        rows_v = np.full(cap_ranks + 2 * LANES, -7, np.int32)
        limit = int(sg.list_limit(int(usize[g]), cap_ranks))
        rank = 0
        for k in range(words):
            for mask, gid, row in halves:
                at = int(sg.clamp_rank(rank, limit))
                m = mask[k]
                n = int(m.sum())
                ids_v[at : at + n] = gid[k][m]  # store_compressed: set lanes, packed
                rows_v[at : at + n] = row[k][m]
                rank += n
        start = int(sg.clamp_rank(rank, limit))
        for i in range(int(sg.pad_vectors(start, cap_ranks))):
            ids_v[start + i * LANES : start + (i + 1) * LANES] = -1
            rows_v[start + i * LANES : start + (i + 1) * LANES] = np.asarray(sg.padding_rows(jnp.asarray(physw[g, :LANES])))
        rows_out[g], ids_out[g] = rows_v[:cap_ranks], ids_v[:cap_ranks]
    return rows_out, ids_out


def _gather_plan_for(pool, bits, cap_ranks, line_matmul, streams=None):
    _, _, head_axis, _, dim = pool.shape
    return sg.choose_gather_plan(
        bits.shape[0],
        cap_ranks,
        workers=WORKERS,
        group_bytes=RATIO * head_axis * dim * 4,
        width=bits.shape[2],
        repeat=1 if line_matmul else RATIO,
        streams=streams,
    )


def _emulate_gather(pool, bits, rows, ids, *, line_matmul, plan, item_steps=None):
    """``union_sc_gather``: every worker's items, steps and streams, kernel-indexed.

    Returns the KV bytes, the lines and per-row write counts (rows never
    written keep sentinels). ``pool`` and ``rows`` None: ``union_sc_lines``
    (lines only; KV and its write counts are None).
    """
    ngroups, cap_ranks = ids.shape
    _, words, width = bits.shape
    repeat = 1 if line_matmul else RATIO
    ranks = ngroups * cap_ranks
    with_kv = pool is not None
    kv = kv_writes = None
    if with_kv:
        pool_u8 = _u8(pool)
        groups = pool_u8.reshape(-1, RATIO, *pool_u8.shape[2:])
        kv = np.full((ranks, *groups.shape[1:]), 0xA5, np.uint8)
        kv_writes = np.zeros(ranks, np.int32)
        rows_f = np.asarray(rows).reshape(-1)
    lines = np.full((ranks * repeat, width), -7, np.int32)
    line_writes = np.zeros(ranks * repeat, np.int32)
    ids_f = np.asarray(ids).reshape(-1)
    bits_f = np.asarray(bits).reshape(ngroups * words, width)
    jlane = jnp.arange(plan.lanes, dtype=jnp.int32)
    span = plan.lanes
    for worker in range(plan.workers):
        for k in range(plan.items_per_worker):
            item = int(sg.worker_item(worker, k, plan))
            block = int(sg.staged_block(item, plan))
            staged = slice(block * plan.part_ranks, (block + 1) * plan.part_ranks)
            ids_v = ids_f[staged]
            g, part, rank0 = (int(x) for x in sg.item_coords(jnp.int32(item), plan))
            if item_steps is None:
                steps = plan.steps_per_part
            else:
                steps = int(item_steps[block])
            if not (item < plan.items and steps > 0):
                continue
            for step in range(steps):
                for s in range(plan.streams):
                    off = int(sg.window_offset(step, s, plan))
                    iw = jnp.asarray(ids_v[off : off + span])
                    if with_kv:
                        rw = jnp.asarray(rows_f[staged][off : off + span])
                        dst = int(sg.window_rank(rank0, step, s, plan))
                        kv_rows = np.asarray(sg.kv_row_index(rw, RATIO))
                        kv[dst : dst + span] = groups[kv_rows]
                        kv_writes[dst : dst + span] += 1
                    brows = bits_f[np.asarray(sg.bits_row_index(iw, g * words))]
                    line = np.asarray(sg.line_of(jnp.asarray(brows), iw[:, None]))
                    base = int(sg.lines_block(item, step, plan)) * plan.chunk * repeat
                    for t in range(repeat):
                        at = base + np.asarray(sg.line_row(s, jlane, t, plan, repeat))
                        lines[at] = line
                        line_writes[at] += 1
    return kv, lines, kv_writes, line_writes


# Run tables: the kernel's own tile code on numpy stand-ins for the primitives.

GARBAGE = -777  # what the stand-in compressed store leaves past the popcount
TABLE_CASES = [(64, 512, 8), (64, 1024, 4), (128, 1024, 8), (32, 256, 2), (64, 512, 32), (64, 1024, 1),
               (64, 1024, 16)]  # R16: 5 sizes, the other odd carry above 1 (carry_pad)
TABLE_EDGE_CASES = ["one_run", "vector_edges", "padding_contiguity", "decreasing", "all_padding", "mixed"]


class _NumpyOps:
    """``sg.ScOps`` on numpy buffers, so the emulation runs ``sg.run_table_tile`` itself.

    ``store_compressed`` writes the selected lanes packed, then ``GARBAGE`` on
    the rest of its 16 slots: the hardware may leave those slots alone, and the
    kernel must not depend on either. Every access asserts it stays inside its
    buffer (the kernel runs with bounds checks off).
    """

    @staticmethod
    def lane():
        return jnp.arange(LANES, dtype=jnp.int32)

    @staticmethod
    def _at(ref, at):
        at = int(at)
        assert 0 <= at and at + LANES <= ref.size, (at, ref.size)
        return at

    def load(self, ref, at):
        at = self._at(ref, at)
        return jnp.asarray(ref[at : at + LANES])

    def store(self, ref, at, x):
        at = self._at(ref, at)
        ref[at : at + LANES] = np.asarray(x)

    def store_compressed(self, ref, at, x, mask):
        at = self._at(ref, at)
        keep = np.asarray(mask)
        n = int(keep.sum())
        ref[at : at + n] = np.asarray(x)[keep]
        ref[at + n : at + LANES] = GARBAGE

    @staticmethod
    def gather(ref, index):
        index = np.asarray(index)
        assert (index >= 0).all() and (index < ref.size).all(), index
        return jnp.asarray(ref[index])

    @staticmethod
    def cumsum(x):
        return jnp.cumsum(x, dtype=jnp.int32)

    @staticmethod
    def popcount(mask):
        return int(np.asarray(mask).sum())

    @staticmethod
    def loop(lower, upper, *, carry=None, unroll=1):
        def run(body):
            state = carry
            for i in range(int(lower), int(upper)):
                if carry is None:
                    body(i)
                else:
                    state = body(i, state)
            return state

        return run


def _emulate_tables(rows, ids, *, max_run, plan, even_carry=True, copy_unroll=1):
    """``union_sc_run_tables``: every worker's steps, every tile through ``sg.run_table_tile``.

    The scratch persists across a worker's tiles and starts as sentinels; each
    step's output blocks are written back whole, stale slots included (as the
    pipeline does). Returns ``(counts, entries)`` shaped as the kernel's and
    per-step write-back counts.
    """
    rows_f = np.asarray(rows, np.int32).reshape(-1)
    ids_f = np.asarray(ids, np.int32).reshape(-1)
    gpt, k = plan.gpt, plan.tiles_per_step
    nsizes = len(sg.run_sizes(max_run))
    counts = np.full(plan.total * LANES, -3, np.int32)
    entries = np.full(rows_f.size, -3, np.int32)
    writes = np.zeros(plan.steps, np.int32)
    ops = _NumpyOps()
    for worker in range(plan.workers):
        scratch = tuple(
            np.full(n, -5, np.int32) for n in sg.table_scratch_sizes(gpt, nsizes)
        )
        counts_b = np.full(k * LANES, -9, np.int32)
        entries_b = np.full(k * gpt, -9, np.int32)
        first, count = (int(x) for x in sg.worker_steps(worker, plan))
        for i in range(count):
            step = first + i
            ranks = slice(step * k * gpt, (step + 1) * k * gpt)
            rows_b, ids_b = rows_f[ranks].copy(), ids_f[ranks].copy()
            for t in range(k):
                sg.run_table_tile(
                    ops, rows_b, ids_b, counts_b, entries_b, t, scratch,
                    gpt=gpt, ratio=RATIO, max_run=max_run, even_carry=even_carry,
                    copy_unroll=copy_unroll,
                )
            entries[ranks] = entries_b
            counts[step * k * LANES : (step + 1) * k * LANES] = counts_b
            writes[step] += 1
    counts = counts.reshape(plan.total, LANES)[:, :nsizes]
    return (
        counts.reshape(plan.ngroups, plan.tiles, nsizes),
        entries.reshape(plan.ngroups, -1),
        writes,
    )


def _tables_plan_for(rows, kv_tile, *, workers=WORKERS, tiles_per_step=None):
    gpt = kv_tile // RATIO
    ngroups, cap_ranks = rows.shape
    return sg.table_plan(
        ngroups, cap_ranks // gpt, gpt, workers=workers, tiles_per_step=tiles_per_step
    )


def _table_edge_case(case, kv_tile):
    """(rows, ids): 2 query groups x 3 tiles of hand-built runs (any rows/ids are valid input)."""
    gpt = kv_tile // RATIO
    n = 3 * gpt
    rng = np.random.default_rng(TABLE_EDGE_CASES.index(case) + kv_tile)
    pad_row = 3200 * 5
    rows = np.full((2, n), pad_row, np.int32)
    ids = np.full((2, n), -1, np.int32)

    def runs(lengths, start_row, gap):
        """Live runs of the given lengths, ``gap`` groups apart in the pool."""
        out, row = [], start_row
        for length in lengths:
            out.extend(row + RATIO * np.arange(length))
            row = out[-1] + RATIO * (1 + gap)
        return np.asarray(out[:n], np.int32)

    if case == "one_run":  # every rank live and contiguous: cut only at tile boundaries
        rows[:] = 7 * 3200 + RATIO * np.arange(n)
        ids[:] = np.arange(n)
    elif case == "vector_edges":  # run ends on both sides of the 16-lane boundaries
        lengths = [15, 1, 16, 17, 31, 33, 8, 7, 9, 64, 3, 5, 2, 47, 49, 10, 6, 11, 12, 13] * 8
        rows[0] = runs(lengths, 3 * 3200, 1)
        rows[1] = runs(lengths[::-1], 9 * 3200, 3)
        ids[:] = np.arange(n)
    elif case == "padding_contiguity":  # contiguous rows across a live/padding edge never join
        live0 = gpt + 37
        rows[0, :live0] = pad_row - RATIO * live0 + RATIO * np.arange(live0)
        ids[0, :live0] = np.arange(live0)
        live1 = slice(50, n - 20)
        rows[1, live1] = pad_row + RATIO * (1 + np.arange(n - 70))
        ids[1, live1] = np.arange(n - 70)
    elif case == "decreasing":  # no contiguity at all: every live run has length 1
        rows[:] = 20 * 3200 - RATIO * np.arange(n)
        ids[:] = np.arange(n)
    elif case == "mixed":  # random runs, gaps and padding stretches
        for g in range(2):
            j = 0
            while j < n:
                length = int(rng.integers(1, 41))
                if rng.random() < 0.25:
                    j += length  # a padding stretch
                    continue
                row = int(rng.integers(0, 4000)) * RATIO
                stop = min(n, j + length)
                rows[g, j:stop] = row + RATIO * np.arange(stop - j)
                ids[g, j:stop] = j + np.arange(stop - j)
                j = stop
    return jnp.asarray(rows), jnp.asarray(ids)


def _plan_tables_inputs(seed, group, kv_tile, **plan_kwargs):
    """``union_bits_plan`` inputs and the list reference's ``(rows, ids)``."""
    q, pool, table_row, cap, plan = _plan_inputs(seed, group, **plan_kwargs)
    bits, ubits, usize, _ = plan
    rows, ids = _reference_list(
        ubits, usize, table_row, cap=cap, kv_tile=kv_tile, page_size=3200
    )
    return q, pool, table_row, cap, plan, rows, ids


@pytest.fixture
def plan_kernels(monkeypatch):
    """``union_bits_plan`` off TPU: its Pallas kernel in interpret mode (as test_qwen4_exp_union_gather)."""
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(pl, "pallas_call", interpreted)
    yield
    jax.clear_caches()


def _xla_tables(rows, ids, kv_tile, max_run):
    counts, entries = ud.union_run_tables_xla(
        rows, ids, ratio=RATIO, kv_tile=kv_tile, max_run=max_run
    )
    return np.asarray(counts), np.asarray(entries)


def _assert_tables_equal(got, want):
    np.testing.assert_array_equal(np.asarray(got[0]), want[0])
    np.testing.assert_array_equal(np.asarray(got[1]), want[1])



# CPU tests.


@pytest.mark.parametrize(
    "union_cap,kv_tile",
    [(96, 512), (300, 512), (1000, 1024), (10624, 1024), (200, 256)],
)
def test_cap_ranks_is_the_references_padded_length(union_cap, kv_tile):
    rows, _ = _reference_list(
        jnp.zeros((2, 8), jnp.int32),
        jnp.zeros((2,), jnp.int32),
        jnp.arange(4, dtype=jnp.int32),
        cap=union_cap,
        kv_tile=kv_tile,
    )
    assert sg.cap_ranks_for(union_cap, RATIO, kv_tile) == rows.shape[1]


def _list_case(case):
    """(ubits, usize, page_rows, cap_ranks, cap, kv_tile) with usize/cap edge cases."""
    _, ubits, usize, page_rows, _ = _synthetic(
        LIST_CASES.index(case) + 1,
        ngroups=5,
        words=24,
        per_group_rows=case == "per_group_rows",
    )
    cap = int(usize.max())
    if case == "short":  # usize below the popcount: the tail ranks are padding
        usize = usize.at[1].add(-37).at[3].set(5)
    if case == "long":  # usize above the popcount: ids run out first
        usize = usize.at[2].add(11)
    if case == "truncated":  # popcount past the cap: keep the first cap_ranks ids
        cap = int(usize.min()) // 2
    if case == "empty":
        ubits = ubits.at[2].set(0)
        usize = usize.at[2].set(0)
    kv_tile = 256
    cap_ranks = sg.cap_ranks_for(cap, RATIO, kv_tile)
    return ubits, usize, page_rows, cap_ranks, cap, kv_tile


@pytest.mark.parametrize("case", LIST_CASES)
def test_list_emulation_is_the_reference(case):
    ubits, usize, page_rows, cap_ranks, cap, kv_tile = _list_case(case)
    want_rows, want_ids = _reference_list(
        ubits, usize, page_rows, cap=cap, kv_tile=kv_tile
    )
    physw = _physw(ubits, page_rows, 256)
    got_rows, got_ids = _emulate_list(ubits, usize, physw, cap_ranks=cap_ranks)
    np.testing.assert_array_equal(got_ids, np.asarray(want_ids))
    np.testing.assert_array_equal(got_rows, np.asarray(want_rows))
    if case == "truncated":
        assert (np.asarray(want_ids) >= 0).all()  # every rank live: the clamp ran


def test_line_of_is_the_references_shift_and_mask():
    rng = np.random.default_rng(0)
    rows = rng.integers(-(2**31), 2**31, size=(64, 128), dtype=np.int64)
    rows = rows.astype(np.int32)
    rows[0] = -1  # every bit set, sign bit included
    ids = rng.integers(-1, 4096, size=(64,), dtype=np.int32)
    ids[:4] = [-1, 31, 32, 63]
    got = np.asarray(sg.line_of(jnp.asarray(rows), jnp.asarray(ids)[:, None]))
    shift = (np.maximum(ids, 0) % 32)[:, None].astype(np.uint32)
    want = np.where(ids[:, None] >= 0, (rows.view(np.uint32) >> shift) & 1, 0)
    np.testing.assert_array_equal(got, want.astype(np.int32))


@pytest.mark.parametrize(
    "ngroups,cap_ranks,streams",
    [
        (50, 10752, 2),
        (50, 10752, 1),
        (3, 384, 2),
        (7, 192, 1),
        (33, 1024, 2),
        (1, 64, 2),
        (64, 12544, 2),
    ],
)
def test_gather_plan_covers_every_rank_once(ngroups, cap_ranks, streams):
    plan = sg.gather_plan(
        ngroups, cap_ranks, workers=WORKERS, lanes=LANES, streams=streams
    )
    assert plan.parts & (plan.parts - 1) == 0
    assert plan.parts * plan.part_ranks == cap_ranks
    assert plan.part_ranks % plan.chunk == 0
    hits = np.zeros(ngroups * cap_ranks, np.int32)
    for worker in range(WORKERS):
        for k in range(plan.items_per_worker):
            item = int(sg.worker_item(worker, k, plan))
            if item >= plan.items:
                continue
            g, part, rank0 = (int(x) for x in sg.item_coords(jnp.int32(item), plan))
            assert rank0 == g * cap_ranks + part * plan.part_ranks
            for step in range(plan.steps_per_part):
                for s in range(streams):
                    start = int(sg.window_rank(rank0, step, s, plan))
                    hits[start : start + LANES] += 1
    np.testing.assert_array_equal(hits, 1)
    if (ngroups, cap_ranks, streams) == (50, 10752, 2):
        shape = (plan.parts, plan.items_per_worker, plan.steps_per_part)
        assert shape == (16, 25, 21)


@pytest.mark.parametrize("line_matmul,streams,per_group_rows", GATHER_CASES)
def test_gather_emulation_is_the_reference(line_matmul, streams, per_group_rows):
    bits, ubits, usize, page_rows, pool = _synthetic(
        10 + int(line_matmul), ngroups=5, words=24, per_group_rows=per_group_rows
    )
    cap, kv_tile = int(usize.max()), 256
    rows, ids = _reference_list(ubits, usize, page_rows, cap=cap, kv_tile=kv_tile)
    want_kv, want_lines = ud.union_gather_reference(
        pool, bits, rows, ids, ratio=RATIO, line_matmul=line_matmul
    )
    plan = _gather_plan_for(pool, bits, rows.shape[1], line_matmul, streams)
    if streams is None:  # the VMEM budget picks 1 stream for the 4x lines
        assert plan.streams == (2 if line_matmul else 1)
    kv, lines, kv_writes, line_writes = _emulate_gather(
        pool, bits, rows, ids, line_matmul=line_matmul, plan=plan
    )
    np.testing.assert_array_equal(kv_writes, 1)
    np.testing.assert_array_equal(line_writes, 1)
    np.testing.assert_array_equal(kv.reshape(-1), _u8(want_kv).reshape(-1))
    np.testing.assert_array_equal(lines, np.asarray(want_lines))
    # Both live lines and padding ranks are covered.
    assert np.asarray(want_lines).any() and (np.asarray(ids) < 0).any()


@pytest.mark.parametrize("kv_tile", [256, 512, 1024])
def test_live_rank_end_is_the_attention_kernels_tile_count(kv_tile):
    union_cap = 10624
    cap_ranks = sg.cap_ranks_for(union_cap, RATIO, kv_tile)
    usize = [0, 1, 63, 64, 65, 255, 256, 257, 5000, union_cap - 1, union_cap]
    usize = np.array(usize + [union_cap + 1, 20000], np.int32)
    read = _read_ranks(usize, cap=union_cap, kv_tile=kv_tile, cap_ranks=cap_ranks)
    want = read.reshape(len(usize), cap_ranks).sum(1)
    got = sg.live_rank_end(
        jnp.asarray(usize), cap_ranks=cap_ranks, ratio=RATIO, kv_tile=kv_tile
    )
    np.testing.assert_array_equal(np.asarray(got), want)


def _dead_tile_case():
    bits, ubits, usize, page_rows, pool = _synthetic(21, ngroups=6, words=24)
    kv_tile = 256
    cap = int(usize.max())
    usize = usize.at[1].set(3).at[2].set(0).at[4].set(cap + 500)  # 1 tile; 0; past cap
    rows, ids = _reference_list(ubits, usize, page_rows, cap=cap, kv_tile=kv_tile)
    want_kv, want_lines = ud.union_gather_reference(
        pool, bits, rows, ids, ratio=RATIO, line_matmul=True
    )
    read = _read_ranks(usize, cap=cap, kv_tile=kv_tile, cap_ranks=rows.shape[1])
    return pool, bits, rows, ids, usize, kv_tile, want_kv, want_lines, read


def test_dead_tile_skip_writes_exactly_the_tiles_the_attention_kernel_reads():
    pool, bits, rows, ids, usize, kv_tile, want_kv, want_lines, read = _dead_tile_case()
    plan = _gather_plan_for(pool, bits, rows.shape[1], True)
    steps = sg.item_live_steps(usize, plan, ratio=RATIO, kv_tile=kv_tile)
    kv, lines, kv_writes, _ = _emulate_gather(
        pool, bits, rows, ids, line_matmul=True, plan=plan, item_steps=steps
    )
    assert read.any() and not read.all()
    np.testing.assert_array_equal(kv_writes, read.astype(np.int32))
    want = _u8(want_kv).reshape(read.size, -1)
    np.testing.assert_array_equal(kv.reshape(read.size, -1)[read], want[read])
    np.testing.assert_array_equal(lines[read], np.asarray(want_lines)[read])


@pytest.mark.parametrize("group,kv_tile,max_run", TABLE_CASES)
def test_tables_emulation_is_the_xla_tables(plan_kernels, group, kv_tile, max_run):
    """``run_table_tile`` on ``union_bits_plan`` lists, through the kernel's work split."""
    *_, rows, ids = _plan_tables_inputs(group + kv_tile + max_run, group, kv_tile)
    want = _xla_tables(rows, ids, kv_tile, max_run)
    plan = _tables_plan_for(rows, kv_tile)
    counts, entries, writes = _emulate_tables(rows, ids, max_run=max_run, plan=plan)
    np.testing.assert_array_equal(writes, 1)
    _assert_tables_equal((counts, entries), want)
    # Live runs coalesced and padding present: the inputs exercise both.
    assert (np.asarray(ids) < 0).any() and want[0][..., 0].sum() > 0


@pytest.mark.parametrize("case", ["vector_edges", "padding_contiguity", "mixed"])
def test_tables_emulation_copy_unroll_1(case):
    """The default copy/fill unroll (1 since i1u) keeps the XLA tables in the emulation on the R4 edge cases."""
    rows, ids = _table_edge_case(case, 1024)
    want = _xla_tables(rows, ids, 1024, 4)
    plan = _tables_plan_for(rows, 1024, workers=3, tiles_per_step=1)
    counts, entries, writes = _emulate_tables(rows, ids, max_run=4, plan=plan, copy_unroll=1)
    np.testing.assert_array_equal(writes, 1)
    _assert_tables_equal((counts, entries), want)


def test_carry_pad_makes_odd_carries_even():
    """One dummy count for an odd carry above 1 (R4: 3 sizes, R16: 5), none otherwise or with even_carry=False."""
    assert [sg.carry_pad(len(sg.run_sizes(r))) for r in (1, 2, 4, 8, 16, 32)] == [0, 0, 1, 0, 1, 0]
    assert [sg.carry_pad(len(sg.run_sizes(r)), even_carry=False) for r in (4, 16)] == [0, 0]


@pytest.mark.parametrize("even_carry", [True, False])
@pytest.mark.parametrize("case", ["vector_edges", "padding_contiguity", "mixed"])
@pytest.mark.parametrize("max_run", [4, 16])
def test_tables_emulation_padded_carry_keeps_the_tables(case, max_run, even_carry):
    """The i1o R4 failures' mixed-size tiles (and R16's): the padded carry and the old one both give the XLA tables in
    the emulation (the TPU failure is a lowering question, settled on TPU); the dummy never reaches the counts."""
    rows, ids = _table_edge_case(case, 1024)
    want = _xla_tables(rows, ids, 1024, max_run)
    plan = _tables_plan_for(rows, 1024, workers=3, tiles_per_step=1)
    counts, entries, writes = _emulate_tables(rows, ids, max_run=max_run, plan=plan, even_carry=even_carry)
    np.testing.assert_array_equal(writes, 1)
    _assert_tables_equal((counts, entries), want)
    assert (np.asarray(want[0]) > 0).sum(axis=(0, 1)).min() > 0 or max_run == 16  # every size used (R4)


@pytest.mark.parametrize("case", TABLE_EDGE_CASES)
@pytest.mark.parametrize(
    "kv_tile,max_run,workers,tiles_per_step",
    [(512, 8, WORKERS, None), (1024, 4, 3, 1), (512, 32, 2, 3), (1024, 1, WORKERS, None)],
)
def test_tables_emulation_edge_cases(case, kv_tile, max_run, workers, tiles_per_step):
    """Tile and vector boundaries, live/padding edges, no runs, one run; multi-step workers."""
    rows, ids = _table_edge_case(case, kv_tile)
    want = _xla_tables(rows, ids, kv_tile, max_run)
    plan = _tables_plan_for(rows, kv_tile, workers=workers, tiles_per_step=tiles_per_step)
    counts, entries, writes = _emulate_tables(rows, ids, max_run=max_run, plan=plan)
    np.testing.assert_array_equal(writes, 1)
    _assert_tables_equal((counts, entries), want)


def test_tables_edge_cases_hit_what_they_claim():
    """The edge inputs really contain what their names say (else the tests above prove less)."""
    gpt = 128
    rows, ids = (np.asarray(x) for x in _table_edge_case("one_run", 512))
    counts, _ = _xla_tables(rows, ids, 512, 8)
    np.testing.assert_array_equal(counts[..., 0], gpt // 8)  # one run per tile, cut at the boundary
    rows, ids = (np.asarray(x) for x in _table_edge_case("padding_contiguity", 512))
    edge = np.flatnonzero(np.diff((ids[0] >= 0).astype(int)))[0]
    assert rows[0, edge + 1] == rows[0, edge] + RATIO  # contiguous rows, live then padding
    edge = np.flatnonzero(np.diff((ids[1] >= 0).astype(int)))[0]
    assert rows[1, edge + 1] == rows[1, edge] + RATIO  # contiguous rows, padding then live
    counts, _ = _xla_tables(*_table_edge_case("decreasing", 512), 512, 8)
    assert counts[..., :-1].sum() == 0  # pieces of 1 only


@pytest.mark.parametrize(
    "ngroups,tiles,workers",
    [(50, 42, 32), (100, 42, 32), (50, 21, 32), (2, 3, 32), (1, 41, 32), (7, 5, 3)],
)
def test_table_plan_covers_every_tile_once(ngroups, tiles, workers):
    plan = sg.table_plan(ngroups, tiles, 256, workers=workers)
    assert plan.total % plan.tiles_per_step == 0
    hits = np.zeros(plan.total, np.int32)
    for worker in range(workers):
        first, count = (int(x) for x in sg.worker_steps(worker, plan))
        for step in range(first, first + count):
            hits[step * plan.tiles_per_step : (step + 1) * plan.tiles_per_step] += 1
    np.testing.assert_array_equal(hits, 1)
    for gpt in (128, 256, 512):
        need = sg.table_vmem_bytes(gpt, plan.tiles_per_step, 6)
        assert need <= sg._GATHER_VMEM_BUDGET


def test_run_table_args_are_checked():
    for bad in (dict(max_run=3), dict(max_run=64), dict(kv_tile=4096)):
        args = dict(ratio=RATIO, kv_tile=1024, max_run=8) | bad
        with pytest.raises(ValueError):
            sg.check_run_table_args((2, 2048), (2, 2048), **args)
    with pytest.raises(ValueError, match="whole tiles"):
        sg.check_run_table_args((2, 300), (2, 300), ratio=RATIO, kv_tile=1024, max_run=8)
    assert sg.check_run_table_args((2, 768), (2, 768), ratio=RATIO, kv_tile=1024, max_run=8) == (256, 3)


def _lines_plan_for(bits, cap_ranks, line_matmul, streams=None):
    return sg.choose_gather_plan(
        bits.shape[0],
        cap_ranks,
        workers=WORKERS,
        group_bytes=0,
        width=bits.shape[2],
        repeat=1 if line_matmul else RATIO,
        streams=streams,
    )


@pytest.mark.parametrize("line_matmul", [True, False])
@pytest.mark.parametrize("skip_dead", [False, True])
def test_lines_emulation_is_the_reference(line_matmul, skip_dead):
    """``union_sc_lines``: the gather's line path alone (lines-only plan, optional dead-tile skip)."""
    pool, bits, rows, ids, usize, kv_tile, _, _, read = _dead_tile_case()
    want = np.asarray(ud.union_lines_reference(bits, ids, ratio=RATIO, line_matmul=line_matmul))
    plan = _lines_plan_for(bits, ids.shape[1], line_matmul)
    assert plan.streams == 2  # no KV buffers: two streams fit for both line forms
    steps = sg.item_live_steps(usize, plan, ratio=RATIO, kv_tile=kv_tile) if skip_dead else None
    kv, lines, _, line_writes = _emulate_gather(
        None, bits, None, ids, line_matmul=line_matmul, plan=plan, item_steps=steps
    )
    assert kv is None
    repeat = 1 if line_matmul else RATIO
    read = np.repeat(read, repeat) if skip_dead else np.ones(want.shape[0], bool)
    np.testing.assert_array_equal(line_writes, read.astype(np.int32))
    np.testing.assert_array_equal(lines[read], want[read])
    assert want[read].any()


@pytest.fixture
def tpu7x_abstract():
    """Trace and lower for a TPU7x device without one (``use_abstract_mesh``)."""
    from jax.sharding import AbstractDevice, AbstractMesh, AxisType, use_abstract_mesh

    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    mesh = AbstractMesh((1,), ("x",), (AxisType.Explicit,), abstract_device=device)
    with use_abstract_mesh(mesh):
        yield
    jax.clear_caches()


@pytest.mark.parametrize(
    "variant",
    [
        "list",
        "list_shared",
        "gather_mm",
        "gather",
        "gather_mm_live",
        "gather_s1",
        "lines_mm",
        "lines",
        "lines_mm_live",
        "tables_g256_r8",
        "tables_g128_r4",
        "tables_g256_r4",
        "tables_g256_r16",
        "tables_g256_r4_cu2",
        "tables_g512_r32",
    ],
)
def test_kernels_lower_to_mosaic_for_tpu7x(tpu7x_abstract, variant):
    """Pallas -> Mosaic lowering at the realistic size (no SC compile, no run)."""
    ngroups, words, cap_ranks = 50, 784, 10752
    spec = jax.ShapeDtypeStruct
    if variant.startswith("tables"):
        gpt, max_run = (int(x[1:]) for x in variant.split("_")[1:3])
        unroll = 2 if variant.endswith("_cu2") else 1
        fn = jax.jit(
            functools.partial(
                sg.union_sc_run_tables, ratio=RATIO, kv_tile=gpt * RATIO, max_run=max_run,
                copy_unroll=unroll,
            )
        )
        args = (spec((ngroups, cap_ranks), jnp.int32),) * 2
    elif variant.startswith("lines"):
        live = variant.endswith("live")

        def lines(bits, ids, usize):
            return sg.union_sc_lines(
                bits,
                ids,
                ratio=RATIO,
                line_matmul="mm" in variant,
                usize=usize if live else None,
                kv_tile=1024 if live else None,
            )

        fn = jax.jit(lines)
        args = (
            spec((ngroups, words, 128), jnp.int32),
            spec((ngroups, cap_ranks), jnp.int32),
            spec((ngroups,), jnp.int32),
        )
    elif variant.startswith("list"):
        shared = variant == "list_shared"
        physw = spec((words,) if shared else (ngroups, words), jnp.int32)
        fn = jax.jit(functools.partial(sg.union_sc_list, cap_ranks=cap_ranks))
        args = (spec((ngroups, words), jnp.int32), spec((ngroups,), jnp.int32), physw)
    else:
        live = variant.endswith("live")

        def gather(pool, bits, rows, ids, usize):
            return sg.union_sc_gather(
                pool,
                bits,
                rows,
                ids,
                ratio=RATIO,
                line_matmul="mm" in variant,
                usize=usize if live else None,
                kv_tile=1024 if live else None,
                streams=1 if variant == "gather_s1" else None,
            )

        fn = jax.jit(gather)
        args = (
            spec((32, 3200, 1, 4, 256), jnp.float8_e4m3fn),
            spec((ngroups, words, 128), jnp.int32),
            spec((ngroups, cap_ranks), jnp.int32),
            spec((ngroups, cap_ranks), jnp.int32),
            spec((ngroups,), jnp.int32),
        )
    lowered = fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "tpu_custom_call" in lowered


def test_copy_unroll_reaches_the_table_kernel(tpu7x_abstract, monkeypatch):
    """copy_unroll reaches the tile code inside the SparseCore kernel (recorded while it traces), and the default is 1
    (i1u, M: 2 corrupts R4 tables on TPU)."""
    seen = []
    original = sg.run_table_tile

    def recording(*args, **kwargs):
        seen.append(kwargs.get("copy_unroll"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sg, "run_table_tile", recording)
    spec = jax.ShapeDtypeStruct
    args = (spec((50, 10752), jnp.int32),) * 2
    for unroll in (1, 2, None):
        seen.clear()
        kw = {} if unroll is None else {"copy_unroll": unroll}
        fn = jax.jit(functools.partial(sg.union_sc_run_tables, ratio=RATIO, kv_tile=1024, max_run=4, **kw))
        fn.trace(*args).lower(lowering_platforms=("tpu",))
        assert seen and set(seen) == {1 if unroll is None else unroll}, (unroll, seen)


# TPU tests: the real SparseCore kernels.


@tpu_only
@pytest.mark.parametrize("case", LIST_CASES)
def test_sc_list_is_the_reference(case):
    ubits, usize, page_rows, cap_ranks, cap, kv_tile = _list_case(case)
    want_rows, want_ids = _reference_list(
        ubits, usize, page_rows, cap=cap, kv_tile=kv_tile
    )
    rows, ids = sg.union_sc_list_pages(
        ubits,
        usize,
        page_rows,
        ratio=RATIO,
        page_size=256,
        union_cap=cap,
        kv_tile=kv_tile,
    )
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(want_ids))
    np.testing.assert_array_equal(np.asarray(rows), np.asarray(want_rows))


@tpu_only
@pytest.mark.parametrize("line_matmul,streams,per_group_rows", GATHER_CASES)
def test_sc_gather_is_the_reference(line_matmul, streams, per_group_rows):
    bits, ubits, usize, page_rows, pool = _synthetic(
        10 + int(line_matmul), ngroups=5, words=24, per_group_rows=per_group_rows
    )
    cap, kv_tile = int(usize.max()), 256
    rows, ids = _reference_list(ubits, usize, page_rows, cap=cap, kv_tile=kv_tile)
    want_kv, want_lines = ud.union_gather_reference(
        pool, bits, rows, ids, ratio=RATIO, line_matmul=line_matmul
    )
    got_kv, got_lines = sg.union_sc_gather(
        pool, bits, rows, ids, ratio=RATIO, line_matmul=line_matmul, streams=streams
    )
    np.testing.assert_array_equal(_u8(got_kv), _u8(want_kv))
    np.testing.assert_array_equal(np.asarray(got_lines), np.asarray(want_lines))


@tpu_only
def test_sc_gather_dead_tile_skip_keeps_every_read_tile():
    pool, bits, rows, ids, usize, kv_tile, want_kv, want_lines, read = _dead_tile_case()
    got_kv, got_lines = sg.union_sc_gather(
        pool,
        bits,
        rows,
        ids,
        ratio=RATIO,
        line_matmul=True,
        usize=usize,
        kv_tile=kv_tile,
    )
    got = _u8(got_kv).reshape(read.size, -1)
    want = _u8(want_kv).reshape(read.size, -1)
    np.testing.assert_array_equal(got[read], want[read])
    got_lines, want_lines = np.asarray(got_lines), np.asarray(want_lines)
    np.testing.assert_array_equal(got_lines[read], want_lines[read])


def _device_mismatches(got, want):
    """Mismatching elements, counted on device (the realistic arrays are GBs)."""
    if got.dtype.itemsize == 1:
        got = jax.lax.bitcast_convert_type(got, jnp.uint8)
        want = jax.lax.bitcast_convert_type(want, jnp.uint8)
    return int(jnp.sum(got != want))


@tpu_only
@pytest.mark.parametrize("skip_dead", [False, True])
def test_sc_list_and_gather_at_the_realistic_size(skip_dead):
    """50 query groups, 784 words, usize about 10K (one past the cap), kv_tile 1024."""
    page_size, kv_tile, union_cap = 3200, 1024, 10624
    bits, ubits, usize, page_rows, pool = _synthetic(
        31,
        ngroups=50,
        words=784,
        group=128,
        density=0.41,
        page_size=page_size,
        per_group_rows=True,
        pool_pages=40,
        dense_group=7,
    )
    median = int(np.median(np.asarray(usize)))
    assert int(usize.max()) > union_cap > median > 9000
    want_rows, want_ids = _reference_list(
        ubits, usize, page_rows, cap=union_cap, kv_tile=kv_tile, page_size=page_size
    )
    rows, ids = sg.union_sc_list_pages(
        ubits,
        usize,
        page_rows,
        ratio=RATIO,
        page_size=page_size,
        union_cap=union_cap,
        kv_tile=kv_tile,
    )
    assert rows.shape == (50, 10752)
    assert _device_mismatches(ids, want_ids) == 0
    assert _device_mismatches(rows, want_rows) == 0
    want_kv, want_lines = ud.union_gather_reference(
        pool, bits, want_rows, want_ids, ratio=RATIO, line_matmul=True
    )
    got_kv, got_lines = sg.union_sc_gather(
        pool,
        bits,
        rows,
        ids,
        ratio=RATIO,
        line_matmul=True,
        usize=usize if skip_dead else None,
        kv_tile=kv_tile if skip_dead else None,
    )
    if skip_dead:
        read = _read_ranks(usize, cap=union_cap, kv_tile=kv_tile, cap_ranks=10752)
        assert not read.all()
        read = jnp.asarray(read)
        kv_read = jnp.repeat(read, RATIO)[:, None, None, None]
        got_kv = jnp.where(kv_read, got_kv, 0)
        want_kv = jnp.where(kv_read, want_kv, 0)
        got_lines = jnp.where(read[:, None], got_lines, 0)
        want_lines = jnp.where(read[:, None], want_lines, 0)
    assert _device_mismatches(got_kv, want_kv) == 0
    assert _device_mismatches(got_lines, want_lines) == 0


def _plan_inputs(
    seed, group, page=3200, tokens=256, context=6000, topk=512, pages=3, picks=40
):
    """As test_qwen4_exp_union_gather: overlapping selections on a permuted table.

    ``pages``: pool pages (the table is a permutation of them); ``picks``:
    random groups per token on top of the recent window and the shared spans.
    """
    from vllm_torchtpu.kernels.qwen4_exp import qsa_union_prefill as up
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    rng = np.random.default_rng(seed)
    lengths = (context + np.arange(1, tokens + 1)).astype(np.int32)
    lengths[-3:] = 0
    blocks = np.full((tokens, topk), -1, np.int32)
    spans = rng.integers(0, context // RATIO - 64, size=64)
    for i, length in enumerate(lengths):
        complete = int(length) // RATIO
        if not complete:
            continue
        chosen = set(range(max(0, complete - 48), complete))
        for s in spans[(i // 8) % 56 : (i // 8) % 56 + 8]:
            chosen |= set(range(int(s), min(int(s) + 12, complete)))
        chosen |= set(rng.integers(0, complete, size=picks).tolist())
        pick = rng.permutation(sorted(chosen))[:topk]
        blocks[i, : len(pick)] = pick
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pages, page, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    table_row = jnp.asarray(rng.permutation(pages), jnp.int32)
    blocks, lengths = jnp.asarray(blocks), jnp.asarray(lengths)
    cap = int(-(-up.union_sizes(blocks, lengths, group=group).max() // 32) * 32)
    plan = union_bits_plan(
        blocks, lengths, group=group, union_cap=cap, groups_cap=pages * page // RATIO
    )
    return q, pool, table_row, cap, plan


@tpu_only
@pytest.mark.parametrize(
    "group,kv_tile,line_matmul,per_group_rows",
    [(64, 512, False, False), (128, 1024, True, False), (32, 256, False, True)],
)
def test_sc_gathered_attention_is_the_walk(group, kv_tile, line_matmul, per_group_rows):
    """End to end on ``union_bits_plan`` inputs, with the dead-tile skip on."""
    page = 3200
    q, pool, table_row, cap, plan = _plan_inputs(group + kv_tile, group, page=page)
    bits, ubits, usize, _ = plan
    if per_group_rows:
        rows_arg = jnp.broadcast_to(table_row, (ubits.shape[0], 3))
    else:
        rows_arg = table_row
    options = dict(
        group=group,
        union_cap=cap,
        kv_tile=kv_tile,
        lean=True,
        fast_walk=True,
        line_matmul=line_matmul,
    )
    walk = np.asarray(ud.union_walk_kernel(q, *plan, pool, rows_arg, **options))
    rows, ids = sg.union_sc_list_pages(
        ubits,
        usize,
        rows_arg,
        ratio=RATIO,
        page_size=page,
        union_cap=cap,
        kv_tile=kv_tile,
    )
    gathered = sg.union_sc_gather(
        pool,
        bits,
        rows,
        ids,
        ratio=RATIO,
        line_matmul=line_matmul,
        usize=usize,
        kv_tile=kv_tile,
    )
    got = ud.union_walk_kernel(q, *plan, pool, rows_arg, gathered=gathered, **options)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), walk.view(np.uint16))
    assert np.abs(walk.astype(np.float32)).max() > 0


def test_padding_rows_broadcast_word_zero():
    """The padding row is word 0's base on every lane (i1l: the SparseCore's zero-index gather gave physw[lane])."""
    bases = jnp.arange(16, dtype=jnp.int32) * 128 + 3200 * 7
    np.testing.assert_array_equal(np.asarray(sg.padding_rows(bases)), np.full(16, 3200 * 7, np.int32))


# TPU tests: run tables and lines.


@tpu_only
@pytest.mark.parametrize("group,kv_tile,max_run", TABLE_CASES)
def test_sc_run_tables_is_the_xla_tables(group, kv_tile, max_run):
    *_, rows, ids = _plan_tables_inputs(group + kv_tile + max_run, group, kv_tile)
    want = _xla_tables(rows, ids, kv_tile, max_run)
    got = sg.union_sc_run_tables(rows, ids, ratio=RATIO, kv_tile=kv_tile, max_run=max_run)
    _assert_tables_equal(got, want)


@tpu_only
@pytest.mark.parametrize("case", TABLE_EDGE_CASES)
@pytest.mark.parametrize(
    "kv_tile,max_run,tiles_per_step", [(512, 8, None), (1024, 4, 1), (512, 32, 3), (1024, 1, None)]
)
def test_sc_run_tables_edge_cases(case, kv_tile, max_run, tiles_per_step):
    rows, ids = _table_edge_case(case, kv_tile)
    want = _xla_tables(rows, ids, kv_tile, max_run)
    got = sg.union_sc_run_tables(
        rows, ids, ratio=RATIO, kv_tile=kv_tile, max_run=max_run, tiles_per_step=tiles_per_step
    )
    _assert_tables_equal(got, want)


@tpu_only
@pytest.mark.parametrize("line_matmul", [True, False])
@pytest.mark.parametrize("skip_dead", [False, True])
def test_sc_lines_is_the_reference(line_matmul, skip_dead):
    pool, bits, rows, ids, usize, kv_tile, _, _, read = _dead_tile_case()
    want = np.asarray(ud.union_lines_reference(bits, ids, ratio=RATIO, line_matmul=line_matmul))
    got = np.asarray(
        sg.union_sc_lines(
            bits,
            ids,
            ratio=RATIO,
            line_matmul=line_matmul,
            usize=usize if skip_dead else None,
            kv_tile=kv_tile if skip_dead else None,
        )
    )
    if skip_dead:
        read = np.repeat(read, 1 if line_matmul else RATIO)
        got, want = got[read], want[read]
    np.testing.assert_array_equal(got, want)


@functools.lru_cache(maxsize=2)
def _realistic_plan(group):
    """50 query groups, unions near 10K groups over a 21-22 page permuted table."""
    tokens, context, page = 50 * group, 60000, 3200
    pages = -(-(context + tokens) // page) + 1
    return _plan_inputs(
        41 + group,
        group,
        page=page,
        tokens=tokens,
        context=context,
        pages=pages,
        picks=16000 // group,
    )


@tpu_only
@pytest.mark.parametrize("group", [64, 128])
def test_sc_run_tables_and_lines_at_the_realistic_size(group):
    """50 query groups x cap near 10K, kv_tile 1024: tables and both line forms on the SparseCore list."""
    kv_tile = 1024
    q, pool, table_row, cap, plan = _realistic_plan(group)
    bits, ubits, usize, _ = plan
    assert ubits.shape[0] == 50 and int(np.median(np.asarray(usize))) > 8000 and cap < 14000
    want_rows, want_ids = _reference_list(
        ubits, usize, table_row, cap=cap, kv_tile=kv_tile, page_size=3200
    )
    rows, ids = sg.union_sc_list_pages(
        ubits, usize, table_row, ratio=RATIO, page_size=3200, union_cap=cap, kv_tile=kv_tile
    )
    assert _device_mismatches(rows, want_rows) == 0 and _device_mismatches(ids, want_ids) == 0
    want = ud.union_run_tables_xla(want_rows, want_ids, ratio=RATIO, kv_tile=kv_tile)
    got = sg.union_sc_run_tables(rows, ids, ratio=RATIO, kv_tile=kv_tile)
    assert _device_mismatches(got[0], want[0]) == 0
    assert _device_mismatches(got[1], want[1]) == 0
    assert (np.asarray(want[0]).sum(axis=(0, 1)) > 0).all()  # pieces of every size
    for line_matmul in (True, False):
        want_lines = ud.union_lines_reference(bits, want_ids, ratio=RATIO, line_matmul=line_matmul)
        got_lines = sg.union_sc_lines(bits, ids, ratio=RATIO, line_matmul=line_matmul)
        assert _device_mismatches(got_lines, want_lines) == 0
        del want_lines, got_lines


def _run_table_walk(q, pool, table_row, cap, plan, *, group, kv_tile, line_matmul, max_run,
                    skip_dead):
    """The walk, and the run-table walk on the SparseCore list, tables and lines."""
    bits, ubits, usize, _ = plan
    options = dict(
        group=group, union_cap=cap, kv_tile=kv_tile, lean=True, fast_walk=True,
        line_matmul=line_matmul,
    )
    walk = np.asarray(ud.union_walk_kernel(q, *plan, pool, table_row, **options))
    rows, ids = sg.union_sc_list_pages(
        ubits, usize, table_row, ratio=RATIO, page_size=3200, union_cap=cap, kv_tile=kv_tile
    )
    tables = sg.union_sc_run_tables(rows, ids, ratio=RATIO, kv_tile=kv_tile, max_run=max_run)
    lines = sg.union_sc_lines(
        bits,
        ids,
        ratio=RATIO,
        line_matmul=line_matmul,
        usize=usize if skip_dead else None,
        kv_tile=kv_tile if skip_dead else None,
    )
    got = ud.union_walk_kernel(
        q, *plan, pool, table_row, run_tables=tables, listed_lines=lines, **options
    )
    return np.asarray(got), walk


@tpu_only
@pytest.mark.parametrize(
    "group,kv_tile,line_matmul,max_run,skip_dead",
    [(64, 512, False, 8, False), (128, 1024, True, 8, True), (64, 1024, True, 4, False)],
)
def test_sc_run_table_attention_is_the_walk(group, kv_tile, line_matmul, max_run, skip_dead):
    """End to end on ``union_bits_plan`` inputs: SparseCore list -> tables + lines -> run-table walk."""
    q, pool, table_row, cap, plan = _plan_inputs(group + kv_tile + max_run + 23, group)
    got, walk = _run_table_walk(
        q, pool, table_row, cap, plan, group=group, kv_tile=kv_tile,
        line_matmul=line_matmul, max_run=max_run, skip_dead=skip_dead,
    )
    np.testing.assert_array_equal(got.view(np.uint16), walk.view(np.uint16))
    assert np.abs(walk.astype(np.float32)).max() > 0


@tpu_only
@pytest.mark.parametrize("group", [64, 128])
def test_sc_run_table_attention_is_the_walk_at_the_realistic_size(group):
    """End to end at 50 query groups x cap near 10K: kv_tile 1024, R8, matmul lines (i1n's arms), dead-tile skip on."""
    line_matmul = True
    q, pool, table_row, cap, plan = _realistic_plan(group)
    got, walk = _run_table_walk(
        q, pool, table_row, cap, plan, group=group, kv_tile=1024,
        line_matmul=line_matmul, max_run=8, skip_dead=True,
    )
    np.testing.assert_array_equal(got.view(np.uint16), walk.view(np.uint16))
    assert np.abs(walk.astype(np.float32)).max() > 0


def test_padding_bits_rows_spread_over_word_rows():
    """Padding ranks read distinct rows among the group's first 8 words (no single hot row); live ids their word."""
    ids = jnp.asarray([[-1] * 16, [33, 64, -1, 5] * 4], jnp.int32)
    rows = np.asarray(sg.bits_row_index(ids, 1000))
    np.testing.assert_array_equal(rows[0], 1000 + (np.arange(16) & 7))
    live = np.asarray(ids[1]) >= 0
    np.testing.assert_array_equal(rows[1][live], 1000 + (np.asarray(ids[1])[live] >> 5))
    assert len(set(rows[0].tolist())) == 8
