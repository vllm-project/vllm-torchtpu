# SPDX-License-Identifier: Apache-2.0
"""Run-based main KV writer (interpret mode): the same pool as
write_main_cache's tiled kernels and a row-by-row reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_fused_kv_write import interpret, raw  # noqa: F401 (fixture)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import main_write, pooled_qsa

PAGES, PAGE_ROWS, WIDTH = 24, 256, 256

CASES = {
    # Prefill requests misaligned against their contexts and crossing pages,
    # with decode/verify rows between them and padding at the end.
    "mixed": ([300, 4, 4, 4, 200, 1, 3], [3, 253, 0, 255, 1001, 7, 100]),
    "aligned": ([256, 128], [0, 512]),
    "decode_rows": ([4] * 40 + [1] * 8, list(range(0, 960, 20))),
}
LAYOUTS = {
    "fp8_paired_slot0": (jnp.float8_e4m3fn, 0),
    "fp8_paired_slot1": (jnp.float8_e4m3fn, 1),
    "fp8_whole_rows": (jnp.float8_e4m3fn, None),
    "bf16_whole_rows": (jnp.bfloat16, None),
}


def make_rows(rng, tokens, dtype, slot):
    if dtype == jnp.float8_e4m3fn:
        rows = rng.normal(0, 1, (tokens, 1, 4, WIDTH))
        if slot is not None:
            rows[:, :, 2:] = 0  # pack_rows: K, V, 0, 0
        pool = rng.normal(0, 1, (PAGES, PAGE_ROWS, 1, 4, WIDTH))
    else:
        rows = rng.normal(0, 1, (tokens, 1, 2, WIDTH))
        pool = rng.normal(0, 1, (PAGES, PAGE_ROWS, 1, 2, WIDTH))
    return jnp.asarray(pool, dtype), jnp.asarray(rows, dtype)


def destinations(rng, query_lens, contexts, dcp=None):
    """``(page, logical, valid)`` as ``pooled_qsa_step`` builds them."""
    tokens = sum(query_lens)
    padded = -(-tokens // 128) * 128
    size, rank = dcp or (1, 0)
    span = size * PAGE_ROWS
    free = iter(rng.permutation(np.arange(PAGES)))
    page = np.full(padded, PAGES, np.int32)
    logical = np.zeros(padded, np.int32)
    valid = np.zeros(padded, bool)
    i = 0
    for n, ctx in zip(query_lens, contexts):
        table = {}
        for t in range(ctx, ctx + n):
            if t // span not in table:
                table[t // span] = next(free)
            logical[i] = t
            if (t // PAGE_ROWS) % size == rank:
                valid[i] = True
                page[i] = table[t // span]
            i += 1
    return jnp.asarray(page), jnp.asarray(logical), jnp.asarray(valid)


def reference(pool, rows, page, logical, valid, slot):
    out = raw(pool).copy()
    new = raw(rows)
    for i in np.flatnonzero(np.asarray(valid)):
        p, o = int(page[i]), int(logical[i]) % PAGE_ROWS
        if slot is None:
            out[p, o] = new[i]
        else:
            out[p, o, :, 2 * slot : 2 * slot + 2] = new[i, :, :2]
    return out


def written(monkeypatch, run_write, pool, rows, page, logical, valid, slot, tile=128):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_RUN_WRITE", run_write)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS", 0)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE", tile)
    jax.clear_caches()
    kw = {} if slot is None else {"slot": slot}
    # Rows enter as an argument: XLA:CPU constant folding of FP8 bitcasts
    # is not byte-exact.
    write = jax.jit(
        lambda pool, rows: pooled_qsa.write_main_cache(
            pool, rows, page, logical, valid, **kw
        )
    )
    return raw(write(pool, rows))


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("dcp", [None, (2, 0), (2, 1)])
def test_run_writer_matches_tiled_writer(interpret, monkeypatch, case, layout, dcp):
    dtype, slot = LAYOUTS[layout]
    rng = np.random.default_rng(len(case) + 3 * len(layout))
    page, logical, valid = destinations(rng, *CASES[case], dcp)
    pool, rows = make_rows(rng, len(page), dtype, slot)
    expected = reference(pool, rows, page, logical, valid, slot)
    tiled = written(monkeypatch, False, pool, rows, page, logical, valid, slot)
    np.testing.assert_array_equal(tiled, expected)
    for tile in (128, 512):
        got = written(monkeypatch, True, pool, rows, page, logical, valid, slot, tile)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("slot", [None, 1])
def test_repeated_destinations_keep_last_writer(interpret, monkeypatch, slot):
    rng = np.random.default_rng(9)
    tokens = 256
    pool, rows = make_rows(rng, tokens, jnp.float8_e4m3fn, slot)
    # Runs of 5 that restart: offsets 3..7, 3..7, ... on one page, plus a
    # second page whose run crosses the 128-row tile edge.
    offset = np.where(np.arange(tokens) < 128, np.arange(tokens) % 5 + 3, np.arange(tokens))
    page = jnp.asarray(np.where(np.arange(tokens) < 128, 2, 5), jnp.int32)
    logical = jnp.asarray(offset, jnp.int32)
    valid = jnp.asarray(np.arange(tokens) % 37 != 11)
    expected = reference(pool, rows, page, logical, valid, slot)
    for tile in (128, 256):
        got = written(monkeypatch, True, pool, rows, page, logical, valid, slot, tile)
        np.testing.assert_array_equal(got, expected)


def test_runs_split_at_pages_tiles_and_gaps():
    page = jnp.asarray([1] * 6 + [2] * 2, jnp.int32)
    offsets = jnp.asarray([0, 1, 2, 3, 5, 6, 7, 8], jnp.int32)
    valid = jnp.asarray([True] * 7 + [False])
    start, length, first, unique = main_write.main_runs(page, offsets, valid, 4, 16)
    count = int(first[-1])
    assert np.asarray(start)[:count].tolist() == [0, 4, 6]
    assert np.asarray(length)[:count].tolist() == [4, 2, 1]
    assert np.asarray(first).tolist() == [0, 1, 3]
    assert np.asarray(unique).tolist() == [1, 1]


# Served geometry of the Qwen3.8 c32 recipe: paired FP8 main-KV pages of 512
# rows, main-KV DCP 2 (a block-table entry spans 1024 tokens), run-writer
# tile 1024, 24 request slots.
SERVED_ROWS, SERVED_DCP, SERVED_SLOTS = 512, 2, 24


def precompile_layout(tokens, slots=SERVED_SLOTS):
    """``Qwen4ExpAdapter.dummy_run``'s rows (``dummy_query_lengths``): one-token
    requests and a last request holding the rest, all at context 0."""
    active = min(tokens, slots)
    lengths = [1] * active + [0] * (slots - active)
    lengths[active - 1] += tokens - active
    return lengths, [0] * slots


def verify_step_layout(tokens, speculative, slots=SERVED_SLOTS):
    """A served mixed step: ``slots - 1`` verify requests of ``speculative + 1``
    rows at contexts on both sides of the entry midpoint, plus one prefill chunk
    filling the step that starts and ends inside a 1024-token entry."""
    window = speculative + 1
    contexts = [509 + 37 * i for i in range(slots - 1)] + [700]
    return [window] * (slots - 1) + [tokens - window * (slots - 1)], contexts


def served_destinations(query_lens, contexts, rank, pages):
    """``qsa_cache_writes``' ``(page, logical, valid)`` under main-KV DCP, with
    each request owning consecutive table entries from page 1."""
    span = SERVED_DCP * SERVED_ROWS
    page, logical, valid = [], [], []
    first = 1
    for n, ctx in zip(query_lens, contexts):
        for t in range(ctx, ctx + n):
            owned = (t // SERVED_ROWS) % SERVED_DCP == rank
            logical.append(t)
            valid.append(owned)
            page.append(first + t // span if owned else pages)
        first += -(-(ctx + n) // span)
    assert first <= pages
    return (np.asarray(page, np.int32), np.asarray(logical, np.int32), np.asarray(valid))


def run_dmas(page, offsets, valid, tile, rows_per_page):
    """Every DMA ``write_main_runs`` issues, as ``(page, row, vmem_row, rows)``,
    from ``main_runs`` and the kernel's power-of-two split of each run."""
    tokens = len(page)
    padded = -(-tokens // tile) * tile
    pad = (0, padded - tokens)
    start, length, first, _ = main_write.main_runs(
        jnp.asarray(np.pad(page, pad)),
        jnp.asarray(np.pad(offsets, pad)),
        jnp.asarray(np.pad(valid, pad)),
        tile,
        rows_per_page,
    )
    start, length, count = np.asarray(start), np.asarray(length), int(first[-1])
    longest = min(tile, rows_per_page)
    sizes = [1 << b for b in range(longest.bit_length() - 1, -1, -1)]
    dmas = []
    for token, run in zip(start[:count], length[:count]):
        for size in sizes:
            if run & size:
                done = run & ~(2 * size - 1)
                dmas.append(
                    (page[token], offsets[token] + done, token % tile + done, size)
                )
    return dmas


def out_of_bounds(dmas, pages, rows_per_page, tile):
    return [
        d
        for d in dmas
        if not (0 <= d[0] < pages and d[1] + d[3] <= rows_per_page and d[2] + d[3] <= tile)
    ]


SERVED_LAYOUTS = {
    **{f"precompile_{t}": (lambda t=t: precompile_layout(t)) for t in (512, 1024, 2048, 3072, 4096, 6144, 8192)},
    **{
        f"verify_k{k}_{t}": (lambda t=t, k=k: verify_step_layout(t, k))
        for k in (2, 3)
        for t in (512, 1024, 4096, 8192)
    },
}


@pytest.mark.parametrize("tile", [512, 1024])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("layout", sorted(SERVED_LAYOUTS))
def test_served_run_dmas_stay_in_bounds(layout, rank, tile):
    """Each run DMA stays inside its page, its tile's VMEM rows and the pool,
    for the precompile buckets and K=2 / K=3 verify steps (j-4a493b32 halted
    in this kernel at the 4096 bucket with BoundsCheck on dma.hbm_to_vmem)."""
    pages = 64
    query_lens, contexts = SERVED_LAYOUTS[layout]()
    page, logical, valid = served_destinations(query_lens, contexts, rank, pages)
    dmas = run_dmas(page, logical % SERVED_ROWS, valid, tile, SERVED_ROWS)
    assert sum(d[3] for d in dmas) == int(valid.sum())
    assert out_of_bounds(dmas, pages, SERVED_ROWS, tile) == []
    if not valid.any():
        return  # this rank owns none of the step's rows
    # Positive control: one valid row sent past the pool is caught.
    bad = page.copy()
    bad[np.flatnonzero(valid)[-1]] = pages
    assert out_of_bounds(run_dmas(bad, logical % SERVED_ROWS, valid, tile, SERVED_ROWS), pages, SERVED_ROWS, tile)


@pytest.mark.parametrize("layout", ["precompile_4096", "verify_k2_4096", "verify_k3_4096"])
@pytest.mark.parametrize("rank", [0, 1])
def test_served_run_writer_checked_interpreter(monkeypatch, layout, rank):
    """The kernel itself on the TPU interpreter that raises on out-of-bounds
    DMAs (whole rows: the paired path issues the same DMAs plus a merge the
    checked interpreter cannot discharge), against the row-by-row reference."""
    if not hasattr(pltpu, "InterpretParams"):
        pytest.skip("jax without the checked TPU interpreter")
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *a, **k: original(*a, **{**k, "interpret": pltpu.InterpretParams()}),
    )
    pages, width = 48, 256
    query_lens, contexts = SERVED_LAYOUTS[layout]()
    page, logical, valid = served_destinations(query_lens, contexts, rank, pages)
    rng = np.random.default_rng(rank)
    pool = jnp.asarray(
        rng.normal(0, 1, (pages, SERVED_ROWS, 1, 4, width)), jnp.float8_e4m3fn
    )
    rows = jnp.asarray(rng.normal(0, 1, (len(page), 1, 4, width)), jnp.float8_e4m3fn)
    got = main_write.write_main_runs(
        pool,
        rows,
        jnp.asarray(page),
        jnp.asarray(logical % SERVED_ROWS),
        jnp.asarray(valid),
        tile=1024,
    )
    expected = raw(pool).copy()
    new = raw(rows)
    for i in np.flatnonzero(valid):
        expected[page[i], logical[i] % SERVED_ROWS] = new[i]
    np.testing.assert_array_equal(raw(got), expected)


def recorded_dmas(monkeypatch):
    """Patch the run writer's DMAs to log ``(to_hbm, page, row, vmem_row, rows,
    pool_pages, page_rows, vmem_rows)`` for every copy the kernel issues,
    paired reads included (the plain interpreter clamps out-of-range slices
    instead of raising, so the bounds are checked on the log)."""
    log = []
    original = pltpu.make_async_copy

    def indices(ref):
        (indexer,) = ref.transforms
        return indexer.indices, ref.ref.shape

    def record(source, dest, semaphore):
        (s_idx, s_shape), (d_idx, d_shape) = indices(source), indices(dest)
        if 5 not in (len(s_shape), len(d_shape)):
            return original(source, dest, semaphore)  # a wait descriptor
        to_hbm = len(d_shape) == 5
        h_idx, h_shape, v_idx, v_shape = (
            (d_idx, d_shape, s_idx, s_shape) if to_hbm else (s_idx, s_shape, d_idx, d_shape)
        )
        assert h_idx[1].size == v_idx[0].size

        def note(page, row, vmem_row):
            log.append(
                (to_hbm, int(page), int(row), int(vmem_row), h_idx[1].size,
                 h_shape[0], h_shape[1], v_shape[0])
            )

        jax.debug.callback(note, h_idx[0], h_idx[1].start, v_idx[0].start)
        return original(source, dest, semaphore)

    monkeypatch.setattr(main_write.pltpu, "make_async_copy", record)
    return log


def dma_out_of_bounds(log):
    return [
        d for d in log
        if not (0 <= d[1] < d[5] and 0 <= d[2] and d[2] + d[4] <= d[6]
                and 0 <= d[3] and d[3] + d[4] <= d[7])
    ]


PAIRED_LAYOUTS = {
    "precompile_4096": lambda: precompile_layout(4096),
    "verify_k2_4096": lambda: verify_step_layout(4096, 2),
    "verify_k3_4096": lambda: verify_step_layout(4096, 3),
    # _warmup_one_spec_request's prefill: one fresh request whose table is
    # range(1, 1 + entries).
    **{f"spec_warmup_{p}": (lambda p=p: ([p], [0])) for p in (1024, 3072, 4096)},
}


@pytest.mark.parametrize("slot", [0, 1, None])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("layout", sorted(PAIRED_LAYOUTS))
def test_served_paired_run_writer_dmas_in_bounds(interpret, monkeypatch, layout, rank, slot):
    """``write_main_cache`` at the c32 recipe (RUN_WRITE=1, TILE=1024, paired
    FP8 pages, main-KV DCP 2): every DMA the kernel issues, the paired
    HBM->VMEM merge reads included, stays inside the pool, its page and its
    tile's VMEM rows, and the pool matches the row-by-row reference."""
    log = recorded_dmas(monkeypatch)
    pages = 48
    query_lens, contexts = PAIRED_LAYOUTS[layout]()
    page, logical, valid = served_destinations(query_lens, contexts, rank, pages)
    rng = np.random.default_rng(7 + rank)
    pool = jnp.asarray(rng.normal(0, 1, (pages, SERVED_ROWS, 1, 4, 256)), jnp.float8_e4m3fn)
    rows = rng.normal(0, 1, (len(page), 1, 4, 256))
    if slot is not None:
        rows[:, :, 2:] = 0
    rows = jnp.asarray(rows, jnp.float8_e4m3fn)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_RUN_WRITE", True)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE", 1024)
    kw = {} if slot is None else {"slot": slot}
    got = pooled_qsa.write_main_cache(
        pool, rows, jnp.asarray(page), jnp.asarray(logical), jnp.asarray(valid), **kw
    )
    got = raw(jax.block_until_ready(got))
    jax.effects_barrier()
    expected_reads = int(valid.sum()) if slot is not None else 0
    assert sum(d[4] for d in log if not d[0]) == expected_reads
    assert sum(d[4] for d in log if d[0]) == int(valid.sum())
    assert dma_out_of_bounds(log) == []
    expected, new = raw(pool).copy(), raw(rows)
    for i in np.flatnonzero(valid):
        p, o = page[i], logical[i] % SERVED_ROWS
        if slot is None:
            expected[p, o] = new[i]
        else:
            expected[p, o, :, 2 * slot : 2 * slot + 2] = new[i, :, :2]
    np.testing.assert_array_equal(got, expected)


def test_recorded_dmas_catch_sentinel_page(interpret, monkeypatch):
    """Positive control: a valid row on the ``len(pool)`` sentinel page shows
    up as an out-of-bounds paired read."""
    log = recorded_dmas(monkeypatch)
    pool = jnp.zeros((4, SERVED_ROWS, 1, 4, 256), jnp.float8_e4m3fn)
    rows = jnp.zeros((1024, 1, 4, 256), jnp.float8_e4m3fn)
    page = jnp.full((1024,), 1, jnp.int32).at[700].set(len(pool))
    offsets = jnp.arange(1024, dtype=jnp.int32) % SERVED_ROWS
    jax.block_until_ready(
        main_write.write_main_runs(
            pool, rows, page, offsets, jnp.ones(1024, bool), keep=0xFFFF0000, tile=1024
        )
    )
    jax.effects_barrier()
    bad = dma_out_of_bounds(log)
    assert bad and {d[0] for d in bad} == {False, True} and all(d[1] == 4 for d in bad)
