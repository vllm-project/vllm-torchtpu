# SPDX-License-Identifier: Apache-2.0
"""Run-based main KV writer (interpret mode): the same pool as
write_main_cache's tiled kernels and a row-by-row reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.qwen38_flash.experimental.kernels.test_fused_kv_write import (  # noqa: F401 (fixture)
    interpret,
    raw,
)
from vllm_torchtpu.models.qwen38_flash.experimental import settings
from vllm_torchtpu.models.qwen38_flash.experimental.kernels import (
    main_write,
    pooled_qsa,
)

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
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_MAIN_RUN_WRITE", run_write)
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS", 0)
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE", tile)
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
def test_run_writer_matches_tiled_writer(interpret, monkeypatch, case, layout, dcp):  # noqa: F811
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
def test_repeated_destinations_keep_last_writer(interpret, monkeypatch, slot):  # noqa: F811
    rng = np.random.default_rng(9)
    tokens = 256
    pool, rows = make_rows(rng, tokens, jnp.float8_e4m3fn, slot)
    # Runs of 5 that restart: offsets 3..7, 3..7, ... on one page, plus a
    # second page whose run crosses the 128-row tile edge.
    offset = np.where(
        np.arange(tokens) < 128, np.arange(tokens) % 5 + 3, np.arange(tokens)
    )
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
