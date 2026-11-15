# SPDX-License-Identifier: Apache-2.0
"""Decode fixed-cost flags against their off path, byte for byte, at served shapes.

- VLLM_TPU_PLE_SKIP_IDLE_ROWS: `pooled_short_conv_speculative` output and the
  whole pool after the call.
- VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS: the resident raw lookup in chunks of up
  to 192 tokens against 8-token chunks, on a table of the served shard's shape.
- VLLM_TPU_QSA_PAIRED_MAIN_PAD_TILE: the whole pool after both layers of a
  page pair write their main KV rows, at row counts that are not multiples
  of 128 (the paired-slot kernel against the XLA gather and scatter).

Served per-rank geometry (Qwen3.8-Flash, TP8, fp8 KV pool of 512-token
blocks): PLE history 9 x 10240 bf16, MTP K=3, max_num_seqs 128.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pool as ple_pool
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.ple import resident_lookup_raw

WINDOW = 4  # MTP K=3
REQ_ROWS = 128  # max_num_seqs
POOL_BLOCK = (512, 1, 4, 256)
POOL_DTYPE = jnp.float8_e4m3fn

# PLE: (ple_conv_kernel_size - 1) * ngram_size history rows of hidden * hc channels.
PLE_C, PLE_S, PLE_DIL = 10240, 9, 3
# PLE table: one TP shard of the served FP8 table (u8 [50000240, 128]).
TABLE_ROWS, HEADS, HEAD_DIM = 50000240, 16, 160
LOCAL_ROWS = TABLE_ROWS * 128 // HEAD_DIM


def _random_pool(seed, blocks):
    bits = jax.random.bits(jax.random.key(seed), (blocks,) + POOL_BLOCK, jnp.uint8)
    return jax.lax.bitcast_convert_type(bits, POOL_DTYPE)


def _pool_bytes(pool):
    return np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8))


def _rows(segments, rows=REQ_ROWS):
    """Per-row (length, windowed, initial, has_state) from leading segments.

    ``segments``: (count, length, windowed, initial, has_state) groups in
    row order; the remaining rows are idle (no tokens, no state).
    """
    out = []
    for count, length, windowed, initial, has_state in segments:
        out += [(length, windowed, initial, has_state)] * count
    assert len(out) <= rows
    out += [(0, False, False, False)] * (rows - len(out))
    return out


# name -> (token bucket, row segments). Rows with tokens lead, as the runner
# lays them out; one layout also has a windowed row without a state slot.
PLE_LAYOUTS = {
    "decode24_t128": (128, [(24, 4, True, True, True)]),
    "decode48_t256": (256, [(48, 4, True, True, True)]),
    "decode128_t512": (512, [(128, 4, True, True, True)]),
    "decode16_t64": (64, [(16, 4, True, True, True)]),
    "ones_t128": (128, [(100, 1, True, True, True), (7, 4, True, True, True)]),
    "mixed_t128": (
        128,
        [
            (17, 4, True, True, True),
            (1, 4, True, True, False),
            (4, 1, True, True, True),
            (1, 30, False, True, True),
            (1, 12, False, False, True),
        ],
    ),
    "prefill_t1024": (
        1024,
        [
            (100, 4, True, True, True),
            (1, 300, False, True, True),
            (1, 200, False, False, True),
        ],
    ),
}


def _ple_inputs(name, seed):
    tokens, segments = PLE_LAYOUTS[name]
    rows = _rows(segments)
    rng = np.random.default_rng(seed)
    lengths = np.array([r[0] for r in rows], np.int32)
    assert lengths.sum() <= tokens
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    windowed = np.array([r[1] for r in rows])
    initial = np.array([r[2] for r in rows])
    has_state = np.array([r[3] for r in rows])
    blocks = 1 + WINDOW * REQ_ROWS + 8
    perm = rng.permutation(np.arange(1, blocks))[: WINDOW * REQ_ROWS]
    ckpt = perm.reshape(REQ_ROWS, WINDOW).astype(np.int32)
    ckpt[~has_state] = 0
    ids = np.where(has_state, ckpt[:, 0], 0).astype(np.int32)
    offsets = rng.integers(0, WINDOW, REQ_ROWS)
    read_ids = ckpt[np.arange(REQ_ROWS), offsets].astype(np.int32)
    pool = _random_pool(seed, blocks)
    # Finite histories in every block a row reads; garbage everywhere else.
    histories = jax.random.normal(
        jax.random.key(seed + 1), (REQ_ROWS, PLE_S * PLE_C), jnp.bfloat16
    )
    pool = jax.jit(ple_pool.write_region)(pool, histories, jnp.asarray(read_ids))
    x = jax.random.normal(jax.random.key(seed + 2), (tokens, PLE_C), jnp.bfloat16)
    weight = jax.random.normal(jax.random.key(seed + 3), (PLE_C, 4), jnp.bfloat16)
    return (
        x,
        weight,
        pool,
        jnp.asarray(starts),
        jnp.asarray(ids),
        jnp.asarray(initial),
        jnp.asarray(read_ids),
        jnp.asarray(ckpt),
        jnp.asarray(windowed),
    )


def _ple_run(monkeypatch, flag, args):
    monkeypatch.setenv("VLLM_TPU_PLE_SKIP_IDLE_ROWS", "1" if flag else "0")
    fn = jax.jit(
        lambda *a: ple_pool.pooled_short_conv_speculative(
            *a, dilation=PLE_DIL, state_len=PLE_S
        )
    )
    pool, out = fn(*args)
    return _pool_bytes(pool), np.asarray(jax.lax.bitcast_convert_type(out, jnp.uint16))


@pytest.mark.tpu_test
@pytest.mark.parametrize("name", sorted(PLE_LAYOUTS))
def test_ple_skip_idle_rows_is_byte_identical(monkeypatch, name):
    args = _ple_inputs(name, seed=sorted(PLE_LAYOUTS).index(name) + 11)
    before = _pool_bytes(args[2])
    pool_off, out_off = _ple_run(monkeypatch, False, args)
    pool_on, out_on = _ple_run(monkeypatch, True, args)
    assert not np.array_equal(pool_off, before), "the step wrote no checkpoint"
    assert np.any(out_off != 0), "the step produced no output"
    np.testing.assert_array_equal(out_on, out_off)
    np.testing.assert_array_equal(pool_on, pool_off)


@pytest.mark.tpu_test
def test_ple_selected_read_matches_on_selected_rows():
    blocks = 64
    pool = _random_pool(5, blocks)
    rng = np.random.default_rng(5)
    ids = jnp.asarray(rng.integers(0, blocks, 32), jnp.int32)
    selected = jnp.asarray(rng.random(32) < 0.4).at[0].set(False)
    elements = PLE_S * PLE_C
    full = ple_pool.read_region(pool, ids, elements=elements, dtype=jnp.bfloat16)
    part = ple_pool.read_selected_region(
        pool, ids, selected, elements=elements, dtype=jnp.bfloat16
    )
    rows = np.flatnonzero(np.asarray(selected))
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(part, jnp.uint16))[rows],
        np.asarray(jax.lax.bitcast_convert_type(full, jnp.uint16))[rows],
    )


def hashed_table(rows=TABLE_ROWS):
    """A [rows, 128] uint8 table of mixed bytes, built in one elementwise pass
    (random bits for 6.4 GB would need ~119 GB of threefry temporaries)."""

    def build():
        r = jax.lax.broadcasted_iota(jnp.uint32, (rows, 128), 0)
        c = jax.lax.broadcasted_iota(jnp.uint32, (rows, 128), 1)
        h = r * jnp.uint32(2654435761) + c * jnp.uint32(40503)
        h = h ^ (h >> 15)
        return (h >> 7).astype(jnp.uint8)

    return jax.block_until_ready(jax.jit(build)())


@pytest.fixture(scope="module")
def ple_table():
    return hashed_table()


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [64, 128, 256, 512, 1024])
def test_ple_lookup_one_gather_is_byte_identical(ple_table, tokens):
    rng = np.random.default_rng(tokens)
    vocab_start = 2 * LOCAL_ROWS
    ids = rng.integers(
        vocab_start - 4096, vocab_start + LOCAL_ROWS + 4096, (tokens, HEADS)
    )
    ids[rng.random((tokens, HEADS)) < 0.05] = -1
    ids = jnp.asarray(ids, jnp.int32)
    start = jnp.asarray([vocab_start], jnp.int32)

    def lookup(chunk):
        return jax.jit(
            lambda t, i, s: resident_lookup_raw(
                t, i, s, local_rows=LOCAL_ROWS, head_dim=HEAD_DIM, chunk_tokens=chunk
            )
        )(ple_table, ids, start)

    chunked = np.asarray(lookup(8))
    assert np.any(chunked != 0)
    # Gathers of up to 192 tokens (6144 table rows) ran in the served decode
    # programs; larger chunks are probed one per process by
    # tpu_probe_ple_lookup_chunk.py, since a device fault ends the process.
    for chunk in (64, 128, 192):
        np.testing.assert_array_equal(np.asarray(lookup(chunk)), chunked)


# name -> (token bucket, verify windows of 4, DCP ranks owning a row).
PAIRED_LAYOUTS = {
    "decode22_t96": (96, 22, 1),
    "decode24_t96_dcp4": (96, 24, 4),
    "decode40_t160": (160, 40, 1),
    "decode46_t192": (192, 46, 1),
    "decode48_t192_dcp4": (192, 48, 4),
    "decode80_t320_dcp4": (320, 80, 4),
}


def _paired_inputs(name, seed):
    tokens, requests, dcp = PAIRED_LAYOUTS[name]
    rng = np.random.default_rng(seed)
    pages = 2 * requests + 8
    pool = _random_pool(seed, pages)
    page = np.full(tokens, pages, np.int32)
    logical = np.zeros(tokens, np.int32)
    order = rng.permutation(np.arange(pages))
    for r in range(requests):
        # Each request owns two pages; some windows cross the page boundary.
        start = int(rng.integers(POOL_BLOCK[0] - 6, POOL_BLOCK[0] + 200))
        positions = start + np.arange(WINDOW)
        rows = slice(WINDOW * r, WINDOW * (r + 1))
        logical[rows] = positions
        page[rows] = order[2 * r + positions // POOL_BLOCK[0] - 1]
    valid = np.arange(tokens) < WINDOW * requests
    if dcp > 1:
        valid &= rng.integers(0, dcp, tokens) == 0
    page = np.where(valid, page, pages)
    keys = iter(jax.random.split(jax.random.key(seed), 4))

    def layer_rows():
        k = jax.random.normal(next(keys), (tokens, 1, 256), jnp.bfloat16)
        v = jax.random.normal(next(keys), (tokens, 1, 256), jnp.bfloat16)
        return qsa_rpa.pack_rows(k, v, fp8=True, k_scale=0.375, v_scale=2.5)

    return (
        pool,
        [layer_rows(), layer_rows()],
        (
            jnp.asarray(page),
            jnp.asarray(logical),
            jnp.asarray(valid),
        ),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("name", sorted(PAIRED_LAYOUTS))
def test_paired_main_pad_tile_is_byte_identical(monkeypatch, name):
    pool, layers, (page, logical, valid) = _paired_inputs(
        name, seed=sorted(PAIRED_LAYOUTS).index(name) + 31
    )
    before = _pool_bytes(pool)
    results = {}
    for flag in (False, True):
        monkeypatch.setenv("VLLM_TPU_QSA_PAIRED_MAIN_PAD_TILE", "1" if flag else "0")

        # A new function per flag: jit's trace cache is keyed on the function,
        # and the flag is read at trace time.
        def write_fn(pool, rows, page, logical, valid, slot):
            return pooled_qsa.write_main_cache(
                pool, rows, page, logical, valid, slot=slot
            )

        write = jax.jit(write_fn, static_argnames=("slot",))
        text = (
            write.lower(pool, layers[0], page, logical, valid, slot=0)
            .compile()
            .as_text()
        )
        # The XLA path is a gather and a scatter; the paired-slot writer is
        # the only Pallas call in this function.
        assert ("tpu_custom_call" in text) == flag
        out = pool
        for slot, rows in enumerate(layers):
            out = write(out, rows, page, logical, valid, slot=slot)
        results[flag] = _pool_bytes(out)
    assert not np.array_equal(results[False], before), "the step wrote nothing"
    np.testing.assert_array_equal(results[True], results[False])
