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
"""Rope/TENSORCORE DMA run writer vs the per-token XLA scatter.

`rope_run_write` must leave the pool byte-identical to `_scatter_rows` -- the
whole pool, not just the written slots: partly covered head and tail slabs are
rewritten whole and must keep every byte the sequence does not own.

Runs on CPU in the Pallas TPU interpreter (`pltpu.InterpretParams`), which
models HBM refs, DMA semaphores and async copies, and raises on an
out-of-bounds DMA; plain `interpret=True` clamps the slice instead, which would
hide a run that spills past its page. Both DMA execution modes are exercised:
"on_wait" (a copy lands when waited for) and "eager" (when started). The
"tpu" mode (`tpu_test`) runs the same comparisons with the compiled kernel.
"""

import functools
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils, rope_run_write, scatter
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

LKV_DIM = 512
ROPE_DIM = 64  # bf16 -> 128 bytes, one lane row per token
KV_PACKING = 4
C = rope_run_write.CHUNK_SLABS
INTERPRET = {
    "on_wait": pltpu.InterpretParams(),
    "eager": pltpu.InterpretParams(dma_execution_mode="eager"),
    "tpu": False,
}
MODES = ["on_wait", "eager", pytest.param("tpu", marks=pytest.mark.tpu_test)]

# (page_size, num_pages, num_seqs, pages_per_seq). 1024 is a production page
# (256 slabs); the small pages put many page crossings into every batch.
CONFIGS = [(64, 128, 40, 64), (128, 96, 32, 32), (1024, 64, 24, 4)]


def _spec(cache_type, layout, page_size, num_pages, head_dim=ROPE_DIM):
    return SparseMLAKVCacheSpec.create(
        cache_type, layout, num_pages, page_size, head_dim, KV_PACKING
    )


def _rope_spec(page_size, num_pages):
    return _spec(KVCacheType.ROPE, KVCacheLayout.TENSORCORE, page_size, num_pages)


@functools.cache
def _golden_fn(spec):
    @jax.jit
    def f(pool, k_pe, seq_lens, block_tables, query_start_loc):
        page, slot = kv_cache_utils.get_page_and_slot(
            k_pe.shape[0], seq_lens, block_tables, query_start_loc, spec.page_size
        )
        return kv_cache_utils._scatter_rows(pool, spec, k_pe, page, slot)

    return f


@functools.cache
def _run_fn(spec, mode):
    return jax.jit(
        lambda pool, *args: rope_run_write.write_rope_runs(
            pool, *args, spec, interpret=INTERPRET[mode]
        )
    )


def _build(spec, num_tokens, num_seqs, pages_per_seq, segs, rng):
    """Inputs for a batch of `segs` = [(first_position, q_len), ...].

    Unused sequence slots are padding (q_len 0, seq_len 0), as are the tokens
    past the last segment. Every page a sequence writes is its own physical
    page, never page 0; the block-table entries it does not write stay 0.
    """
    page_size = spec.page_size
    segs = list(segs) + [(0, 0)] * (num_seqs - len(segs))
    q_lens = np.array([n for _, n in segs])
    assert q_lens.sum() <= num_tokens
    seq_lens = np.array([f + n for f, n in segs], np.int32)
    block_tables = np.zeros((num_seqs, pages_per_seq), np.int32)
    free = list(rng.permutation(np.arange(1, spec.num_pages)))
    for s, (f, n) in enumerate(segs):
        if n:
            for k in range(f // page_size, (f + n - 1) // page_size + 1):
                block_tables[s, k] = free.pop()
    query_start_loc = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    # Random old bytes, so a slab rewritten with the wrong old content shows.
    pool = rng.integers(0, 256, spec.shape, dtype=np.uint8)
    k_pe = rng.standard_normal((num_tokens, ROPE_DIM)).astype(np.float32)
    return (
        jnp.asarray(pool),
        jnp.asarray(k_pe, jnp.bfloat16),
        jnp.asarray(seq_lens),
        jnp.asarray(block_tables.reshape(-1)),
        jnp.asarray(query_start_loc),
    )


def _assert_same_pool(got, want):
    bad = np.argwhere(np.asarray(got) != np.asarray(want))
    assert bad.size == 0, (
        f"{len(bad)} bytes differ, first at (page, slab, row, byte)={tuple(bad[0])}"
    )


def _check(spec, args, mode):
    if mode == "tpu" and jax.devices()[0].platform != "tpu":
        pytest.skip("the compiled kernel needs a TPU")
    _assert_same_pool(_run_fn(spec, mode)(*args), _golden_fn(spec)(*args))


def _random_segs(rng, num_tokens, num_seqs, page_size, pages_per_seq):
    """One long prefill segment plus short (speculative-verify-like) windows
    at arbitrary positions, zero-length sequences in between, padding rows at
    the end."""
    max_ctx = page_size * pages_per_seq
    windows = [int(rng.integers(1, 6)) for _ in range(rng.integers(0, num_seqs))]
    budget = num_tokens - sum(windows)
    pad = int(rng.integers(0, min(64, budget)))
    prefill = max(1, min(budget - pad, max_ctx - 1))
    if rng.random() < 0.25:
        prefill = int(rng.integers(1, prefill + 1))
    lens = [prefill, *windows]
    rng.shuffle(lens)
    segs = []
    for n in lens:
        if rng.random() < 0.15:
            segs.append((int(rng.integers(0, 50)), 0))
        segs.append((int(rng.integers(0, max_ctx - n + 1)), n))
    return segs[:num_seqs]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("num_tokens", [512, 1024, 2048])
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"page{c[0]}")
def test_random_mixed_batch_matches_scatter(config, num_tokens, mode, seed):
    page_size, num_pages, num_seqs, pages_per_seq = config
    rng = np.random.default_rng([seed, num_tokens, page_size])
    spec = _rope_spec(page_size, num_pages)
    segs = _random_segs(rng, num_tokens, num_seqs, page_size, pages_per_seq)
    args = _build(spec, num_tokens, num_seqs, pages_per_seq, segs, rng)
    _check(spec, args, mode)


@pytest.mark.parametrize("mode", MODES)
def test_long_prefill_with_short_windows_production_page(mode):
    """A 1920-row prefill segment plus 32 four-row windows, 1024-row pages."""
    rng = np.random.default_rng(1234)
    spec = _rope_spec(1024, 96)
    segs = [(int(rng.integers(0, 2048)), 1920)]
    segs += [(int(rng.integers(0, 4096 - 4)), 4) for _ in range(32)]
    _check(spec, _build(spec, 2048, 40, 6, segs, rng), mode)


@pytest.mark.parametrize(
    "segs",
    [
        # Aligned start, run length (in slabs) around the chunk size.
        pytest.param([(0, C * KV_PACKING)], id="run_C"),
        pytest.param([(0, (C - 1) * KV_PACKING)], id="run_C_minus_1"),
        pytest.param([(0, (C + 1) * KV_PACKING)], id="run_C_plus_1"),
        pytest.param([(128, (2 * C + 3) * KV_PACKING)], id="run_2C_plus_3"),
        pytest.param([(8, KV_PACKING)], id="run_1_full_slab"),
        pytest.param([(9, 1)], id="run_1_single_row"),
        pytest.param([(13, 2)], id="run_1_two_rows_mid_slab"),
        # Start and/or end mid-slab: partly covered head and tail slabs.
        pytest.param([(2, C * KV_PACKING)], id="start_mid_slab"),
        pytest.param([(0, C * KV_PACKING - 1)], id="end_mid_slab"),
        pytest.param([(3, C * KV_PACKING + 2)], id="both_mid_slab"),
        pytest.param([(6, 4)], id="window_straddles_slab"),
        # Page boundaries (128 rows = 32 slabs per page).
        pytest.param([(64, 128)], id="cross_page_aligned"),
        pytest.param([(126, 7)], id="cross_page_mid_slab"),
        pytest.param([(127, 2)], id="cross_page_window"),
        pytest.param([(100, 300)], id="cross_two_pages"),
        pytest.param([(128, 128)], id="whole_page"),
        # Several sequences with zero-length ones between them.
        pytest.param(
            [(5, 200), (0, 0), (61, 4), (127, 3), (40, 0), (255, 1), (300, 17)],
            id="mixed",
        ),
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_edge_cases(segs, mode):
    rng = np.random.default_rng(zlib.crc32(repr(segs).encode()))
    spec = _rope_spec(128, 64)
    _check(spec, _build(spec, 512, 8, 8, segs, rng), mode)


def test_run_table():
    """The plan itself: page splits, sources and lengths for a known batch."""
    spec = _rope_spec(128, 64)  # 32 slabs per page
    rng = np.random.default_rng(7)
    # seq 0: positions 2..301 -> slabs 0..75 on pages 0 (0-31), 1 (32-63) and
    # 2 (64-75); seq 1 is empty; seq 2: positions 127..128, one slab on each of
    # two pages.
    pool, k_pe, seq_lens, block_tables, query_start_loc = _build(
        spec, 512, 4, 8, [(2, 300), (0, 0), (127, 2)], rng
    )
    _, runs = rope_run_write.rope_run_plan(
        pool,
        kv_cache_utils.as_token_bytes(k_pe, spec.token_bytes),
        seq_lens,
        block_tables,
        query_start_loc,
    )
    src, page, slab, length = (np.asarray(x) for x in runs)
    assert len(length) == 512 // 128 + 2 * 4
    live = length > 0
    np.testing.assert_array_equal(length[live], [32, 32, 12, 1, 1])
    np.testing.assert_array_equal(src[live], [0, 32, 64, 76, 77])
    np.testing.assert_array_equal(slab[live], [0, 0, 0, 31, 0])
    tables = np.asarray(block_tables).reshape(4, 8)
    np.testing.assert_array_equal(
        page[live],
        [tables[0, 0], tables[0, 1], tables[0, 2], tables[2, 0], tables[2, 1]],
    )


def test_comparison_detects_a_wrong_byte():
    """Negative control: `_assert_same_pool` must be able to fail."""
    spec = _rope_spec(128, 64)
    args = _build(spec, 512, 4, 8, [(3, 40)], np.random.default_rng(3))
    want = np.asarray(_golden_fn(spec)(*args))
    got = np.array(_run_fn(spec, "on_wait")(*args))
    _assert_same_pool(got, want)
    written = np.argwhere(want != np.asarray(args[0]))[0]
    got[tuple(written)] ^= 1
    with pytest.raises(AssertionError, match="1 bytes differ"):
        _assert_same_pool(got, want)


# Dispatch through `update_sparse_mla_kv_cache`.

MIXED = [(5, 300), (0, 0), (61, 4), (127, 3), (255, 1)]
DECODE = [(61, 4), (127, 3), (255, 1)]


@pytest.fixture
def calls(monkeypatch):
    """Count the rope writers; stub the SparseCore writer (needs a TPU) with
    the identity and run the DMA run writer in the interpreter."""
    counts = {"run": 0, "scatter_rows": 0, "sparsecore": 0}
    real_run = rope_run_write.write_rope_runs
    real_rows = kv_cache_utils._scatter_rows

    def run(*args, **kwargs):
        counts["run"] += 1
        return real_run(*args, **kwargs, interpret=INTERPRET["on_wait"])

    def rows(*args, **kwargs):
        counts["scatter_rows"] += 1
        return real_rows(*args, **kwargs)

    def sparsecore(cache, *args, **kwargs):
        counts["sparsecore"] += 1
        return cache

    monkeypatch.setattr(rope_run_write, "write_rope_runs", run)
    monkeypatch.setattr(kv_cache_utils, "_scatter_rows", rows)
    monkeypatch.setattr(scatter, "scatter", sparsecore)
    return counts


def _update_args(
    num_tokens, segs=MIXED, rope_layout=KVCacheLayout.TENSORCORE, page_size=128
):
    num_pages = 64
    nope = _spec(KVCacheType.NOPE, rope_layout, page_size, num_pages, LKV_DIM)
    rope = _spec(KVCacheType.ROPE, rope_layout, page_size, num_pages)
    rng = np.random.default_rng(num_tokens)
    _, k_pe, seq_lens, block_tables, query_start_loc = _build(
        _rope_spec(page_size, num_pages), num_tokens, 8, 8, segs, rng
    )
    nope_cache = jnp.asarray(rng.integers(0, 256, nope.shape), nope.jax_dtype)
    rope_cache = jnp.asarray(rng.integers(0, 256, rope.shape), rope.jax_dtype)
    kv_c = jnp.asarray(rng.standard_normal((num_tokens, LKV_DIM)), jnp.float8_e4m3fn)
    args = (
        nope_cache,
        rope_cache,
        kv_c,
        k_pe,
        seq_lens,
        block_tables,
        query_start_loc,
    )
    return args, {"nope_spec": nope, "rope_spec": rope}


@pytest.mark.parametrize("num_tokens", [kv_cache_utils.ROPE_RUN_MIN_TOKENS, 1024])
def test_tensorcore_large_bucket_takes_the_run_writer(calls, num_tokens):
    args, specs = _update_args(num_tokens)
    _, rope = kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    assert calls == {"run": 1, "scatter_rows": 0, "sparsecore": 1}
    _, want = kv_cache_utils.update_sparse_mla_kv_cache_jax(*args, **specs)
    _assert_same_pool(rope, want)


@pytest.mark.parametrize("num_tokens", [128, kv_cache_utils.ROPE_RUN_MIN_TOKENS - 1])
def test_tensorcore_small_bucket_keeps_the_scatter(calls, num_tokens):
    args, specs = _update_args(num_tokens, DECODE)
    kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    assert calls == {"run": 0, "scatter_rows": 1, "sparsecore": 1}


def test_sparsecore_rope_keeps_the_sparsecore_writer(calls):
    args, specs = _update_args(1024, rope_layout=KVCacheLayout.SPARSECORE)
    kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    assert calls == {"run": 0, "scatter_rows": 0, "sparsecore": 2}


def test_dcp_writer_keeps_the_scatter(calls):
    args, specs = _update_args(1024)
    dcp = {"dcp_size": 2, "interleave_size": 4}
    _, rope = kv_cache_utils.update_sparse_mla_kv_cache_dcp(
        *args, jnp.int32(1), **specs, **dcp
    )
    assert calls == {"run": 0, "scatter_rows": 1, "sparsecore": 1}
    _, want = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
        *args, jnp.int32(1), **specs, **dcp
    )
    _assert_same_pool(rope, want)


@pytest.mark.tpu_test
@pytest.mark.parametrize("page_size", [128, 1024])
@pytest.mark.parametrize("num_tokens", [kv_cache_utils.ROPE_RUN_MIN_TOKENS, 1024])
def test_update_on_tpu_matches_jax(monkeypatch, num_tokens, page_size):
    """Unstubbed, under jit: the SparseCore nope writer and the compiled run
    writer leave both caches equal to the pure-XLA golden."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the SparseCore writer and the compiled kernel need a TPU")
    traced = []
    real_run = rope_run_write.write_rope_runs

    def run(*args, **kwargs):
        traced.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(rope_run_write, "write_rope_runs", run)
    args, specs = _update_args(num_tokens, page_size=page_size)
    got = jax.jit(
        functools.partial(kv_cache_utils.update_sparse_mla_kv_cache, **specs)
    )(*args)
    want = jax.jit(
        functools.partial(kv_cache_utils.update_sparse_mla_kv_cache_jax, **specs)
    )(*args)
    assert traced == [1]
    for g, w in zip(got, want, strict=True):
        _assert_same_pool(g, w)
