# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DECODE_KV_BLOCK_ONE_WINDOW (q38, DS-O1S): the decode indexer's
KV block on steps that stay one shared-stream query window.

In the decode engine (O1, TP8 / DCP4; T46 j-da0afb0c) a 6656-key block in place
of 3328 saves 32 us per StreamIndex call on one-window steps and costs 7 us on
the 128-row windows of a two-window step, so the larger block is taken only when
the step is one window with it. These tests check (1) the switch: the block the
kernel receives per step (one window -> the one-window block, more windows ->
VLLM_TPU_QSA_DECODE_KV_BLOCK), with the window derived for the candidate block;
(2) exactness: the same score bits and selection with the flag as without, on the
served path (pool-page table) and the plain shared stream. Fixtures (reference
top-k, interpret-mode KV copy, inputs, cases) follow
test_qwen4_exp_indexer_key_lanes.py (DEC-IDX); the real StreamIndex scoring
kernel runs in TPU interpret mode on CPU.
"""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

streamindex = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)
bref_override = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.bref_override"
)

RATIO = 4
ROWS = 128 // RATIO
BLOCK_TOPK = 16
POOL_PAGES, TOKENS_PER_PAGE, WIDTH, KEY_DIM = 24, 512, 256, 128
TABLE_COLUMNS = 8
FLAG = "VLLM_TPU_QSA_DECODE_KV_BLOCK_ONE_WINDOW"
BASE, ONE = 1024, 2048  # small-shape stand-ins for the served 3328 / 6656


class ReferenceTopK:
    """JAX stand-in for the SparseCore selector (lowest IDs win ties)."""

    overlaps_scoring = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None, :] < row_lengths[:, None],
            scores,
            -jnp.inf,
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(values > -jnp.inf, ids, -1)
        bits = jax.lax.bitcast_convert_type(values, jnp.int32)
        return (ids, bits) if return_scores else ids

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, ReferenceTopK)


# (copy source lanes, VMEM window lanes) of every KV copy issued while tracing.
COPIES: list[tuple[int, int]] = []


def _copy_in_pages(self, src_ref, grid_indices):
    """Interpret-mode KV copy: the served copy's pages and lanes, per page.

    The served ``copy_in`` reads the same pages through a flattened HBM view
    (and, with POOL_PAGE_DMA, one copy per pool page); both go through
    ``row_source``, which this override uses as well.
    """
    cache_kv_hbm, page_indices_ref, meta_ref = src_ref
    p_id = grid_indices[0] + bref_override._step_offset(meta_ref, self.chunk_idx)
    seq = meta_ref.batch_tile_idx[p_id]
    kv_p_start = meta_ref.bkv_idx[p_id] * self.bkv_p
    slot = self.current_copy_in_slot
    vmem_dst = self.window_ref.at[slot]
    for batch_idx in range(self.seq_batch_size):
        for i in range(self.bkv_p):
            page = self.page_id(page_indices_ref, seq + batch_idx, kv_p_start, i)
            page = jnp.minimum(page, cache_kv_hbm.shape[0] - 1)
            src = cache_kv_hbm.at[page]
            dst = vmem_dst.at[
                batch_idx,
                pl.ds(
                    i * self.page_size_per_kv_packing,
                    self.page_size_per_kv_packing,
                ),
            ]
            COPIES.append((src.shape[-1], dst.shape[-1]))
            pltpu.make_async_copy(src, dst, self.sem_recvs.at[slot]).start()


@pytest.fixture
def interpret(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(
        streamindex.pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    monkeypatch.setattr(bref_override.StreamIndexKVBufferedRef, "copy_in", _copy_in_pages)
    COPIES.clear()
    for name in (
        FLAG,
        "VLLM_TPU_QSA_INDEXER_KEY_LANES",
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA",
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)


def capture_scores(monkeypatch):
    """Route selection through the reference top-k and record winner scores."""
    calls = []

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, return_scores=True)
        ids, bits = streamindex.streamindex_topk(
            *args, topk_backend=ReferenceTopK(), **kwargs
        )
        calls.append(np.asarray(bits))
        return ids

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    return calls


def ragged_step(q_lens, context, tokens):
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    req = np.zeros(tokens, np.int32)
    lengths = np.zeros(tokens, np.int32)
    for r, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        req[begin:end] = r
        lengths[begin:end] = context[r] + np.arange(1, end - begin + 1)
    return starts, req, lengths


def inputs(seed, q_lens, context, tokens, data="normal", pad="normal"):
    """``data``: key / query values ("integer": exact in any summation order,
    with many exact ties). ``pad``: what the pool's lanes past the 128 key
    lanes hold ("zero", "normal" or "nan")."""
    rng = np.random.default_rng(seed)
    starts, req, lengths = ragged_step(q_lens, context, tokens)

    def sample(shape):
        if data == "integer":
            return rng.integers(-3, 4, size=shape).astype(np.float32)
        return rng.normal(size=shape).astype(np.float32)

    pool = sample((POOL_PAGES, TOKENS_PER_PAGE, 1, 2, WIDTH))
    if pad == "zero":
        pool[..., KEY_DIM:] = 0
    elif pad == "nan":
        pool[..., KEY_DIM:] = np.nan
    table = np.stack(
        [rng.permutation(POOL_PAGES)[:TABLE_COLUMNS] for _ in range(len(context))]
    )
    return dict(
        query=jnp.asarray(sample((tokens, 4, KEY_DIM)), jnp.bfloat16),
        pool=jnp.asarray(pool, jnp.bfloat16),
        lengths=jnp.asarray(lengths),
        table=jnp.asarray(table, jnp.int32),
        req=jnp.asarray(req),
        starts=jnp.asarray(starts),
        context=jnp.asarray(context, jnp.int32),
    )


# Contexts span several KV blocks and table columns, partial ring tails and
# idle rows; verify_k3 is the served k3 shape (4 rows per request).
CASES = {
    "verify_k3": (
        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3001, 2048, 1203] + [0] * 9,
        12,
    ),
    "ragged_1_to_4": (
        [1, 2, 0, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2999, 1022, 7, 4093 - 3, 517, 3500] + [0] * 10,
        16,
    ),
    "decode_prefix": (
        [1] * 9 + [0] * 3,
        [4000, 3, 64, 65, 66, 67, 999, 2048, 3001, 0, 0, 0],
        12,
    ),
}

PATHS = {
    "served": {
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK": str(BASE),
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    },
    "shared": {
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK": str(BASE),
    },
}


def set_path(monkeypatch, path, one=None):
    for name, value in PATHS[path].items():
        monkeypatch.setenv(name, value)
    if one is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, str(one))


def run(monkeypatch, args):
    """select_blocks through the reference top-k: (ids, score bits per call, the KV
    pages per block of the multi-row passes per call)."""
    bits, kv_pages = [], []

    def scored(*a, **kw):
        kv_pages.append(tuple(kw["num_kv_pages_per_block"])[1:])
        kw.update(enable_early_exit=False, return_scores=True)
        ids, b = streamindex.streamindex_topk(*a, topk_backend=ReferenceTopK(), **kw)
        if isinstance(b, jax.core.Tracer):
            # A multi-window step scores inside _select_shared_stream's lax.scan:
            # take each window's bits to the host as the scan runs.
            jax.debug.callback(lambda x: bits.append(np.asarray(x)), b, ordered=True)
        else:
            bits.append(np.asarray(b))
        return ids

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    ids = np.asarray(pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK))
    jax.effects_barrier()  # every window's bits callback has run
    return ids, bits, kv_pages


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_one_window_block_is_exact_and_engaged(monkeypatch, interpret, case, path):
    """One-window steps (the default window is far wider than these steps):
    with the flag the kernel gets the one-window block, and every score bit and
    selected id equals the flag-off call."""
    q_lens, context, tokens = CASES[case]
    args = inputs(3, q_lens, context, tokens, data="normal")
    set_path(monkeypatch, path)
    ids0, bits0, pages0 = run(monkeypatch, args)
    set_path(monkeypatch, path, one=ONE)
    ids1, bits1, pages1 = run(monkeypatch, args)
    assert all(p == (BASE // ROWS,) * 2 for p in pages0), pages0
    assert all(p == (ONE // ROWS,) * 2 for p in pages1), pages1
    np.testing.assert_array_equal(ids1, ids0)
    assert len(bits1) == len(bits0)
    for a, b in zip(bits0, bits1):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("path", sorted(PATHS))
def test_multi_window_steps_keep_the_base_block(monkeypatch, interpret, path):
    """A step cut into several query windows keeps VLLM_TPU_QSA_DECODE_KV_BLOCK
    (as the 256-token bucket's two 128-row windows in the engine), exactly."""
    q_lens, context, tokens = CASES["ragged_1_to_4"]
    args = inputs(5, q_lens, context, tokens, data="normal")
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 8)
    set_path(monkeypatch, path)
    ids0, bits0, pages0 = run(monkeypatch, args)
    set_path(monkeypatch, path, one=ONE)
    ids1, bits1, pages1 = run(monkeypatch, args)
    assert len(pages0) >= 1 and all(p == (BASE // ROWS,) * 2 for p in pages0), pages0
    assert len(pages1) >= 1 and all(p == (BASE // ROWS,) * 2 for p in pages1), pages1
    np.testing.assert_array_equal(ids1, ids0)
    # 16 tokens in windows of 8: the scan scored every window in each call.
    assert len(bits0) == len(bits1) >= 2, (len(bits0), len(bits1))
    for a, b in zip(bits0, bits1):
        np.testing.assert_array_equal(a, b)


def test_switch_rule():
    """The block per step: the one-window block iff the step fits ONE window
    derived for that block; off / equal blocks are no-ops."""
    f = pooled_qsa.one_window_kv_block
    w128 = lambda block: 128
    assert f(112, 3328, 6656, w128) == 6656
    assert f(128, 3328, 6656, w128) == 6656   # boundary: exactly one window
    assert f(129, 3328, 6656, w128) == 3328
    assert f(256, 3328, 6656, w128) == 3328   # two 128-row windows
    assert f(112, 3328, 0, w128) == 3328      # off
    assert f(112, 3328, 3328, w128) == 3328   # same block
    # The window is derived for the candidate block: a block whose headroom
    # shrinks the window below the step keeps the base block.
    shrinking = lambda block: 128 if block <= 3328 else 64
    assert f(112, 3328, 6656, shrinking) == 3328
    assert f(64, 3328, 6656, shrinking) == 6656


@pytest.mark.parametrize("bad", ["100", "160"])
def test_rejects_bad_one_window_block(monkeypatch, interpret, bad):
    q_lens, context, tokens = CASES["verify_k3"]
    args = inputs(1, q_lens, context, tokens)
    set_path(monkeypatch, "served", one=bad)
    with pytest.raises(ValueError, match="VLLM_TPU_QSA_DECODE_KV_BLOCK_ONE_WINDOW"):
        run(monkeypatch, args)


def test_env_parse(monkeypatch):
    """Unset / empty = off; an integer is read; anything else is an error."""
    monkeypatch.delenv(FLAG, raising=False)
    assert pooled_qsa.one_window_env_block() == 0
    monkeypatch.setenv(FLAG, " ")
    assert pooled_qsa.one_window_env_block() == 0
    monkeypatch.setenv(FLAG, "6656")
    assert pooled_qsa.one_window_env_block() == 6656
    for bad in ("abc", "-128", "6656.0"):
        monkeypatch.setenv(FLAG, bad)
        with pytest.raises(ValueError, match=FLAG):
            pooled_qsa.one_window_env_block()
