# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_INDEXER_KEY_LANES (DEC-FB lever 11): the decode / verify
indexer copies only the real 128 lanes of each 256-lane compressed-key row.

The index keys are 128-dim BF16 rows stored in the pool's 256-lane rows; the
upper 128 lanes are never part of a score. With the flag the StreamIndex KV
copies move lanes [0, 128) only, so the key bytes halve and every score,
and so every selection, is the same.

The real StreamIndex scoring kernel runs in TPU interpret mode on CPU; a JAX
top-k stands in for the SparseCore selector, so both arms compare identical
score bits. Interpret mode cannot DMA from the flattened HBM view (or bitcast
refs), so the copy override below moves the same pages through the same
``row_source`` the served copies use, from the BF16 pool. The FP8 carrier's
served call is checked by lowering (scope tag) here and by the TPU bench.
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
indexer_components = importlib.import_module(
    "vllm_torchtpu.kernels.qwen4_exp.indexer_components"
)

RATIO = 4
BLOCK_TOPK = 16
POOL_PAGES, TOKENS_PER_PAGE, WIDTH, KEY_DIM = 24, 512, 256, 128
TABLE_COLUMNS = 8
FLAG = "VLLM_TPU_QSA_INDEXER_KEY_LANES"


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
            src = self.row_source(cache_kv_hbm, page)
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


def select(args, **kw):
    return np.asarray(
        pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK, **kw)
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

# Decode-path flag sets: the per-token decode passes (no query block), the
# shared stream, and the served set (pool-page table, mixed pass only).
PATHS = {
    "per_token": {},
    "shared_256": {
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK": "256",
    },
    "served": {
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK": "1024",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    },
}


def set_path(monkeypatch, path):
    for name, value in PATHS[path].items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("data", ["integer", "normal"])
@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_key_lanes_keep_scores_and_selection(monkeypatch, interpret, case, path, data):
    """128 lanes against whole rows: the same winners and winner score bits,
    with the copies of the flagged arm moving 128 lanes into 128-lane windows."""
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case) + len(path), q_lens, context, tokens, data, pad="normal")
    calls = capture_scores(monkeypatch)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    set_path(monkeypatch, path)

    # COPIES fills while the kernel traces: drop cached traces before each arm.
    jax.clear_caches()
    expected = select(args, max_query_len=4)
    whole = set(COPIES)
    COPIES.clear()
    jax.clear_caches()
    monkeypatch.setenv(FLAG, "128")
    actual = select(args, max_query_len=4)
    narrow = set(COPIES)

    assert whole == {(WIDTH, WIDTH)}
    assert narrow == {(KEY_DIM, KEY_DIM)}
    np.testing.assert_array_equal(actual, expected)
    bits_ref, bits_new = calls
    if data == "integer" or jax.default_backend() == "tpu":
        np.testing.assert_array_equal(bits_new, bits_ref)
    else:
        # XLA:CPU may block a K=256 GEMM (zero-padded query) differently from
        # K=128 in the last bits; the served FP8 carrier's query is never
        # padded, so its contraction is 128 wide on both arms.
        finite = bits_ref != streamindex.BITS_NEG_INF
        np.testing.assert_array_equal(bits_new[~finite], bits_ref[~finite])
        assert np.max(np.abs(bits_new.astype(np.int64) - bits_ref)[finite], initial=0) <= 4
    lengths = np.asarray(args["lengths"])
    assert np.any(lengths // RATIO > BLOCK_TOPK)


@pytest.mark.parametrize("path", sorted(PATHS))
def test_key_lanes_never_read_the_upper_lanes(monkeypatch, interpret, path):
    """NaN in every key row's lanes 128-255: the flagged arm selects what whole
    rows select with zero padding (no NaN reaches a score); whole rows with the
    NaN padding select differently (the poison is visible when read)."""
    q_lens, context, tokens = CASES["verify_k3"]
    clean = inputs(3, q_lens, context, tokens, "integer", pad="zero")
    poisoned = dict(clean, pool=clean["pool"].at[..., KEY_DIM:].set(jnp.nan))
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    set_path(monkeypatch, path)

    expected = select(clean, max_query_len=4)
    control = select(poisoned, max_query_len=4)
    monkeypatch.setenv(FLAG, "128")
    actual = select(poisoned, max_query_len=4)

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(control, expected)


@pytest.mark.parametrize("chunk", [4, 8])
def test_key_lanes_with_query_windows(monkeypatch, interpret, chunk):
    """Windows smaller than the step (several serial StreamIndex calls)."""
    q_lens, context, tokens = CASES["ragged_1_to_4"]
    args = inputs(5, q_lens, context, tokens, "integer")
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: chunk)
    set_path(monkeypatch, "served")
    expected = select(args, max_query_len=4)
    monkeypatch.setenv(FLAG, "128")
    np.testing.assert_array_equal(select(args, max_query_len=4), expected)


@pytest.mark.parametrize("slot", [1, 7])
def test_key_lanes_read_the_layer_slot(monkeypatch, interpret, slot):
    """Packed compressed keys: the narrow copies keep the slot's pages."""
    q_lens, context, tokens = CASES["verify_k3"]
    args = inputs(11 + slot, q_lens, context, tokens, "integer")
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    set_path(monkeypatch, "served")
    expected = select(args, max_query_len=4, slot=slot)
    monkeypatch.setenv(FLAG, "128")
    np.testing.assert_array_equal(select(args, max_query_len=4, slot=slot), expected)


def test_prefill_keeps_whole_rows(monkeypatch, interpret):
    """The flag is the decode / verify indexer's: prefill copies whole rows."""
    args = inputs(0, [24, 0], [600, 0], 24, "integer")
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )
    monkeypatch.setenv(FLAG, "128")
    jax.clear_caches()
    select(args)
    assert set(COPIES) == {(WIDTH, WIDTH)}


@pytest.mark.parametrize("lanes", ["64", "192", "384"])
def test_rejects_bad_key_lanes(monkeypatch, lanes):
    monkeypatch.setenv(FLAG, lanes)
    q_lens, context, tokens = CASES["verify_k3"]
    with pytest.raises(ValueError, match=FLAG):
        select(inputs(0, q_lens, context, tokens, "integer"), max_query_len=4)


def test_streamindex_rejects_lanes_narrower_than_the_query():
    """The kernel's own check: copies narrower than the query cannot score it."""
    with pytest.raises(ValueError, match="key_lanes"):
        streamindex.streamindex_topk(
            jnp.zeros((4, 4, 256), jnp.bfloat16),
            jnp.zeros((4, 4), jnp.float32),
            jnp.zeros((8, 16, 2, 256), jnp.bfloat16),
            jnp.zeros((4,), jnp.int32),
            jnp.zeros((4 * 8,), jnp.int32),
            jnp.arange(5, dtype=jnp.int32),
            jnp.full((3,), 4, jnp.int32),
            k=16,
            compression_ratio=RATIO,
            num_kv_pages_per_block=8,
            num_queries_per_block=1,
            components=indexer_components.qsa_indexing(key_lanes=128),
        )


@pytest.mark.parametrize("lanes", [0, 128])
def test_served_fp8_carrier_lowers_with_the_lane_tag(monkeypatch, lanes):
    """The served call (FP8 carrier, 1664-token pool pages, 3328-key blocks,
    pool-page table and DMA, mixed pass only) lowers for TPU7x; the flagged
    kernel carries "-l128" in its name (its profile engagement mark)."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for name, value in PATHS["served"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "3328")
    monkeypatch.setenv(FLAG, str(lanes))
    shape = jax.ShapeDtypeStruct
    rows, requests = 64, 16
    args = (
        shape((rows, 4, KEY_DIM), jnp.bfloat16),
        shape((96, 1664, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((rows,), jnp.int32),
        shape((requests, 64), jnp.int32),
        shape((rows,), jnp.int32),
        shape((requests + 1,), jnp.int32),
        shape((requests,), jnp.int32),
    )
    select_fn = jax.jit(
        functools.partial(
            pooled_qsa.select_blocks, ratio=RATIO, block_topk=512, max_query_len=4
        )
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        text = select_fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "StreamIdxTC-m-bq_4-bkvp_104" in text
    assert ("StreamIdxTC-m-bq_4-bkvp_104-l128" in text) == (lanes == 128)
