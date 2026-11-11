# SPDX-License-Identifier: Apache-2.0
"""QSA prefill selection at served shapes: token shard and ranged bitset.

Two default-off levers on the replicated prefill graph's selection, checked
bit for bit against the default path on TPU at the c16 shapes (16 request
slots, max_model_len 262144, rows 256 / 512 / 3072, contexts 108K / 233K):

- VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD: each TP rank runs the
  indexer (StreamIndex scoring + SparseCore top-k) on its 1/8 of the rows;
  the concatenated shards must select what the whole step selects, and the
  selection bitset built from them must be identical.
- VLLM_TPU_QSA_PREFILL_BITSET_RANGED: the ranged bitset kernel must equal
  the full compare kernel on the indexer's real selections.

Each case also times both arms (per QSA layer) and prints one
``QSA_SELECT_RESULT`` JSON line.
"""

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    _selection_bitset,
)

RATIO = 4
BLOCK_TOPK = 512  # indexer_budget 2048 / ratio 4
HEADS, HEAD_DIM, WIDTH = 4, 128, 256
PAGE_TOKENS = 128
MAX_MODEL_LEN = 262144
SLOTS = 16  # max_num_seqs at c16
TP = 8
QSA_LAYERS = 12
QUERY_BLOCK = 256
# Verify-style rows ahead of the prefill chunk: (rows, context).
DECODES = ((4, 61_000), (4, 93_500), (4, 120_003), (4, 30_001))


def _step(rows, context, seed=0):
    rng = np.random.default_rng(seed)
    q_lens = [n for n, _ in DECODES] + [rows - sum(n for n, _ in DECODES)]
    contexts = [c for _, c in DECODES] + [context]
    q_lens += [0] * (SLOTS - len(q_lens))
    contexts += [0] * (SLOTS - len(contexts))
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    req = np.zeros(rows, np.int32)
    lengths = np.zeros(rows, np.int32)
    for r in range(SLOTS):
        begin, end = starts[r], starts[r + 1]
        req[begin:end] = r
        lengths[begin:end] = contexts[r] + np.arange(1, end - begin + 1)
    columns = MAX_MODEL_LEN // PAGE_TOKENS
    need = [-(-(c + n) // PAGE_TOKENS) for c, n in zip(contexts, q_lens)]
    pages = sum(need) + 1
    order = rng.permutation(pages)
    table = np.zeros((SLOTS, columns), np.int32)
    used = 0
    for r, n in enumerate(need):
        table[r, :n] = order[used : used + n]
        used += n
    key_pool, key_query = jax.random.split(jax.random.PRNGKey(seed))
    pool = jax.random.normal(
        key_pool, (pages, PAGE_TOKENS, 1, 2, WIDTH), jnp.bfloat16
    )
    query = jax.random.normal(key_query, (rows, HEADS, HEAD_DIM), jnp.bfloat16)
    return dict(
        query=query,
        pool=pool,
        lengths=jnp.asarray(lengths),
        table=jnp.asarray(table),
        req=jnp.asarray(req),
        starts=jnp.asarray(starts),
        context=jnp.asarray(contexts, jnp.int32),
    )


@functools.cache
def _select():
    return jax.jit(
        functools.partial(pooled_qsa.select_blocks, ratio=RATIO, block_topk=BLOCK_TOPK)
    )


@functools.cache
def _shard_select(width):
    def run(query, pool, lengths, table, req, starts, context, index):
        base, sub_starts, sub_context = pooled_qsa.token_shard_step(
            starts, context, index, width
        )

        def own(x):
            return jax.lax.dynamic_slice_in_dim(x, base, width)

        # As pooled_qsa's token_shard branch (max_query_len=1).
        return pooled_qsa.select_blocks(
            own(query), pool, own(lengths), table, own(req), sub_starts,
            sub_context, ratio=RATIO, block_topk=BLOCK_TOPK, max_query_len=1,
        )

    return jax.jit(run)


@functools.cache
def _bitset(ranged):
    return jax.jit(
        functools.partial(
            _selection_bitset,
            ratio=RATIO,
            groups=MAX_MODEL_LEN // RATIO,
            words=MAX_MODEL_LEN // (RATIO * 32),
            query_block=QUERY_BLOCK,
            width=QUERY_BLOCK,
            ranged=ranged,
        )
    )


def _time_us(fn, *args, reps=20):
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        out = None
        for _ in range(reps):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / reps * 1e6)
    return float(np.median(samples))


def _all_gather_us(rows):
    """One [rows / TP, k] int32 all-gather over the local devices."""
    devices = jax.devices()
    if len(devices) < 2:
        return None
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    try:
        shard_map = jax.shard_map
    except AttributeError:  # older JAX
        from jax.experimental.shard_map import shard_map
    n = min(len(devices), TP)
    mesh = Mesh(np.asarray(devices[:n]), ("x",))
    width = rows // TP
    x = jax.device_put(
        jnp.zeros((n * width, BLOCK_TOPK), jnp.int32), NamedSharding(mesh, P("x"))
    )
    gather = jax.jit(
        shard_map(
            lambda a: jax.lax.all_gather(a, "x", axis=0, tiled=True)[None],
            mesh=mesh,
            in_specs=P("x"),
            out_specs=P("x"),
        )
    )
    return _time_us(gather, x, reps=50)


def _ascending_fraction(blocks, lengths):
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    good = 0
    for row, length in enumerate(lengths):
        count = min(BLOCK_TOPK, length // RATIO)
        good += bool(np.all(np.diff(blocks[row, :count]) > 0))
    return good / len(lengths)


@pytest.mark.tpu_test
@pytest.mark.parametrize("context", [108_000, 233_000])
@pytest.mark.parametrize("rows", [256, 512, 3072])
def test_prefill_selection_levers_match_default(rows, context):
    args = _step(rows, context)
    operands = [args[k] for k in ("query", "pool", "lengths", "table", "req",
                                  "starts", "context")]
    full = _select()(*operands)
    width = rows // TP
    shard_fn = _shard_select(width)
    shards = [
        shard_fn(*operands, jnp.asarray([s], jnp.int32)) for s in range(TP)
    ]
    gathered = jnp.concatenate(shards)

    # (A) Token shard: same selection, same bitset.
    full_np, gathered_np = np.asarray(full), np.asarray(gathered)
    same_order = bool(np.array_equal(full_np, gathered_np))
    same_sets = bool(
        np.array_equal(np.sort(full_np, axis=1), np.sort(gathered_np, axis=1))
    )
    bits_full = _bitset(False)(full, args["lengths"])
    bits_gathered = _bitset(False)(gathered, args["lengths"])
    np.testing.assert_array_equal(np.asarray(bits_gathered), np.asarray(bits_full))
    assert same_sets

    # (B) Ranged bitset on the indexer's selections.
    bits_ranged = _bitset(True)(full, args["lengths"])
    np.testing.assert_array_equal(np.asarray(bits_ranged), np.asarray(bits_full))

    t_full = _time_us(_select(), *operands)
    t_shards = [
        _time_us(shard_fn, *operands, jnp.asarray([s], jnp.int32))
        for s in range(TP)
    ]
    t_gather = _all_gather_us(rows)
    t_bits = _time_us(_bitset(False), full, args["lengths"])
    t_ranged = _time_us(_bitset(True), full, args["lengths"])
    shard_cost = max(t_shards) + (t_gather or 0.0)
    result = dict(
        rows=rows,
        context=context,
        same_order=same_order,
        same_sets=same_sets,
        ascending_rows=_ascending_fraction(full, args["lengths"]),
        indexer_full_us=round(t_full, 1),
        indexer_shard_us=[round(t, 1) for t in t_shards],
        all_gather_us=None if t_gather is None else round(t_gather, 1),
        devices=len(jax.devices()),
        indexer_saved_us_per_layer=round(t_full - shard_cost, 1),
        indexer_saved_us_per_step=round(QSA_LAYERS * (t_full - shard_cost), 1),
        bitset_full_us=round(t_bits, 1),
        bitset_ranged_us=round(t_ranged, 1),
        bitset_saved_us_per_step=round(QSA_LAYERS * (t_bits - t_ranged), 1),
    )
    print("QSA_SELECT_RESULT " + json.dumps(result), flush=True)


@functools.cache
def _owned_bitset(ranged, rank, world, per_tile, live_words):
    return jax.jit(
        functools.partial(
            _selection_bitset,
            ratio=RATIO,
            groups=MAX_MODEL_LEN // RATIO,
            words=live_words,
            query_block=QUERY_BLOCK,
            width=QUERY_BLOCK,
            ranged=ranged,
            owned=(rank, world, per_tile, live_words),
        )
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("context", [108_000, 233_000])
@pytest.mark.parametrize("rows", [512, 3072])
def test_ranged_owned_bitset_matches_default(rows, context):
    """DCP-2 (VLLM_TPU_QSA_PREFILL_BITSET_OWNED): each rank's owned-word
    bitset from the ranged kernel equals the full compare kernel's, on the
    indexer's real selections; also times both."""
    args = _step(rows, context)
    operands = [args[k] for k in ("query", "pool", "lengths", "table", "req",
                                  "starts", "context")]
    full = _select()(*operands)
    world, kv_tile = 2, 1024
    per_tile = kv_tile // (RATIO * 32)
    global_words = MAX_MODEL_LEN // (RATIO * 32)
    live_words = ((global_words // world + 127) // 128) * 128
    timings = {}
    for rank in range(world):
        a = _owned_bitset(False, rank, world, per_tile, live_words)
        b = _owned_bitset(True, rank, world, per_tile, live_words)
        np.testing.assert_array_equal(
            np.asarray(b(full, args["lengths"])), np.asarray(a(full, args["lengths"]))
        )
        timings[rank] = (
            round(_time_us(a, full, args["lengths"]), 1),
            round(_time_us(b, full, args["lengths"]), 1),
        )
    print("QSA_OWNED_RESULT " + json.dumps(
        dict(rows=rows, context=context, full_vs_ranged_us=timings)), flush=True)
