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


# ---------------------------------------------------------------------------
# Main-KV DCP-2 (c32): the replicated prefill graph's token-sharded op
# (build_pooled_qsa_token_shard_dcp_op: per-shard token_select_dcp -> gather
# -> shard_attend_dcp merged) against the plain DCP op (pooled_qsa_dcp_step)
# on the same rank pool. Everything must match bit for bit: pool after the
# step, selection, partial out and lse.

DCP_SLOTS = 24  # max_num_seqs at c32
DCP_B, DCP_DH, DCP_H, DCP_INDEX_HEADS, DCP_ROTARY = 512, 256, 3, 4, 64
DCP_TOPK = BLOCK_TOPK * RATIO  # tokens
DCP_DECODE_SEATS = DCP_SLOTS - 1
DCP_DECODE_ROWS = 4  # MTP verify rows per seat


def _dcp_harness():
    import os
    import sys

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen4_exp", "dcp2")
    if here not in sys.path:
        sys.path.insert(0, here)
    import dcp2_harness

    return dcp2_harness


def _dcp_step(rows, context, seed=0):
    h = _dcp_harness()
    rng = np.random.default_rng(seed)
    prefill = rows - DCP_DECODE_SEATS * DCP_DECODE_ROWS
    assert prefill > 0
    q_lens = np.array([DCP_DECODE_ROWS] * DCP_DECODE_SEATS + [prefill], np.int64)
    contexts = np.concatenate(
        [rng.integers(4_000, 60_000, DCP_DECODE_SEATS), [context]]
    ).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    seq = contexts + q_lens
    comp_need = [int(-(-n // DCP_B)) for n in seq]
    cols = max(comp_need)
    full_need = sum(comp_need)
    num_pages = full_need + -(-full_need // 2) + len(seq) + full_need + len(seq) + 8
    layout = h.build_layout(rng, seq, DCP_B, num_pages)
    free = iter(layout.free_pages)
    comp = np.zeros((len(seq), cols), np.int32)
    for r, n in enumerate(comp_need):
        comp[r, :n] = [next(free) for _ in range(n)]
    ring = np.array([next(free) for _ in range(len(seq))], np.int32)
    keep = sorted({0, *comp.ravel().tolist(), *ring.tolist()})
    full = np.asarray(h.raw(h.random_fp8(rng, (num_pages, DCP_B, 1, 4, DCP_DH))))
    positions = np.concatenate(
        [contexts[r] + np.arange(q_lens[r]) for r in range(len(seq))]
    ).astype(np.int32)

    def normal(*shape, scale=1.0):
        return jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16)

    pair_qgate = normal(rows, 2 * 2 * DCP_H * DCP_DH)
    shared = (
        normal(rows, DCP_DH), normal(rows, DCP_DH),
        normal(rows, (DCP_INDEX_HEADS + 1) * HEAD_DIM),
        normal(DCP_DH, scale=0.1), normal(DCP_DH, scale=0.1),
        normal(HEAD_DIM, scale=0.1), normal(HEAD_DIM, scale=0.1),
        jnp.asarray(positions),
        jnp.pad(normal(MAX_MODEL_LEN, DCP_ROTARY), ((0, 0), (0, 128 - DCP_ROTARY))),
    )
    meta = (jnp.asarray(starts), jnp.asarray(contexts, jnp.int32), jnp.asarray(ring),
            jnp.asarray(layout.dcp_table), jnp.asarray(comp))
    pools = [
        h.from_raw(h.poison_tail_rows(
            h.project_rank_pool(full, layout, r, keep_pages=keep), layout, r, seq))
        for r in range(2)
    ]
    return pair_qgate, shared, meta, pools, int(prefill)


def _dcp_args(pair_qgate, shared, meta, pool):
    starts, context, ring, table, comp = meta
    return (pair_qgate, *shared, pool, starts, context, ring, table, comp)


@functools.cache
def _dcp_fns(max_query_len, donate=False):
    """(plain, select, attend); ``donate`` donates the pool as the served ops do
    (timing: otherwise every call copies the whole pool)."""
    sel = dict(compress_ratio=RATIO, token_topk=DCP_TOPK, norm_eps=1e-6,
               rotary_dim=DCP_ROTARY, main_slot=0, max_query_len=max_query_len)
    pool_arg = dict(donate_argnums=(10,)) if donate else {}
    plain = jax.jit(functools.partial(pooled_qsa.pooled_qsa_dcp_step, **sel), **pool_arg)
    select = jax.jit(functools.partial(
        pooled_qsa.pooled_qsa_token_select_dcp_step, shard_size=TP, **sel), **pool_arg)
    attend = jax.jit(functools.partial(
        pooled_qsa.pooled_qsa_shard_attend_dcp_step, kv_heads=1, compress_ratio=RATIO,
        token_topk=DCP_TOPK, main_slot=0, max_query_len=max_query_len, merged=True),
        **(dict(donate_argnums=(7,)) if donate else {}))
    return plain, select, attend


def _time_chained_us(fn, args, pool_index, reps=5):
    """Median us per call with the pool donated and threaded through calls
    (the step's writes are idempotent, so every call does the same work)."""
    args = list(args)
    args[pool_index] = jnp.copy(args[pool_index])
    samples = []
    for i in range(2 + 3):
        start = time.perf_counter()
        out = None
        for _ in range(reps):
            out = fn(*args)
            args[pool_index] = out[0]
        jax.block_until_ready(out)
        if i >= 2:
            samples.append((time.perf_counter() - start) / reps * 1e6)
    return float(np.median(samples))


@pytest.mark.tpu_test
@pytest.mark.parametrize("context", [108_000, 233_000])
@pytest.mark.parametrize("rows", [256, 512, 3072])
def test_replicated_token_shard_dcp2_matches_plain_dcp(rows, context):
    h = _dcp_harness()
    pair_qgate, shared, meta, pools, prefill = _dcp_step(rows, context)
    plain, select, attend = _dcp_fns(prefill)
    t_plain_fn, t_select_fn, t_attend_fn = _dcp_fns(prefill, donate=True)
    starts, ctx, _, table, _ = meta
    checks, times = {}, {}
    for r in range(2):
        rank = jnp.asarray([r], jnp.int32)
        args = _dcp_args(pair_qgate, shared, meta, pools[r])
        p_pool, p_out, p_lse, p_sel = plain(*args, rank)
        shards = [select(*args, jnp.asarray([s], jnp.int32), rank) for s in range(TP)]
        s_pool, q, gate, lengths, req, st, _ = shards[0]
        gathered = jnp.concatenate([x[-1] for x in shards])[:rows]
        a_pool, a_out, a_lse, a_sel = attend(
            q, gate, lengths, req, st, table, ctx, s_pool, gathered, rank)
        checks[r] = dict(
            pool=bool(np.array_equal(h.raw(p_pool), h.raw(a_pool))),
            select_pool=bool(np.array_equal(h.raw(p_pool), h.raw(s_pool))),
            selected=bool(np.array_equal(np.asarray(p_sel), np.asarray(a_sel))),
            out=bool(np.array_equal(np.asarray(p_out), np.asarray(a_out))),
            lse=bool(np.array_equal(np.asarray(p_lse), np.asarray(a_lse))),
            out_max_abs=float(np.nanmax(np.abs(
                np.asarray(p_out, np.float32) - np.asarray(a_out, np.float32)))),
        )
        t_plain = _time_chained_us(t_plain_fn, (*args, rank), 10)
        t_sel = [_time_chained_us(t_select_fn, (*args, jnp.asarray([s], jnp.int32), rank), 10)
                 for s in range(TP)]
        t_att = _time_chained_us(
            t_attend_fn, (q, gate, lengths, req, st, table, ctx, s_pool, gathered, rank), 7)
        times[r] = (t_plain, max(t_sel), t_att)
    t_gather = _all_gather_us(rows) or 0.0
    t_plain = max(t[0] for t in times.values())
    t_rp = max(t[1] + t[2] for t in times.values()) + t_gather
    result = dict(
        rows=rows, context=context, slots=DCP_SLOTS, checks=checks,
        plain_dcp_us=round(t_plain, 1),
        rp_select_max_us=round(max(t[1] for t in times.values()), 1),
        rp_attend_us=round(max(t[2] for t in times.values()), 1),
        all_gather_us=round(t_gather, 1),
        saved_us_per_layer=round(t_plain - t_rp, 1),
        saved_us_per_step=round(QSA_LAYERS * (t_plain - t_rp), 1),
    )
    print("QSA_DCP2_RESULT " + json.dumps(result), flush=True)
    for r, c in checks.items():
        assert all(c[k] for k in ("pool", "selected", "out", "lse")), (r, c)


# ---------------------------------------------------------------------------
# Diagnostic: where the RP + DCP-2 selection differs from the plain DCP op.
# Every StreamIndex call of the step is recorded twice: the served top-k ids
# and a top-(k + 16) with scores, so a flipped group's score in both paths and
# the scores at the cut can be read back.


def _bf16_ulp(x):
    x = np.abs(np.asarray(x, np.float64))
    return np.where(x > 0, 2.0 ** (np.floor(np.log2(np.maximum(x, 1e-30))) - 7), 0.0)


def _record_indexer(monkeypatch, calls):
    from jax.experimental import io_callback

    original = pooled_qsa.streamindex_topk

    def store(main, ids, scores):
        calls.append((np.asarray(main), np.asarray(ids), np.asarray(scores)))

    def wrapped(*args, **kwargs):
        main = original(*args, **kwargs)
        if kwargs.get("return_scores"):
            return main
        k = kwargs["k"]
        ids, scores = original(
            *args, **{**kwargs, "k": k + 16, "return_scores": True, "enable_early_exit": False}
        )
        if scores.dtype == jnp.int32:
            scores = jax.lax.bitcast_convert_type(scores, jnp.float32)
        io_callback(store, None, main, ids, scores.astype(jnp.float32), ordered=True)
        return main

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", wrapped)


def _rows_of(calls):
    main = np.concatenate([c[0] for c in calls])
    ids = np.concatenate([c[1] for c in calls])
    scores = np.concatenate([c[2] for c in calls])
    return main, ids, scores


@pytest.mark.tpu_test
@pytest.mark.parametrize("served_mql", [False, True], ids=["mql_prefill", "mql_served4"])
def test_dcp2_3072_108k_selection_dump(monkeypatch, served_mql):
    import os

    rows, context = 3072, 108_000
    pair_qgate, shared, meta, pools, prefill = _dcp_step(rows, context)
    mql = 4 if served_mql else prefill
    _dcp_fns.cache_clear()
    full_calls, shard_calls = [], []
    _record_indexer(monkeypatch, full_calls)
    plain, select, attend = _dcp_fns(mql)
    starts, ctx, _, table, _ = meta
    report = dict(rows=rows, context=context, max_query_len=mql, ranks={})
    for r in range(2):
        rank = jnp.asarray([r], jnp.int32)
        args = _dcp_args(pair_qgate, shared, meta, pools[r])
        full_calls.clear()
        p_pool, p_out, p_lse, p_sel = jax.block_until_ready(plain(*args, rank))
        jax.effects_barrier()
        full = _rows_of(full_calls)
        per_shard = []
        for s in range(TP):
            full_calls.clear()
            per_shard.append(jax.block_until_ready(
                select(*args, jnp.asarray([s], jnp.int32), rank)))
            jax.effects_barrier()
            shard_calls.append(_rows_of(full_calls))
        shard = tuple(np.concatenate([c[i] for c in shard_calls[-TP:]]) for i in range(3))
        s_pool, q, gate, lengths, req, st, _ = per_shard[0]
        gathered = jnp.concatenate([x[-1] for x in per_shard])[:rows]
        _, a_out, a_lse, a_sel = attend(q, gate, lengths, req, st, table, ctx, s_pool,
                                        gathered, rank)
        p_sel, a_sel = np.asarray(p_sel), np.asarray(a_sel)
        sel_diff = {}
        for row in range(rows):
            a, b = set(p_sel[row][p_sel[row] >= 0]), set(a_sel[row][a_sel[row] >= 0])
            if a != b:
                sel_diff[row] = len(a ^ b)
        # Group level: the served top-k of each path, scores from the k+16 dumps.
        f_main, f_ids, f_sc = full
        s_main, s_ids, s_sc = shard
        assert len(f_main) == rows and len(s_main) == rows, (f_main.shape, s_main.shape)
        k = f_main.shape[1]
        flips, max_delta, max_delta_ulp, worst = [], 0.0, 0.0, None
        for row in range(rows):
            fd = {int(g): float(v) for g, v in zip(f_ids[row], f_sc[row]) if g >= 0}
            sd = {int(g): float(v) for g, v in zip(s_ids[row], s_sc[row]) if g >= 0}
            for g in fd.keys() & sd.keys():
                d = abs(fd[g] - sd[g])
                if d > max_delta:
                    max_delta, worst = d, (row, g, fd[g], sd[g])
                u = float(_bf16_ulp(fd[g]))
                if u and d / u > max_delta_ulp:
                    max_delta_ulp = d / u
            fm = set(int(g) for g in f_main[row] if g >= 0)
            sm = set(int(g) for g in s_main[row] if g >= 0)
            if fm != sm:
                fs, ss = sorted(fd.values(), reverse=True), sorted(sd.values(), reverse=True)
                cut = lambda v: [v[k - 1] if len(v) >= k else None,
                                 v[k] if len(v) > k else None]
                flips.append(dict(
                    row=row,
                    only_full=[(g, fd.get(g), sd.get(g)) for g in sorted(fm - sm)],
                    only_shard=[(g, fd.get(g), sd.get(g)) for g in sorted(sm - fm)],
                    full_cut_k_k1=cut(fs), shard_cut_k_k1=cut(ss),
                ))
        report["ranks"][r] = dict(
            selected_rows_differing=len(sel_diff),
            selected_entries_differing=int(sum(sel_diff.values())),
            selected_diff_rows=sorted(sel_diff)[:64],
            group_rows_flipped=len(flips),
            max_abs_score_delta=max_delta,
            max_delta_in_bf16_ulps=max_delta_ulp,
            worst=worst,
            out_equal=bool(np.array_equal(np.asarray(p_out), np.asarray(a_out))),
            flips=flips[:40],
            indexer_calls=dict(full=len(full_calls), shard_chunks=len(shard_calls[-1][0])),
        )
    text = json.dumps(report, default=float)
    out_dir = os.environ.get("CDK_OUTPUT_DIR")
    if out_dir:
        with open(os.path.join(out_dir, f"dcp2_dump_mql{mql}.json"), "w") as f:
            f.write(text)
    summary = {r: {k2: v for k2, v in d.items() if k2 != "flips"}
               for r, d in report["ranks"].items()}
    print("QSA_DCP2_DUMP " + json.dumps(dict(max_query_len=mql, ranks=summary),
                                        default=float), flush=True)
    for r, d in report["ranks"].items():
        for f in d["flips"][:5]:
            print("QSA_DCP2_FLIP", mql, r, json.dumps(f, default=float), flush=True)
