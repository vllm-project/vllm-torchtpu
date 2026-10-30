# SPDX-License-Identifier: Apache-2.0
"""End to end: pooled_qsa_dcp_step on both ranks + dcp_combine vs pooled_qsa_step.

Recipe (2d): run ``pooled_qsa_dcp_step`` for rank 0 and rank 1 on
NaN-poisoned pool copies (owned main-KV pages under the 2B table; compressed
keys, rings and the null page replicated), stack (out, lse), merge each TP
rank's heads with ``qsa_rpa.dcp_combine`` and compare with
``pooled_qsa_step``'s gated output for that rank's H heads; selected tokens
must be identical and each rank's pool must equal the projection of the
replicated pool after the step (writes exact).

The indexer cannot run on CPU: ``select_blocks`` is a deterministic stub
shared by both paths. Replicated decode uses the CPU RPA stand-in; replicated
prefill runs the block-sparse kernel in interpret mode. The per-rank partial
attention is either set6's kernels (``production``, skipped until
``rpa_attention`` / ``block_sparse_gqa_attention`` take ``return_lse``) or
the harness reference patched in under ``qsa_rpa.decode_attention`` /
``_prefill_attention`` (``reference``), which exercises 2d's routing,
lse channel and op body today.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    D,
    random_fp8,
    build_layout,
    check_gated,
    from_raw,
    poison_tail_rows,
    project_rank_pool,
    rank_operand,
    raw,
    reference_attend,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)

B, DH, H, INDEX_DIM, INDEX_HEADS, RATIO, ROTARY = 512, 256, 3, 128, 4, 4, 64
TOPK = 256  # tokens (64 groups): keeps CPU time bounded
K_SCALE, V_SCALE = 0.375, 2.5


def stub_select(query, pool, lengths, table, req, starts, context, *, ratio, block_topk,
                **_):
    """Deterministic pseudo-random top-k over each row's complete groups."""
    complete = lengths // ratio
    g = jnp.arange(8192, dtype=jnp.float32)
    score = jnp.sin((g[None] + 1) * 12.9898 + lengths[:, None].astype(jnp.float32) * 78.233)
    score = jnp.where(g[None] < complete[:, None], score, -jnp.inf)
    order = jnp.argsort(-score, axis=1)[:, :block_topk].astype(jnp.int32)
    count = jnp.minimum(complete, block_topk)
    return jnp.where(jnp.arange(block_topk)[None] < count[:, None], order, -1)


def _implemented():
    return ("return_lse" in inspect.signature(qsa_rpa.rpa_attention).parameters
            and "return_lse" in inspect.signature(block_sparse_gqa_attention).parameters)


def _reference_route(original, monkeypatch, name):
    def route(*args, dcp=None, return_lse=False, **kw):
        if dcp is None:
            return original(*args, **kw)
        assert return_lse
        q, gate, blocks, lengths, table, cache = args[:6]
        out, lse = reference_attend(
            q, gate, np.asarray(blocks), np.asarray(lengths), np.asarray(table), cache,
            kv_heads=kw["kv_heads"], ratio=kw["ratio"], k_scale=kw.get("k_scale", 1.0),
            v_scale=kw.get("v_scale", 1.0), kv_head_offset=kw.get("kv_head_offset", 0),
            dcp=dcp)
        return out, lse, cache

    monkeypatch.setattr(qsa_rpa, name, route)


@pytest.fixture(params=["reference", "production"])
def partial_kernels(request, monkeypatch):
    if request.param == "production":
        if not _implemented():
            pytest.skip("set6 return_lse kernels not landed")
        return
    _reference_route(qsa_rpa.decode_attention, monkeypatch, "decode_attention")
    _reference_route(qsa_rpa._prefill_attention, monkeypatch, "_prefill_attention")


def make_step(seed, query_lens, contexts):
    rng = np.random.default_rng(seed)
    query_lens, contexts = np.asarray(query_lens), np.asarray(contexts)
    requests = len(contexts)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    tokens = max(8, -(-int(starts[-1]) // 8) * 8)
    seq = np.maximum(contexts + query_lens, 1)
    cols = int(-(-seq.max() // B))
    num_pages = int(sum(-(-seq // B)) * 2 + requests * (cols + 1) + 4)
    layout = build_layout(rng, seq, B, num_pages)
    free = iter(layout.free_pages)
    comp = np.array([[next(free) for _ in range(cols)] for _ in range(requests)], np.int32)
    ring = np.array([next(free) for _ in range(requests)], np.int32)
    keep = sorted({0, *comp.ravel().tolist(), *ring.tolist()})
    full = random_fp8(rng, (num_pages, B, 1, 4, DH))
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(query_lens[r])

    def normal(*shape, scale=1.0):
        return jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16)

    qgate = [normal(tokens, 2 * H * DH) for _ in range(D)]  # per TP rank of the pair
    shared = dict(
        key=normal(tokens, DH), value=normal(tokens, DH),
        index_qk=normal(tokens, (INDEX_HEADS + 1) * INDEX_DIM),
        nq=normal(DH, scale=0.1), nk=normal(DH, scale=0.1),
        niq=normal(INDEX_DIM, scale=0.1), nik=normal(INDEX_DIM, scale=0.1),
        positions=jnp.asarray(positions),
        trig=jnp.pad(normal(16384, ROTARY), ((0, 0), (0, 128 - ROTARY))),
    )
    meta = dict(starts=jnp.asarray(starts), context=jnp.asarray(contexts, jnp.int32),
                ring_ids=jnp.asarray(ring), compressed_table=jnp.asarray(comp))
    return layout, full, keep, qgate, shared, meta, seq, starts


def _args(qg, s, pool, m, table):
    return (qg, s["key"], s["value"], s["index_qk"], s["nq"], s["nk"], s["niq"], s["nik"],
            s["positions"], s["trig"], pool, m["starts"], m["context"], m["ring_ids"],
            jnp.asarray(table), m["compressed_table"])


def check_op(query_lens, contexts, *, main_slot=0, max_query_len=1, seed=0):
    layout, full, keep, qgate, s, m, seq, starts = make_step(seed, query_lens, contexts)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot,
                max_query_len=max_query_len)
    rep = [pooled_qsa.pooled_qsa_step(*_args(qgate[r], s, full, m, layout.full_table), **opts)
           for r in range(D)]
    full_after = raw(rep[0][0])
    pair_qgate = jnp.concatenate(qgate, axis=1)
    dcp = []
    for r in range(D):
        pool = project_rank_pool(raw(full), layout, r, keep_pages=keep)
        pool = poison_tail_rows(pool, layout, r, seq)
        out = pooled_qsa.pooled_qsa_dcp_step(
            *_args(pair_qgate, s, from_raw(pool), m, layout.dcp_table), rank_operand(r),
            **opts)
        dcp.append(out)
        # Writes inside the op: the rank pool is the replicated pool's projection.
        want = poison_tail_rows(project_rank_pool(full_after, layout, r, keep_pages=keep),
                                layout, r, seq)
        got = raw(out[0])
        assert np.array_equal(got, want), f"rank {r} pool differs after the step"
    np.testing.assert_array_equal(np.asarray(dcp[0][3]), np.asarray(dcp[1][3]))
    out_pair = jnp.stack([jnp.asarray(d[1], jnp.float32) for d in dcp])
    lse_pair = jnp.stack([jnp.asarray(d[2], jnp.float32) for d in dcp])
    tokens = int(starts[-1])
    live = np.zeros((len(out_pair[0]), H), bool)
    live[:tokens] = True
    reports = {}
    for r in range(D):
        np.testing.assert_array_equal(np.asarray(dcp[0][3]), np.asarray(rep[r][2]))
        gate = qgate[r].reshape(len(qgate[r]), H, 2 * DH)[..., DH:]
        got = qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank_operand(r))
        want = np.asarray(rep[r][1], np.float32).reshape(-1, H, DH)
        reports[r] = check_gated(got, want, live, what=f"op rank {r}")
    return reports


OP_CASES = {
    # c8 MTP verify: 4 rows per seat, crossing B and 2B, one idle seat.
    "verify": dict(query_lens=[4, 4, 0, 4, 4, 4, 4, 4], contexts=[37, 509, 0, 1022, 2046,
                   1535, 3, 4100], max_query_len=4),
    # Mixed: decode prefix (2 verify seats) + an unaligned 300-token prefill piece.
    "mixed": dict(query_lens=[4, 4, 300], contexts=[1020, 511, 1301], max_query_len=4),
}


@pytest.mark.parametrize("case", sorted(OP_CASES))
def test_dcp_op_matches_replicated_op(interpret, rpa_stand_in, monkeypatch, partial_kernels,
                                      case):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    print(case, check_op(**OP_CASES[case]))


def check_sp_op(query_lens, contexts, *, main_slot=0, max_query_len=1, seed=0, size=D):
    """SP token-shard pair: token_select on each of ``size`` token shards ->
    gather blocks -> shard_attend, DCP (per pair rank, poisoned pools) vs
    replicated. The pair ranks double as the token shards (shard = rank)."""
    layout, full, keep, qgate, s, m, seq, starts = make_step(seed, query_lens, contexts)
    sel = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
               k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot,
               max_query_len=max_query_len)
    att = dict(kv_heads=1, compress_ratio=RATIO, token_topk=TOPK, k_scale=K_SCALE,
               v_scale=V_SCALE, main_slot=main_slot, max_query_len=max_query_len,
               merged=True)
    # Replicated: every TP rank selects its shard, then attends its H heads.
    rep = []
    for r in range(D):
        shards = [pooled_qsa.pooled_qsa_token_select_step(
            *_args(qgate[r], s, full, m, layout.full_table), rank_operand(sh),
            shard_size=size, **sel) for sh in range(size)]
        pool, q, gate, lengths, req, st, _ = shards[0]
        gathered = jnp.concatenate([x[-1] for x in shards])
        rep.append(pooled_qsa.pooled_qsa_shard_attend_step(
            q, gate, lengths, req, st, jnp.asarray(layout.full_table), m["context"], pool,
            gathered, **att))
    full_after = raw(shards[0][0])
    pair_qgate = jnp.concatenate(qgate, axis=1)
    pools = [from_raw(poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep),
                                       layout, r, seq)) for r in range(D)]
    dcp = []
    for r in range(D):
        shards = [pooled_qsa.pooled_qsa_token_select_dcp_step(
            *_args(pair_qgate, s, pools[r], m, layout.dcp_table), rank_operand(sh),
            rank_operand(r), shard_size=size, **sel) for sh in range(size)]
        pool, q, gate, lengths, req, st, _ = shards[0]
        want = poison_tail_rows(project_rank_pool(full_after, layout, r, keep_pages=keep),
                                layout, r, seq)
        assert np.array_equal(raw(pool), want), f"rank {r} pool differs after SP select"
        gathered = jnp.concatenate([x[-1] for x in shards])
        dcp.append(pooled_qsa.pooled_qsa_shard_attend_dcp_step(
            q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"], pool,
            gathered, rank_operand(r), **att))
    out_pair = jnp.stack([jnp.asarray(d[1], jnp.float32) for d in dcp])
    lse_pair = jnp.stack([jnp.asarray(d[2], jnp.float32) for d in dcp])
    live = np.zeros((out_pair.shape[1], H), bool)
    live[: int(starts[-1])] = True
    reports = {}
    for r in range(D):
        np.testing.assert_array_equal(np.asarray(dcp[r][3]), np.asarray(rep[r][2]))
        gate = qgate[r].reshape(len(qgate[r]), H, 2 * DH)[..., DH:]
        got = qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank_operand(r))
        want = np.asarray(rep[r][1], np.float32).reshape(-1, H, DH)
        reports[r] = check_gated(got, want, live, what=f"SP op rank {r}")
    return reports


SP_CASES = {
    # A prefill-sized SP step: two unaligned pieces crossing B / 2B.
    "prefill_pieces": dict(query_lens=[200, 120], contexts=[1301, 700]),
}


@pytest.mark.parametrize("case", sorted(SP_CASES))
def test_dcp_sp_token_shard_matches_replicated(interpret, rpa_stand_in, monkeypatch,
                                               partial_kernels, case):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    print(case, check_sp_op(**SP_CASES[case]))
