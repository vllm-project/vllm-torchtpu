# SPDX-License-Identifier: Apache-2.0
"""q38 BLSTACK-IDX (VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD) at the step level.

1. ``pooled_qsa_token_select_compressed_dcp_step(decode_rows=True)`` on each
   token shard writes exactly the unsharded owned-group select step's caches
   and returns its q / gate / lengths / req / starts, and hands
   ``select_blocks`` the per-row inputs of the unsharded call with the decode
   routing (``max_query_len`` kept). ``select_blocks`` is replaced by a
   fingerprint of every row's own inputs (query bits, causal length from the
   re-based starts / context, length, request table row, routing), so a shard
   whose rows saw anything else than in the whole step fails here.
   ``test_q38_blstack_idx_row_shard_indexer.py`` proves on the real StreamIndex
   kernel that those per-row inputs fix the row's local lists.
2. TP = S * D end to end: every simulated TP rank (shard ``r // D``, DCP rank
   ``r % D``) runs its shard's select step, the real ``_gathered_lists``
   rebuilds the ``[D, T, k]`` lists from one TP all-gather (packed and not),
   and the real merge-and-attend step runs on them; pool, output, LSE and
   ``selected`` equal the unsharded compressed-DCP path's bit for bit, with
   and without VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from dcp2_harness import D, from_raw, poison_tail_rows, project_rank_pool, rank_operand, raw
from test_dcp2_compressed import lax_topk, stub_select_dcp
from test_dcp2_op import (  # noqa: F401 (fixture)
    K_SCALE,
    RATIO,
    ROTARY,
    TOPK,
    V_SCALE,
    _args,
    make_step,
    partial_kernels,
)

from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa
from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

EMPTY = int(np.iinfo(np.int32).max)

CASES = {
    # Verify rows of 4 per seat, an idle seat, contexts across B and 2B.
    "verify": dict(query_lens=[4, 4, 0, 4, 4, 4, 4, 4], contexts=[37, 509, 0, 1022, 2046,
                   1535, 3, 4100]),
    # A seat cut by the 2-shard boundary (rows 14..17 across 16), ragged rows.
    "cut": dict(query_lens=[4, 4, 3, 3, 4, 1, 4, 3], contexts=[1020, 511, 1301, 64, 700,
                5, 2047, 4093]),
    # The second shard holds only padding.
    "padding": dict(query_lens=[4, 1, 0, 0, 0, 0, 0, 0], contexts=[3001, 999, 0, 0, 0, 0,
                    0, 0]),
}


def _bits(x):
    """Raw bits of an array (NaN-safe exact comparison for any float dtype)."""
    a = np.asarray(x)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _mix(*xs):
    h = jnp.uint32(2166136261)
    for x in xs:
        h = (h ^ jnp.asarray(x).astype(jnp.uint32)) * jnp.uint32(16777619)
    return h


def fingerprint_select(log):
    """``select_blocks`` stand-in: each row's (vals, pos) is a hash of that
    row's own inputs only; dead rows (past ``starts[-1]``) are empty."""

    def select(query, pool, lengths, table, req, starts, context, *, ratio, block_topk,
               slot=0, max_query_len=1, shard=None, dcp=None, after_scoring=None):
        assert dcp is not None and shard is None and after_scoring is None
        rows = len(query)
        decode = rows <= len(context) * max_query_len
        log.append(dict(rows=rows, slots=len(context), max_query_len=max_query_len,
                        decode=bool(decode)))
        t = jnp.arange(rows, dtype=jnp.int32)
        r = jnp.clip(jnp.searchsorted(starts[1:], t, side="right"), 0, len(context) - 1)
        live = t < starts[-1]
        causal = context[r] + (t - starts[r]) + 1
        qbits = jax.lax.bitcast_convert_type(query.astype(jnp.float32), jnp.uint32)
        qsum = jnp.sum(qbits.reshape(rows, -1) * jnp.uint32(2654435761), axis=1)
        tsum = jnp.sum(table[r].astype(jnp.uint32) * jnp.uint32(40503), axis=1)
        rank = jnp.asarray(dcp[0]).reshape(-1)[0]
        f = _mix(qsum, causal, lengths, tsum, req, max_query_len, decode, rank, slot)
        j = jnp.arange(block_topk, dtype=jnp.uint32)
        h = (f[:, None] + j[None] * jnp.uint32(2246822519)) & jnp.uint32(0xFFFFF)
        vals = jnp.where(live[:, None], h.astype(jnp.float32) / 1048576.0, -jnp.inf)
        pos = jnp.where(live[:, None], (h & jnp.uint32(0x7FFFF)).astype(jnp.int32), EMPTY)
        return vals.astype(jnp.float32), pos

    return select


def _opts(max_query_len):
    return dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=0, max_query_len=max_query_len)


def _rank_pool(layout, full, keep, seq, r):
    return from_raw(poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep),
                                     layout, r, seq))


@pytest.mark.parametrize("shards", [2, 4])
@pytest.mark.parametrize("case", sorted(CASES))
def test_shard_select_steps_equal_the_whole_step(interpret, monkeypatch, case, shards):
    log = []
    monkeypatch.setattr(pooled_qsa, "select_blocks", fingerprint_select(log))
    layout, full, keep, qgate, s, m, seq, starts = make_step(5, **CASES[case])
    tokens = len(s["positions"])
    assert tokens % shards == 0, tokens
    live = int(starts[-1])
    pair_qgate = jnp.concatenate(qgate, axis=1)
    for r in range(D):
        pool = _rank_pool(layout, full, keep, seq, r)
        whole = pooled_qsa.pooled_qsa_compressed_dcp_select_step(
            *_args(pair_qgate, s, pool, m, layout.dcp_table), rank_operand(r), dcp_size=D,
            **_opts(4))
        log.clear()
        parts = [
            pooled_qsa.pooled_qsa_token_select_compressed_dcp_step(
                *_args(pair_qgate, s, pool, m, layout.dcp_table), rank_operand(k),
                rank_operand(r), shard_size=shards, dcp_size=D, decode_rows=True,
                **_opts(4))
            for k in range(shards)
        ]
        # Every shard took the decode indexer on its T / S rows.
        assert [x["rows"] for x in log] == [tokens // shards] * shards, log
        assert all(x["decode"] and x["max_query_len"] == 4 for x in log), log
        for k, part in enumerate(parts):
            for i, name in ((0, "pool"), (1, "q"), (2, "gate"), (3, "lengths"), (4, "req"),
                            (5, "starts")):
                np.testing.assert_array_equal(_bits(part[i]), _bits(whole[i]),
                                              f"rank {r} shard {k}: {name} differs")
        got_v = np.concatenate([np.asarray(p[6]) for p in parts])
        got_p = np.concatenate([np.asarray(p[7]) for p in parts])
        np.testing.assert_array_equal(_bits(got_v), _bits(whole[6]),
                                      f"rank {r}: local list values differ")
        np.testing.assert_array_equal(got_p, np.asarray(whole[7]),
                                      f"rank {r}: local list group ids differ")
        assert np.all(got_p[:live] != EMPTY) and np.all(got_p[live:] == EMPTY)
    print(f"Q38_BLIDX STEP_EQUAL case={case} shards={shards} tokens={tokens} live={live}")


def test_flag_off_token_select_keeps_the_prefill_routing(interpret, monkeypatch):
    """``decode_rows`` off (the prefill sequence-parallel token shard, Benliang's
    path): the shard is scored with max_query_len 1, as before the port."""
    log = []
    monkeypatch.setattr(pooled_qsa, "select_blocks", fingerprint_select(log))
    layout, full, keep, qgate, s, m, seq, starts = make_step(5, **CASES["verify"])
    pool = _rank_pool(layout, full, keep, seq, 0)
    pooled_qsa.pooled_qsa_token_select_compressed_dcp_step(
        *_args(jnp.concatenate(qgate, axis=1), s, pool, m, layout.dcp_table),
        rank_operand(1), rank_operand(0), shard_size=2, dcp_size=D, **_opts(4))
    assert [x["max_query_len"] for x in log] == [1], log


def test_row_shard_select_logs_each_traced_shape(interpret, monkeypatch):
    """Engagement: the decode-row select step logs once per traced call."""
    # The submodule (the package attribute `logger` is a logger object).
    logger_mod = importlib.import_module("vllm_torchtpu.logger")

    lines = []

    class Recorder:
        def info(self, msg, *args):
            lines.append(msg % args)

    monkeypatch.setattr(logger_mod, "init_logger", lambda name: Recorder())
    monkeypatch.setattr(pooled_qsa, "select_blocks", fingerprint_select([]))
    layout, full, keep, qgate, s, m, seq, starts = make_step(5, **CASES["verify"])
    pool = _rank_pool(layout, full, keep, seq, 1)
    args = _args(jnp.concatenate(qgate, axis=1), s, pool, m, layout.dcp_table)
    pooled_qsa.pooled_qsa_token_select_compressed_dcp_step(
        *args, rank_operand(1), rank_operand(1), shard_size=2, dcp_size=D, decode_rows=True,
        **_opts(4))
    assert len(lines) == 1 and "row-shard select traced: tokens 32" in lines[0], lines
    lines.clear()
    pooled_qsa.pooled_qsa_token_select_compressed_dcp_step(
        *args, rank_operand(1), rank_operand(1), shard_size=2, dcp_size=D, **_opts(4))
    assert lines == [], lines  # the prefill token shard logs nothing
    print("Q38_BLIDX ENGAGE_TRACE_LINE ok")


def test_plain_step_is_the_compressed_select_step(interpret, monkeypatch):
    """The plain op's step ignores its shard operand: same outputs as the
    compressed-DCP select step for any operand value."""
    monkeypatch.setattr(pooled_qsa, "select_blocks", fingerprint_select([]))
    layout, full, keep, qgate, s, m, seq, starts = make_step(7, **CASES["cut"])
    pool = _rank_pool(layout, full, keep, seq, 1)
    args = _args(jnp.concatenate(qgate, axis=1), s, pool, m, layout.dcp_table)
    want = pooled_qsa.pooled_qsa_compressed_dcp_select_step(
        *args, rank_operand(1), dcp_size=D, **_opts(4))
    for operand in (0, 1, 3):
        got = pooled_qsa.pooled_qsa_row_plain_compressed_dcp_step(
            *args, rank_operand(operand), rank_operand(1), dcp_size=D, **_opts(4))
        assert len(got) == len(want) == 8
        for a, b in zip(got, want):
            np.testing.assert_array_equal(_bits(a), _bits(b))


class _GatherAll:
    """A TP group whose all_gather returns every simulated rank's input in rank
    order, call by call: a recording pass per rank fills ``inputs[rank]``."""

    def __init__(self, world):
        self.world_size, self.inputs, self.rank, self.calls = world, {}, None, 0

    def all_gather(self, x, dim=0):
        assert dim == 0
        i, self.calls = self.calls, self.calls + 1
        if self.rank is not None:  # recording pass: any well-shaped result
            self.inputs.setdefault(self.rank, []).append(x.clone())
            return torch.cat([x] * self.world_size, dim=0)
        return torch.cat([self.inputs[r][i] for r in range(self.world_size)], dim=0)


def _gather(vals_by_rank, pos_by_rank, world, size, tokens):
    """The real ``_gathered_lists`` on every simulated rank's lists."""
    group = _GatherAll(world)
    for r in range(world):
        group.rank, group.calls = r, 0
        qsa_op._gathered_lists(torch.from_numpy(vals_by_rank[r]),
                               torch.from_numpy(pos_by_rank[r]), group, size, tokens)
    group.rank, group.calls = None, 0
    v, p = qsa_op._gathered_lists(torch.from_numpy(vals_by_rank[0]),
                                  torch.from_numpy(pos_by_rank[0]), group, size, tokens)
    return jnp.asarray(v.numpy()), jnp.asarray(p.numpy())


@pytest.mark.parametrize("packed", [0, 1])
@pytest.mark.parametrize("local_merge", [False, True])
@pytest.mark.parametrize("case", ["verify", "cut"])
def test_tp_ranks_attend_exactly_as_compressed_dcp(interpret, rpa_stand_in, monkeypatch,
                                                   partial_kernels, case, packed, local_merge):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select_dcp)
    monkeypatch.setattr(dcp_select, "_sc_topk", lax_topk)
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER", str(packed))
    shards = 2
    world = shards * D
    layout, full, keep, qgate, s, m, seq, starts = make_step(3, **CASES[case])
    tokens = len(s["positions"])
    live = int(starts[-1])
    pair_qgate = jnp.concatenate(qgate, axis=1)
    pools = [_rank_pool(layout, full, keep, seq, d) for d in range(D)]
    args = [_args(pair_qgate, s, pools[d], m, layout.dcp_table) for d in range(D)]
    att = dict(kv_heads=1, compress_ratio=RATIO, token_topk=TOPK, k_scale=K_SCALE,
               v_scale=V_SCALE, main_slot=0, max_query_len=4)
    # Unsharded compressed DCP (Benliang's served path): pair gather of whole-step lists.
    whole = [pooled_qsa.pooled_qsa_compressed_dcp_select_step(
        *args[d], rank_operand(d), dcp_size=D, **_opts(4)) for d in range(D)]
    ck_v, ck_p = _gather([np.asarray(w[6]) for w in whole], [np.asarray(w[7]) for w in whole],
                         D, D, tokens)
    # Row shard: TP rank r scores shard r // D over DCP rank r % D's owned groups.
    sharded = {}
    for r in range(world):
        k, d = divmod(r, D)
        sharded[r] = pooled_qsa.pooled_qsa_token_select_compressed_dcp_step(
            *args[d], rank_operand(k), rank_operand(d), shard_size=shards, dcp_size=D,
            decode_rows=True, **_opts(4))
    rs_v, rs_p = _gather([np.asarray(sharded[r][6]) for r in range(world)],
                         [np.asarray(sharded[r][7]) for r in range(world)], world, D, tokens)
    np.testing.assert_array_equal(np.asarray(rs_v)[:, :live].view(np.int32),
                                  np.asarray(ck_v)[:, :live].view(np.int32))
    np.testing.assert_array_equal(np.asarray(rs_p)[:, :live], np.asarray(ck_p)[:, :live])
    for r in range(world):
        k, d = divmod(r, D)
        pool, q, gate, lengths, req, st, _, _ = sharded[r]
        got = pooled_qsa.pooled_qsa_compressed_dcp_attend_step(
            q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"], pool,
            rs_v, rs_p, rank_operand(d), dcp_size=D, local_merge=local_merge, **att)
        pool_w, q_w, gate_w, lengths_w, req_w, st_w, _, _ = whole[d]
        want = pooled_qsa.pooled_qsa_compressed_dcp_attend_step(
            q_w, gate_w, lengths_w, req_w, st_w, jnp.asarray(layout.dcp_table), m["context"],
            pool_w, ck_v, ck_p, rank_operand(d), dcp_size=D, local_merge=local_merge, **att)
        np.testing.assert_array_equal(_bits(got[0]), _bits(want[0]), f"TP rank {r}: pool")
        for i, name in ((1, "out"), (2, "lse"), (3, "selected")):
            np.testing.assert_array_equal(_bits(got[i])[:live], _bits(want[i])[:live],
                                          f"TP rank {r}: {name}")
    print(f"Q38_BLIDX TP_ATTEND_EQUAL case={case} packed={packed} local_merge={int(local_merge)} "
          f"world={world} live={live}")
