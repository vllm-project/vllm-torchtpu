# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_DCP end to end (indexer stubbed): each rank's
owned-group select step, the pair's gathered local top-k lists, the exact merge
and the DCP attention vs the replicated step.

``select_blocks`` is a deterministic stub (``test_dcp2_op.stub_select``'s
scores); its DCP form returns each rank's local top-k over its owned groups as
``(vals, pos)``, so the merge must rebuild exactly the replicated top-k. Gated
outputs within the combine's rounding; ``selected`` equal as a set per row.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    D,
    ON_TPU,
    check_gated,
    from_raw,
    poison_tail_rows,
    project_rank_pool,
    rank_operand,
    raw,
)
from test_dcp2_op import (
    DH,
    H,
    K_SCALE,
    OP_CASES,
    RATIO,
    ROTARY,
    TOPK,
    V_SCALE,
    _args,
    make_step,
    partial_kernels,  # noqa: F401 (fixture)
    stub_select,
)

from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa, qsa_rpa


def _scores(lengths):
    g = jnp.arange(8192, dtype=jnp.float32)
    return jnp.sin((g[None] + 1) * 12.9898 + lengths[:, None].astype(jnp.float32) * 78.233)


def stub_select_dcp(query, pool, lengths, table, req, starts, context, *, ratio, block_topk,
                    dcp=None, **kw):
    """``stub_select``'s scores; with ``dcp`` the rank's local top-k over its
    owned complete groups as ``(vals, pos)``."""
    if dcp is None:
        return stub_select(query, pool, lengths, table, req, starts, context, ratio=ratio,
                           block_topk=block_topk, **kw)
    rank, size = dcp
    page = pool.shape[1]
    g = jnp.arange(8192, dtype=jnp.int32)
    owned = ((g * ratio) // page) % size == jnp.asarray(rank).reshape(-1)[0]
    valid = (g[None] < (lengths // ratio)[:, None]) & owned[None]
    score = jnp.where(valid, _scores(lengths), -jnp.inf)
    order = jnp.argsort(-score, axis=1)[:, :block_topk].astype(jnp.int32)
    vals = jnp.take_along_axis(score, order, axis=1)
    pos = jnp.where(jnp.isfinite(vals), order, dcp_select.EMPTY)
    return vals, pos


def lax_topk(bits, k):
    """The merge's top-k off TPU: slots of the k largest raw-bit keys, -1 for
    empty slots (only the threshold value is used, so tie order is free)."""
    neg_inf = jax.lax.bitcast_convert_type(jnp.float32(-jnp.inf), jnp.int32)
    slots = jax.lax.top_k(dcp_select.monotone_key(bits), k)[1].astype(jnp.int32)
    filled = jnp.take_along_axis(bits, slots, axis=1) != neg_inf
    return jnp.where(filled, slots, -1)


def check_compressed(query_lens, contexts, *, main_slot=0, max_query_len=1, seed=0):
    layout, full, keep, qgate, s, m, seq, starts = make_step(seed, query_lens, contexts)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot,
                max_query_len=max_query_len)
    rep = [pooled_qsa.pooled_qsa_step(*_args(qgate[r], s, full, m, layout.full_table), **opts)
           for r in range(D)]
    pair_qgate = jnp.concatenate(qgate, axis=1)
    selects = []
    for r in range(D):
        pool = poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep),
                                layout, r, seq)
        selects.append(pooled_qsa.pooled_qsa_compressed_dcp_select_step(
            *_args(pair_qgate, s, from_raw(pool), m, layout.dcp_table), rank_operand(r),
            dcp_size=D, **opts))
    vals_pair = jnp.stack([x[6] for x in selects])
    pos_pair = jnp.stack([x[7] for x in selects])
    att = dict(kv_heads=1, compress_ratio=RATIO, token_topk=TOPK, k_scale=K_SCALE,
               v_scale=V_SCALE, main_slot=main_slot, max_query_len=max_query_len)
    dcp = []
    for r in range(D):
        pool, q, gate, lengths, req, st, _, _ = selects[r]
        dcp.append(pooled_qsa.pooled_qsa_compressed_dcp_attend_step(
            q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"], pool,
            vals_pair, pos_pair, rank_operand(r), dcp_size=D, **att))
    tokens = int(starts[-1])
    for r in range(D):
        # Every rank merges the same global selection, equal to the replicated one as a set.
        got = np.sort(np.asarray(dcp[r][3])[:tokens], axis=1)
        want = np.sort(np.asarray(rep[r][2])[:tokens], axis=1)
        np.testing.assert_array_equal(got, want)
    out_pair = jnp.stack([jnp.asarray(d[1], jnp.float32) for d in dcp])
    lse_pair = jnp.stack([jnp.asarray(d[2], jnp.float32) for d in dcp])
    live = np.zeros((out_pair.shape[1], H), bool)
    live[:tokens] = True
    reports = {}
    for r in range(D):
        gate = qgate[r].reshape(len(qgate[r]), H, 2 * DH)[..., DH:]
        got = qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank_operand(r))
        want = np.asarray(rep[r][1], np.float32).reshape(-1, H, DH)
        reports[r] = check_gated(got, want, live, what=f"compressed DCP rank {r}")
    return reports


@pytest.mark.parametrize("case", sorted(OP_CASES))
def test_compressed_dcp_matches_replicated(interpret, rpa_stand_in, monkeypatch,
                                           partial_kernels, case):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select_dcp)
    if not ON_TPU:
        monkeypatch.setattr(dcp_select, "_sc_topk", lax_topk)
    print(case, check_compressed(**OP_CASES[case]))
