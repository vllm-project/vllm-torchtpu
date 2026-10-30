# SPDX-License-Identifier: Apache-2.0
"""MTP draft under VLLM_TPU_QSA_MTP_KV_DCP: the draft's rows / shared-selection
DCP steps on both ranks + dcp_combine vs the replicated draft steps.

Same recipe as ``test_dcp2_op``: NaN-poisoned rank pools projected from the
replicated pool, a deterministic indexer stub, writes exact, selections
identical, gated outputs within the combine's rounding. The draft's first step
attends each request's last row (``rows``), later steps reuse the target's
selection (``shared``).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    D,
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
    RATIO,
    ROTARY,
    TOPK,
    V_SCALE,
    _args,
    make_step,
    partial_kernels,  # noqa: F401 (fixture)
    stub_select,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa


def _last_rows(starts, tokens):
    """Each request's last row (-1 for an idle seat), padded to ``tokens``
    rows when there are more seats than the decode path holds."""
    starts = np.asarray(starts)
    rows = np.where(starts[1:] > starts[:-1], starts[1:] - 1, -1).astype(np.int32)
    return jnp.asarray(rows)


def check_draft(variant, query_lens, contexts, *, max_query_len=1, seed=0):
    layout, full, keep, qgate, s, m, seq, starts = make_step(seed, query_lens, contexts)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, max_query_len=max_query_len)
    tokens = len(qgate[0])
    rows = _last_rows(starts, tokens)
    # The target's selection the draft reuses: a replicated step's `selected`.
    target = pooled_qsa.pooled_qsa_step(*_args(qgate[0], s, full, m, layout.full_table),
                                        **opts)[2]
    extra = {
        "rows": ((rows,), pooled_qsa.pooled_qsa_rows_step, pooled_qsa.pooled_qsa_rows_dcp_step),
        "shared": ((target,), pooled_qsa.pooled_qsa_shared_step,
                   pooled_qsa.pooled_qsa_shared_dcp_step),
        "shared_rows": ((rows, target[jnp.maximum(rows, 0)]),
                        pooled_qsa.pooled_qsa_shared_rows_step,
                        pooled_qsa.pooled_qsa_shared_rows_dcp_step),
    }
    operands, rep_step, dcp_step = extra[variant]
    rep = [rep_step(*_args(qgate[r], s, full, m, layout.full_table), *operands, **opts)
           for r in range(D)]
    full_after = raw(rep[0][0])
    pair_qgate = jnp.concatenate(qgate, axis=1)
    dcp = []
    for r in range(D):
        pool = poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep),
                                layout, r, seq)
        out = dcp_step(*_args(pair_qgate, s, from_raw(pool), m, layout.dcp_table), *operands,
                       rank_operand(r), **opts)
        want = poison_tail_rows(project_rank_pool(full_after, layout, r, keep_pages=keep),
                                layout, r, seq)
        assert np.array_equal(raw(out[0]), want), f"rank {r} pool differs after the step"
        dcp.append(out)
    out_pair = jnp.stack([jnp.asarray(d[1], jnp.float32) for d in dcp])
    lse_pair = jnp.stack([jnp.asarray(d[2], jnp.float32) for d in dcp])
    attended = out_pair.shape[1]
    if variant == "shared":
        live_rows = np.arange(attended) < int(starts[-1])
        picked = np.arange(attended)
    else:
        live_rows = np.asarray(rows) >= 0
        picked = np.maximum(np.asarray(rows), 0)
    live = np.repeat(live_rows[:, None], H, axis=1)
    reports = {}
    for r in range(D):
        np.testing.assert_array_equal(np.asarray(dcp[r][3]), np.asarray(rep[r][2]))
        gate = qgate[r][picked].reshape(attended, H, 2 * DH)[..., DH:]
        got = qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank_operand(r))
        want = np.asarray(rep[r][1], np.float32).reshape(-1, H, DH)
        reports[r] = check_gated(got, want, live, what=f"draft {variant} rank {r}")
    return reports


DRAFT_CASES = {
    # Draft step 0 after a verify: last rows crossing B / 2B, an idle seat.
    "verify": dict(query_lens=[4, 4, 0, 4, 4], contexts=[37, 509, 0, 1022, 2046],
                   max_query_len=4),
    # A prefill chunk in the draft's single graph: more rows than seats.
    "mixed": dict(query_lens=[4, 300], contexts=[1020, 1301], max_query_len=4),
}


@pytest.mark.parametrize("variant", ["rows", "shared", "shared_rows"])
@pytest.mark.parametrize("case", sorted(DRAFT_CASES))
def test_dcp_draft_matches_replicated_draft(interpret, rpa_stand_in, monkeypatch,
                                            partial_kernels, case, variant):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    print(case, variant, check_draft(variant, **DRAFT_CASES[case]))
