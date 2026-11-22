# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_WHOLE_EXPORT_SELECTION: the whole-step kernel's exported ids -> the MTP
draft's shared-selection sink rows (CPU, no kernel).

The kernel exports per verify row its pool-path ids (`QSAFullStepBlock._select`): the
selected groups below `g_a = context // 4`, ascending, slots past the count 0. `g_a`
itself never sits in a slot. `selection_groups` must give back the stock selection
(`topk_reference`: top-K by score, ties to the lower id, over the row's visible groups)
and `export_selection` the stock `selected` rows (`qsa_rpa.selected_tokens`) that
`Qwen4ExpQSA.attend_with_indices` writes to the sink.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens
from vllm_torchtpu.megakernel.wholestep.indexer import topk_reference
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import (
    export_selection,
    selection_groups,
)

T = 4


def _kernel_ids(scores, nv, ga, K):
    """One row's exported ids as the kernel builds them (pool path only)."""
    sel = topk_reference(scores, nv, K)
    pool = sorted(g for g in sel if g < ga)
    row = np.zeros(K, np.int32)
    row[: len(pool)] = pool
    return row, sorted(sel)


def _case(contexts, K, seed, live=None, tie=False):
    rng = np.random.default_rng(seed)
    nreq = len(contexts)
    live = [True] * nreq if live is None else live
    M = nreq * T
    ids = np.full((M, 1, K), 7, np.int32)  # idle rows: whatever the kernel left there
    want = np.full((M, K), -1, np.int32)
    lens = np.zeros(M, np.int32)
    for q, c in enumerate(contexts):
        scores = rng.standard_normal(c // 4 + 1).astype(np.float32)
        if tie:  # ties everywhere: the lower id wins
            scores = np.round(scores)
        if seed % 2 == 0:  # g_a (the step's first group) wins: the kernel's slot K-1 case
            scores[c // 4] = 10.0
        for t in range(T):
            r = q * T + t
            if not live[q]:
                continue
            lens[r] = c + t + 1
            nv = lens[r] // 4
            ids[r, 0], sel = _kernel_ids(scores, nv, c // 4, K)
            want[r, : len(sel)] = sel
    q_len = np.where(live, T, 0)
    starts = np.concatenate([[0], np.cumsum(q_len)]).astype(np.int32)
    return ids, want, lens, np.asarray(contexts, np.int32), starts


@pytest.mark.parametrize("contexts,K,live,tie", [
    # 3001: rows see 750 / 751 groups > K, g_a = 750 complete from row 2 on.
    ([3001, 100, 4096, 5], 64, None, False),
    # non-multiple-of-4 contexts around the budget edge (K groups = 4 K tokens).
    ([255, 256, 257, 258, 259, 260], 64, None, False),
    # an idle request between live ones (its ids are stale: ignored).
    ([200777 % 9000, 0, 8191, 1023], 512, [True, False, True, True], False),
    ([1027, 1031, 2050, 4099], 128, None, True),
])
def test_selection_groups_restore_the_stock_selection(contexts, K, live, tie):
    n_ga = 0
    for seed in range(4):
        ids, want, lens, ctx, starts = _case(contexts, K, seed, live, tie)
        groups, got_lens = selection_groups(jnp.asarray(ids), jnp.asarray(ctx),
                                            jnp.asarray(starts), T=T, K=K)
        np.testing.assert_array_equal(np.asarray(got_lens), lens)
        np.testing.assert_array_equal(np.asarray(groups), want)
        n_ga += int(sum((ids[r, 0, K - 1] == 0) and lens[r] // 4 > K
                        for r in range(len(lens))))
        rows = len(lens) - 2  # the call's rows (B) may be fewer than the kernel's (M)
        got = export_selection(jnp.asarray(ids), jnp.asarray(ctx), jnp.asarray(starts),
                               T=T, K=K, rows=rows)
        ref = selected_tokens(jnp.asarray(want[:rows]), jnp.asarray(lens[:rows]), 4 * K, 4)
        assert got.shape == (rows, 4 * K + 3) and got.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    if K == 64 and contexts[0] == 3001:
        assert n_ga > 0  # the g_a-in-slot-K-1 case is exercised


def test_selection_groups_catch_a_dropped_g_a():
    """Positive control: an export that loses g_a's slot rule fails the comparison."""
    K, contexts = 64, [3001, 4096]
    for seed in range(8):
        ids, want, _, ctx, starts = _case(contexts, K, seed)
        hit = [r for r in range(len(want)) if ids[r, 0, K - 1] == 0 and want[r, K - 1] >= 0]
        if hit:
            break
    assert hit
    bad = ids.copy()
    bad[hit[0], 0, K - 1] = bad[hit[0], 0, K - 2] + 1  # a pool id where g_a belongs
    groups, _ = selection_groups(jnp.asarray(bad), jnp.asarray(ctx), jnp.asarray(starts),
                                 T=T, K=K)
    assert not np.array_equal(np.asarray(groups), want)
