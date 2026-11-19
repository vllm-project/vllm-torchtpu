# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_DCP: the pair's owned local top-k lists merged by
merge_threshold / keep_mask give exactly the replicated top-k (raw-bit score
order, lower group id on ties), including ties straddling slot k across ranks,
+-0.0, rows with fewer than k groups, and empty rows."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.dcp_select import (
    EMPTY,
    keep_mask,
    merge_threshold,
)

B_GROUPS = 8  # groups per owner block (B / ratio) in these small cases


def key_of(vals):
    bits = vals.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def replicated_topk(vals, n, k):
    """The replicated SC order: raw-bit key descending, group id ascending."""
    order = sorted(range(n), key=lambda g: (-key_of(vals[g : g + 1])[0], g))
    return set(order[:k])


def lax_like_topk(bits, k):
    """Slots of the k largest raw-bit keys per row, arbitrary tie order."""
    b = np.asarray(bits).astype(np.int64)
    key = np.where(b < 0, b ^ 0x7FFFFFFF, b)
    out = np.full((key.shape[0], k), -1, np.int32)
    neg_inf = np.array(-np.inf, np.float32).view(np.int32)
    for r in range(key.shape[0]):
        valid = np.where(b[r] != neg_inf)[0]
        best = valid[np.argsort(-key[r, valid], kind="stable")][:k]
        # Reverse the order among ties on purpose: the merge must not depend on it.
        out[r, : len(best)] = best[::-1]
    return jnp.asarray(out)


def make_row(rng, n, kind):
    if kind == "random":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "ties":
        return rng.choice(np.float32([0.5, 0.25, 0.25, 1.0, -0.0, 0.0]), n).astype(np.float32)
    if kind == "flat":
        return np.full(n, 0.125, np.float32)
    raise ValueError(kind)


def local_topk(vals, n, rank, k):
    owned = [g for g in range(n) if (g // B_GROUPS) % 2 == rank]
    order = sorted(owned, key=lambda g: (-key_of(vals[g : g + 1])[0], g))[:k]
    v = np.full(k, -np.inf, np.float32)
    p = np.full(k, EMPTY, np.int32)
    perm = np.random.default_rng(len(order)).permutation(len(order))  # unsorted, like SC
    for i, j in enumerate(perm):
        v[i] = vals[order[j]]
        p[i] = order[j]
    return v, p


@pytest.mark.parametrize("kind", ["random", "ties", "flat"])
@pytest.mark.parametrize("k", [8, 16])
def test_union_of_kept_entries_is_the_replicated_topk(kind, k):
    rng = np.random.default_rng(k)
    rows = []
    for n in (0, 5, k - 1, k, k + 3, 3 * k, 60):  # empty, short and long rows
        rows.append((n, make_row(rng, n, kind)))
    vp = np.zeros((2, len(rows), k), np.float32)
    pp = np.zeros((2, len(rows), k), np.int32)
    for t, (n, vals) in enumerate(rows):
        for rank in range(2):
            vp[rank, t], pp[rank, t] = local_topk(vals, n, rank, k)
    thr_val, thr_pos = merge_threshold(
        jnp.asarray(vp), jnp.asarray(pp), k, topk=lax_like_topk
    )
    for t, (n, vals) in enumerate(rows):
        kept = set()
        for rank in range(2):
            mask = np.asarray(
                keep_mask(jnp.asarray(vp[rank, t : t + 1]), jnp.asarray(pp[rank, t : t + 1]),
                          thr_val[t : t + 1], thr_pos[t : t + 1])
            )[0]
            kept |= {int(p) for p in pp[rank, t][mask]}
        assert kept == replicated_topk(vals, n, k), (kind, k, n)


def test_negative_zero_ranks_below_positive_zero():
    k = 1
    vp = np.array([[[-0.0]], [[0.0]]], np.float32)
    pp = np.array([[[0]], [[B_GROUPS]]], np.int32)  # rank 0 has the lower id
    thr_val, thr_pos = merge_threshold(jnp.asarray(vp), jnp.asarray(pp), k, topk=lax_like_topk)
    keep0 = bool(np.asarray(keep_mask(jnp.asarray(vp[0]), jnp.asarray(pp[0]), thr_val, thr_pos))[0, 0])
    keep1 = bool(np.asarray(keep_mask(jnp.asarray(vp[1]), jnp.asarray(pp[1]), thr_val, thr_pos))[0, 0])
    assert (keep0, keep1) == (False, True)
