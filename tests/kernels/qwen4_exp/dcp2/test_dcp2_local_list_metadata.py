# SPDX-License-Identifier: Apache-2.0
"""``_dcp_page_metadata`` of a rank's own list (VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE,
``dcp[3]``) equals that of the global merged list, for every rank of a DCP
group of 4: the same pages, kv_lens, cu_q_lens and distribution.

The local lists come in SparseCore slot order (shuffled within the valid
prefix); rows cover dead rows, a tail-only row, fewer than k valid groups,
distinct / tied / mostly-zero / signed-zero scores, and the zero-filled rows
the RPA batch pads a step to (blocks 0, length 0), which must own nothing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa, qsa_rpa

D, K, RATIO, PAGE = 4, 64, 4, 128
GPP = PAGE // RATIO  # groups per page, the ownership unit


def lax_topk(bits, k):
    neg_inf = jax.lax.bitcast_convert_type(jnp.float32(-jnp.inf), jnp.int32)
    slots = jax.lax.top_k(dcp_select.monotone_key(bits), k)[1].astype(jnp.int32)
    filled = jnp.take_along_axis(bits, slots, axis=1) != neg_inf
    return jnp.where(filled, slots, -1)


def local_lists(scores, lengths, seed):
    """Each rank's top-K over its owned complete groups (lower id first on
    ties), valid entries first in a shuffled order, ``EMPTY`` after."""
    rng = np.random.default_rng(seed)
    t, n = scores.shape
    g = np.arange(n)
    vals, pos = [], []
    for r in range(D):
        owned = (g // GPP) % D == r
        valid = (g[None] < (lengths // RATIO)[:, None]) & owned[None]
        s = np.where(valid, scores, -np.inf)
        order = np.argsort(-s, axis=1, kind="stable")[:, :K]
        v = np.take_along_axis(s, order, axis=1)
        p = np.where(np.isfinite(v), order, dcp_select.EMPTY)
        for i in range(t):
            nv = int(np.isfinite(v[i]).sum())
            perm = rng.permutation(nv)
            v[i, :nv], p[i, :nv] = v[i, :nv][perm], p[i, :nv][perm]
        vals.append(v.astype(np.float32))
        pos.append(p.astype(np.int32))
    return jnp.asarray(np.stack(vals)), jnp.asarray(np.stack(pos))


def scores_of(kind, rng, shape):
    if kind == "distinct":
        return rng.normal(size=shape)
    if kind == "tied":
        return np.floor(2 * rng.normal(size=shape))
    if kind == "zero":
        return np.where(rng.random(shape) < 0.97, 0.0, rng.normal(size=shape))
    return np.where(rng.random(shape) < 0.5, -0.0, 0.0)


@pytest.mark.parametrize("kind", ["distinct", "tied", "zero", "signed_zero"])
def test_local_list_metadata_matches_global(interpret, kind):
    rng = np.random.default_rng(0)
    t, n, padded = 48, 2048, 16
    lengths = rng.integers(1, n * RATIO, t).astype(np.int32)
    lengths[:6] = [0, 3, 37, 4 * K - 1, 4 * K, 4 * K + 3]
    vals_pair, pos_pair = local_lists(scores_of(kind, rng, (t, n)).astype(np.float32), lengths, 1)
    whole = pooled_qsa.compressed_dcp_blocks(vals_pair, pos_pair, K, topk=lax_topk)
    cols = n * RATIO // (D * PAGE) + 1
    table = jnp.asarray(rng.permutation(8192)[: (t + padded) * cols].reshape(t + padded, cols)
                        .astype(np.int32))
    # The RPA batch pads a step's rows with zeros (rpa_attention's split).
    all_lengths = jnp.asarray(np.concatenate([lengths, np.zeros(padded, np.int32)]))
    for r in range(D):
        rank = jnp.asarray([r], jnp.int32)
        own = pooled_qsa.compressed_dcp_own_blocks(vals_pair, pos_pair, rank, K, topk=lax_topk)
        a = qsa_rpa._dcp_page_metadata(
            jnp.pad(whole, ((0, padded), (0, 0))), all_lengths, table, ratio=RATIO,
            page_size=PAGE, dcp=(rank, D, 0))
        b = qsa_rpa._dcp_page_metadata(
            jnp.pad(own, ((0, padded), (0, 0))), all_lengths, table, ratio=RATIO,
            page_size=PAGE, dcp=(rank, D, 0, True))
        for name, x, y in zip(("pages", "kv_lens", "cu_q_lens", "distribution"), a, b):
            np.testing.assert_array_equal(np.asarray(y), np.asarray(x), f"rank {r} {name}")
        assert not np.asarray(b[1])[t:].any(), f"rank {r}: a padded row owns keys"
