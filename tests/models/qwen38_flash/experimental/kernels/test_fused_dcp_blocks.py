# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_FUSED_DCP_BLOCKS: the fused TensorCore merge gives exactly
compressed_dcp_blocks' selection (ascending group ids, -1 padded) on random
scores, bf16 and coarse ties straddling slot k across ranks, all-equal rows,
-inf entries, rows with fewer than k valid candidates, and ragged row counts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import pooled_qsa
from vllm_torchtpu.models.qwen38_flash.experimental.kernels.dcp_select import (
    EMPTY,
    monotone_key,
)
from vllm_torchtpu.models.qwen38_flash.experimental.kernels.fused_dcp_blocks import (
    KEY_NEG_INF,
    fused_dcp_blocks,
)

K = 64
BLOCK = 8  # groups per owner block


def sc_like_topk(bits, k):
    """The SparseCore top-k's contract: k largest raw-bit keys, keys <= -inf never
    taken.
    """
    key = monotone_key(bits)
    valid = key > KEY_NEG_INF
    _, idx = jax.lax.top_k(jnp.where(valid, key, jnp.iinfo(jnp.int32).min), k)
    return jnp.where(jnp.take_along_axis(valid, idx, axis=1), idx, -1)


def make_lists(kind, t, groups, seed):
    rng = np.random.default_rng(seed)
    vals = np.zeros((2, t, K), np.float32)
    pos = np.full((2, t, K), EMPTY, np.int32)
    for r in range(2):
        owned = np.array(
            [(i // BLOCK) * 2 * BLOCK + r * BLOCK + i % BLOCK for i in range(groups)],
            np.int32,
        )
        s = rng.standard_normal((t, groups)).astype(np.float32)
        if kind == "bf16":
            s = np.asarray(jnp.asarray(s).astype(jnp.bfloat16).astype(jnp.float32))
        elif kind == "coarse":
            s = np.round(s * 4) / 4
        elif kind == "flat":
            s[:] = 0.125
        elif kind == "neginf":
            s[:, ::3] = -np.inf
        n = min(K, groups)
        for i in range(t):
            sel = np.argsort(-s[i], kind="stable")[:n]
            sel = rng.permutation(sel)  # SC lists are not sorted by score
            vals[r, i, :n] = s[i, sel]
            pos[r, i, :n] = owned[sel]
    return jnp.asarray(vals), jnp.asarray(pos)


@pytest.mark.parametrize(
    "kind,t,groups",
    [
        ("random", 200, 4 * K),
        ("bf16", 200, 4 * K),
        ("coarse", 200, 4 * K),
        ("flat", 130, 4 * K),
        ("neginf", 128, 4 * K),
        ("random", 37, K // 2),
        ("coarse", 64, K // 2 + 3),
    ],
)
def test_fused_matches_compressed_dcp_blocks(kind, t, groups):
    vals, pos = make_lists(kind, t, groups, seed=t + groups)
    want = pooled_qsa.compressed_dcp_blocks(vals, pos, K, topk=sc_like_topk)
    interpret = jax.devices()[0].platform != "tpu"
    got = fused_dcp_blocks(vals, pos, K, interpret=interpret)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
