# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for VLLM_TPU_MK_INDEXER_FAST_THR (radix-4 packed thresholds
and fast bounded/T=1 compact_union)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel.wholestep import indexer as ix
from vllm_torchtpu.megakernel.wholestep import union_attn as ua
from vllm_torchtpu.megakernel.wholestep.qsa_full_idx import (
    QSAFullStepIdxBlock,
    thresholds_radix2,
)
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc


@pytest.mark.parametrize("kind", ["plain", "ties", "zero"])
@pytest.mark.parametrize("S,K", [(16, 64), (32, 64), (16, 512)])
def test_indexer_fast_thr_matches_plain_and_radix2(kind: str, S: int, K: int):
    """Packed radix-4 `ix.thresholds(..., fast=True)` matches both plain
    `ix.thresholds` and oxpd's `thresholds_radix2` bit-for-bit."""
    G = S * 128
    key = jax.random.PRNGKey(13 + len(kind) + S + K)
    if kind == "zero":
        s = jnp.zeros((4, S, 128), F32)
    elif kind == "ties":
        s = jnp.floor(jax.random.uniform(key, (4, S, 128), F32) * 3.0)
    else:
        s = jax.random.uniform(key, (4, S, 128), F32)
    ids = jnp.broadcast_to(
        jnp.arange(S, dtype=I32)[:, None] * 128 + jnp.arange(128, dtype=I32)[None],
        (4, S, 128),
    )
    nv = jnp.asarray([0, min(K, G), min(K + 5, G), G], I32).reshape(4, 1, 1)
    bits = jnp.where(ids < nv, jax.lax.bitcast_convert_type(s, I32), -1)
    t_ref, m_ref = ix.thresholds(bits, ids, nv, K, G)
    t_r2, m_r2 = thresholds_radix2(bits, ids, nv, K, G)
    t_fast, m_fast = ix.thresholds(bits, ids, nv, K, G, fast=True)
    np.testing.assert_array_equal(np.asarray(t_fast), np.asarray(t_ref))
    np.testing.assert_array_equal(np.asarray(m_fast), np.asarray(m_ref))
    np.testing.assert_array_equal(np.asarray(t_r2), np.asarray(t_ref))
    np.testing.assert_array_equal(np.asarray(m_r2), np.asarray(m_ref))


@pytest.mark.parametrize("n_sel", [0, 1, 128, 129, 256, 257, 512])
@pytest.mark.parametrize("t1", [False, True])
def test_compact_union_fast_matches_plain(n_sel: int, t1: bool):
    """`compact_union(..., t1=..., fast=True)` matches baseline bit-for-bit."""
    S_u, KU = 16, 512
    rng = np.random.default_rng(200 + n_sel + int(t1))
    code = np.zeros((S_u * 128,), dtype=np.int32)
    if n_sel > 0:
        picks = rng.choice(S_u * 128, size=n_sel, replace=False)
        code[picks] = 1 if t1 else rng.integers(1, 16, size=n_sel, dtype=np.int32)
    code_j = jnp.asarray(code.reshape(S_u, 128))
    consts = ua.union_consts(S_u, KU)
    ref_i, ref_c, ref_n = ua.compact_union(code_j, KU, consts)
    got_i, got_c, got_n = ua.compact_union(code_j, KU, consts, t1=t1, fast=True)
    np.testing.assert_array_equal(np.asarray(got_n), np.asarray(ref_n))
    np.testing.assert_array_equal(np.asarray(got_i), np.asarray(ref_i))
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(ref_c))


def test_indexer_fast_thr_env_wiring(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_INDEXER_FAST_THR", raising=False)
    assert "fast_thr" not in wc.indexed_options()
    monkeypatch.setenv("VLLM_TPU_MK_INDEXER_FAST_THR", "1")
    assert wc.indexed_options().get("fast_thr") is True
    blk = QSAFullStepIdxBlock(
        max_groups=1024,
        pages_per_seq=32,
        union_attn=True,
        per_request_scores=True,
        radix_bits=2,
        fast_thr=True,
    )
    assert blk.fast_thr is True and blk.radix_bits == 2
