# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU interpret-mode unit tests for wholestep QSA indexer & union compaction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.megakernel.wholestep import indexer as ix
from vllm_torchtpu.megakernel.wholestep import union_attn as ua


def _thresholds_unrolled_ref(
    bits: jax.Array, ids: jax.Array, nv: jax.Array, K: int, G: int
) -> tuple[jax.Array, jax.Array]:
    """Reference unrolled Python-loop implementation of `indexer.thresholds`."""
    cnt = lambda m: jnp.sum(m.astype(jnp.int32), axis=(1, 2), keepdims=True)  # noqa: E731
    over = nv > K
    lo = jnp.zeros(nv.shape, jnp.int32)
    for b in range(30, -1, -1):
        cand = lo | (1 << b)
        lo = jnp.where(cnt(bits >= cand) >= K, cand, lo)
    thr = jnp.where(over, lo, 0)
    eq = bits == thr
    need = K - cnt(bits > thr)
    t = jnp.zeros(nv.shape, jnp.int32)
    for b in range(int(G - 1).bit_length() - 1, -1, -1):
        cand = t | (1 << b)
        t = jnp.where(cnt(eq & (ids < cand)) < need, cand, t)
    return thr, jnp.where(over & (cnt(eq) > need), t, G)


@pytest.mark.parametrize("tie_skip", [False, True])
@pytest.mark.parametrize("early_exit", [(), (23, 4)])
def test_thresholds_matches_unrolled_reference(tie_skip: bool, early_exit: tuple):
    rng = np.random.default_rng(7)
    R_rows, S, K = 4, 8, 64
    G = S * 128
    # Mix distinct scores, exact ties (row 1), and under-budget rows (row 2).
    scores = rng.uniform(0.0, 10.0, size=(R_rows, S, 128)).astype(np.float32)
    scores[1] = np.floor(scores[1] * 4.0) / 4.0  # force ties at threshold
    bits = jax.lax.bitcast_convert_type(jnp.array(scores), jnp.int32)
    ids = jax.lax.broadcasted_iota(
        jnp.int32, (R_rows, S, 128), 1
    ) * 128 + jax.lax.broadcasted_iota(jnp.int32, (R_rows, S, 128), 2)
    nv = jnp.array([[[200]], [[500]], [[30]], [[G]]], dtype=jnp.int32)
    bits = jnp.where(ids < nv, bits, -1)

    ref_thr, ref_tm = _thresholds_unrolled_ref(bits, ids, nv, K, G)
    got_thr, got_tm = ix.thresholds(
        bits, ids, nv, K, G, tie_skip=tie_skip, early_exit=early_exit
    )

    # Under early_exit, `thr` can stop slightly below the K-th value when the
    # selected set is already exact; verify exact selected mask in all modes,
    # and exact (thr, tmax) when early_exit is off.
    over = nv > K
    sel_ref = (over & ((bits > ref_thr) | ((bits == ref_thr) & (ids <= ref_tm)))) | (
        ~over & (bits >= 0)
    )
    sel_got = (over & ((bits > got_thr) | ((bits == got_thr) & (ids <= got_tm)))) | (
        ~over & (bits >= 0)
    )
    np.testing.assert_array_equal(np.asarray(sel_got), np.asarray(sel_ref))
    if not early_exit:
        np.testing.assert_array_equal(np.asarray(got_thr), np.asarray(ref_thr))
        np.testing.assert_array_equal(np.asarray(got_tm), np.asarray(ref_tm))


@pytest.mark.parametrize("use_map_id", [False, True])
@pytest.mark.parametrize("base_rank", [0, 1])
@pytest.mark.parametrize("n_active", [0, 7, 320])
def test_compact_union_into_interpret_matches_reference(
    use_map_id: bool, base_rank: int, n_active: int
):
    rng = np.random.default_rng(13 + n_active + base_rank)
    S, KU, D = 8, 512, 2
    code_np = np.zeros((S, 128), dtype=np.int32)
    if n_active > 0:
        chosen = rng.choice(S * 128, size=n_active, replace=False)
        for idx in chosen:
            code_np[idx // 128, idx % 128] = int(rng.integers(1, 16))
    code = jnp.array(code_np)
    consts = ua.union_consts(S, KU)
    map_fn = (lambda x: x ^ 1) if use_map_id else None

    ref_ids, ref_codes, ref_n = ua.compact_union(code, KU, consts)
    if map_fn is not None:
        ref_ids = map_fn(ref_ids)

    def _kernel(code_ref, id_ref, cd_ref, n_ref):
        nu = ua.compact_union_into(
            code_ref[...],
            KU,
            ua.union_consts(S, KU),
            id_ref,
            cd_ref,
            req_idx=0,
            base_col=base_rank * KU,
            map_id=map_fn,
        )
        n_ref[0] = jnp.sum(nu)

    got_ids, got_codes, got_n = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, 1, D * KU), jnp.int32),
            jax.ShapeDtypeStruct((1, 1, D * KU), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        interpret=True,
    )(code)

    assert int(got_n[0]) == int(jnp.sum(ref_n))
    check_len = ((n_active + ua.UNION_CHUNK - 1) // ua.UNION_CHUNK) * ua.UNION_CHUNK
    if check_len > 0:
        sl = slice(base_rank * KU, base_rank * KU + check_len)
        np.testing.assert_array_equal(
            np.asarray(got_ids[0, :, sl]), np.asarray(ref_ids[:, :check_len])
        )
        np.testing.assert_array_equal(
            np.asarray(got_codes[0, :, sl]), np.asarray(ref_codes[:, :check_len])
        )
