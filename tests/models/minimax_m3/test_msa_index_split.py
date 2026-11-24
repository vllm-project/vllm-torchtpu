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
"""Decode indexer split over the DCP groups (`msa_dcp.split_candidates` +
`split_merge`), interpret mode: every rank's merged candidates equal
`msa_write_and_candidates_dcp`'s bit for bit, 4-bit caches, c48 (DCP 2, one
KV head, 4 groups) and c64 (DCP 4, two KV heads, 2 groups) layouts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

PAGE, D, TOPK, ROWS, HEADS = 128, 128, 16, 8, 4


def _step(rng, dcp, ctx):
    reqs = len(ctx)
    T = reqs * ROWS
    seq_lens = np.asarray(ctx) + ROWS
    logical = -(-seq_lens // (dcp * PAGE))
    max_seqs = reqs + 1
    pps = int(logical.max()) + 1
    num_pages = int(logical.sum()) + 2
    tables = np.zeros((max_seqs, pps), np.int32)
    perm = rng.permutation(np.arange(1, num_pages))
    at = 0
    for j in range(reqs):
        tables[j, : logical[j]] = perm[at : at + logical[j]]
        at += logical[j]
    sl = np.zeros(max_seqs, np.int32)
    sl[:reqs] = seq_lens
    cu = np.full(max_seqs + 1, T, np.int32)
    cu[: reqs + 1] = np.arange(reqs + 1) * ROWS
    dist = np.array([0, 0, reqs], np.int32)
    meta = tuple(jnp.asarray(x) for x in (sl, tables.reshape(-1), cu, dist))
    return T, num_pages, meta


def _caches(rng, num_pages, g, ties):
    kv = jnp.asarray(
        rng.integers(0, 256, (num_pages, PAGE // 8, 4, 2 * g * D), np.uint8)
    )
    nib = rng.integers(0, 16, (num_pages, PAGE, D), np.uint8)
    scale = rng.uniform(0.05, 0.2, (num_pages, PAGE)).astype(np.float32)
    if ties:  # whole pages with equal keys: equal block scores, ties by block
        nib[1::3] = nib[1]
        scale[1::3] = scale[1]
    idx = msa_fp4.pack_rows(jnp.asarray(nib), jnp.asarray(scale))
    sc = jnp.asarray(
        rng.uniform(0.02, 0.1, (num_pages, msa_fp4.SCALE_ROWS, PAGE)), jnp.float32
    )
    return kv, idx, sc


@pytest.mark.parametrize(
    "dcp,ctx,ties",
    [
        (2, [3000, 9000, 700, 5000], False),
        (2, [6000, 2500], True),
        (4, [9000, 4100, 12000], False),
        (4, [8000], True),
    ],
)
def test_split_equals_served(dcp, ctx, ties):
    rng = np.random.default_rng(dcp * 100 + len(ctx) + 7 * ties)
    g = dcp // 2
    groups = HEADS // g
    T, num_pages, meta = _step(rng, dcp, ctx)
    sl, tb, cu, dist = meta
    kw = dict(topk=TOPK, init_blocks=1, local_blocks=1, dcp=dcp, interpret=True)
    iq_all = jnp.asarray(rng.standard_normal((T, HEADS, D)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((T, g, D)), jnp.bfloat16)
    ik = jnp.asarray(rng.standard_normal((T, D)), jnp.bfloat16)
    for r in range(dcp):
        kv, idx, sc = _caches(rng, num_pages, g, ties)
        served, written = {}, None
        for grp in range(groups):
            out = msa_dcp.msa_write_and_candidates_dcp(
                k, k, iq_all[:, grp * g : (grp + 1) * g], ik, kv, idx, sl, tb, cu,
                dist, rank=r, scale_cache=sc, **kw,
            )  # fmt: skip
            written = written or out[:3]
            served[grp] = np.asarray(out[-1])
        _, idx_w, sc_w = written
        parts = jnp.stack(
            [
                msa_dcp.split_candidates(
                    iq_all,
                    idx_w,
                    sl,
                    tb,
                    cu,
                    dist,
                    rank=r,
                    part=p,
                    parts=groups,
                    cache_heads=g,
                    scale_cache=sc_w,
                    **kw,
                )  # fmt: skip
                for p in range(groups)
            ]
        )
        for grp in range(groups):
            got = np.asarray(msa_dcp.split_merge(parts, grp, g, TOPK))
            assert got.shape == served[grp].shape
            np.testing.assert_array_equal(got, served[grp])
            assert np.isfinite(got[:, :TOPK]).any()


def test_merge_packed_order():
    # Ties to the lower block, -inf padding last, picks capped at topk.
    inf = -np.inf
    a = np.array([[[5.0, 3.0, inf, 4.0, 8.0, -1.0]]], np.float32)  # blocks 4, 8
    b = np.array([[[5.0, 5.0, 1.0, 2.0, 6.0, 10.0]]], np.float32)  # blocks 2, 6, 10
    got = np.asarray(
        msa_dcp.merge_packed_candidates(jnp.asarray(np.concatenate([a, b])), 3)
    )
    np.testing.assert_array_equal(got, [[5.0, 5.0, 5.0, 2.0, 4.0, 6.0]])


if __name__ == "__main__":
    jax.config.update("jax_platforms", "cpu")
    pytest.main([__file__, "-q"])
