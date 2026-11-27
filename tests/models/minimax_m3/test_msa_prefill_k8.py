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
"""TPU_MINIMAX_M3_MSA_PREFILL_K8: the glue around the head-parallel MSA
prefill attention keeps its bits.

- unset / "0": no lever; "1" = "merge+scat"; unknown levers raise;
- cmp / scat: the packed page cache tables equal the default sorts' tables;
- merge / mergedus: the decode rows' merge into the head-parallel output
  equals the cond's, with the decode rows leading or not;
- (host devices, interpret) `attend_and_merge` at DCP 4 with row split and a
  packed page cache: K8 "1" / "mergedus+scat" / "merge+cmp" give the incumbent's bits."""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, msa_dcp_heads

ENV = msa_attention_lse.K8_ENV
SORT = "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT"


def test_levers(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert msa_attention_lse.k8_levers() == frozenset()
    assert msa_attention_lse.k8_levers("0") == frozenset()
    assert msa_attention_lse.k8_levers("1") == {"merge", "scat"}
    assert msa_attention_lse.k8_levers("mergedus+scat") == {"mergedus", "scat"}
    monkeypatch.setenv(ENV, "merge")
    assert msa_attention_lse.k8_levers() == {"merge"}
    with pytest.raises(ValueError):
        msa_attention_lse.k8_levers("merge+radix")


def _unions(rng, calls, tiles, width, num_keys, max_union):
    rows = np.full((calls, tiles, width), -1, np.int32)
    for c in range(calls):
        for t in range(tiles):
            n = int(rng.integers(0, max_union + 1))
            rows[c, t, :n] = np.sort(rng.choice(num_keys, n, replace=False))
    return jnp.asarray(rows.reshape(calls, -1))


@pytest.mark.parametrize(
    "tiles,width,num_keys,max_union,cache,tile_pages",
    [
        (8, 256, 300, 40, 512, 32),  # unions past the sorted width: streams
        (16, 256, 2000, 60, 64, 64),  # more distinct keys than slots
        (16, 256, 2000, 100, 1024, 192),
        (16, 256, 3000, 200, 1024, 192),  # some calls stream
        (16, 64, 50, 50, 512, 128),
        (4, 256, 9, 0, 16, 32),  # no pages at all
    ],
)
@pytest.mark.parametrize("mode", ["packed", "packed_batched"])
def test_tables_match(monkeypatch, tiles, width, num_keys, max_union, cache, tile_pages, mode):
    monkeypatch.setattr(msa_attention_lse, "PAGE_CACHE_TILE_PAGES", tile_pages)
    rng = np.random.default_rng(tiles * 7 + max_union)

    def tables(unions, sort, k8):
        monkeypatch.setenv(SORT, sort)
        monkeypatch.setenv(ENV, k8)
        return np.asarray(
            jax.jit(
                lambda u: msa_attention_lse._page_cache_tables(u, tiles, cache, num_keys)
            )(unions)
        )

    for spread in (max_union, max(max_union // 3, 1)):
        unions = _unions(rng, 4, tiles, width, num_keys, spread)
        ref = tables(unions, "", "0")
        assert np.array_equal(tables(unions, mode, "0"), ref)
        for k8 in ("cmp", "scat", "1"):
            np.testing.assert_array_equal(tables(unions, mode, k8), ref, err_msg=k8)


@pytest.mark.parametrize("lead", [True, False])
def test_merge_tail(monkeypatch, lead):
    """`attend_and_merge`'s merge of the decode rows (stand-in attentions):
    every K8 merge lever gives the cond's bits."""
    T, Hl, D, nseq, dr = 2048, 8, 128, 16, 8
    rng = np.random.default_rng(int(lead))
    hp = jnp.asarray(rng.standard_normal((Hl, T, D)), jnp.bfloat16)
    monkeypatch.setattr(msa_dcp_heads, "_head_parallel_attend", lambda *a, **kw: hp)
    monkeypatch.setattr(
        msa_dcp_heads, "_dcp_attend", lambda q, *a, **kw: (q * 3).astype(q.dtype)
    )
    # 5 decode requests (1, 8, 3, 8, 2 rows) and two prefill requests; the
    # decode requests lead or sit between and after the prefill ones.
    dec = [1, 8, 3, 8, 2]
    qlen = dec + [1000, T - 1000 - sum(dec)] if lead else [600] + dec[:3] + [T - 600 - sum(dec)] + dec[3:]
    qlen = np.array(qlen + [0] * (nseq - len(qlen)), np.int32)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(qlen)]), jnp.int32)
    dist = jnp.asarray([0, 0, 7], jnp.int32)
    q = jnp.asarray(rng.standard_normal((T, Hl, D)), jnp.bfloat16)
    seq_lens = jnp.asarray(qlen * 3, jnp.int32)

    def run(k8):
        monkeypatch.setenv(ENV, k8)
        f = jax.jit(
            lambda q, cu: msa_dcp_heads.attend_and_merge(
                jnp.zeros((1,), jnp.int32), q, None, None, jnp.zeros((T, 64), jnp.float32),
                seq_lens, jnp.zeros((nseq * 4,), jnp.int32), cu, dist, axis="d", dcp=4,
                kv_head_replicas=2, sm_scale=0.1, topk=16, decode_rows=dr,
                rs_out_precision=True,
            )
        )
        return np.asarray(f(q, cu).astype(jnp.float32))

    ref = run("0")
    # the decode rows got the stand-in, the others the head-parallel rows
    rows = np.concatenate([np.arange(c, c + n) for c, n in zip(np.asarray(cu[:-1]), qlen) if 0 < n <= dr])
    exp = np.asarray(jnp.asarray(q[rows] * 3, jnp.bfloat16).astype(jnp.float32))
    np.testing.assert_array_equal(ref[rows], exp)
    for k8 in ("merge", "mergedus", "1"):
        np.testing.assert_array_equal(run(k8), ref, err_msg=k8)


_E2E = textwrap.dedent(
    """
    import os, sys
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "8:16::64"
    os.environ["TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT"] = "packed"
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "1"
    import functools, numpy as np, jax, jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4
    dcp, R, Hl, D, S, K = 4, 2, 8, 128, 128, 16
    heads = dcp // R
    C, T, TREAL = 8, 2048, 2000
    nseq, pool, nblk = 4, 20, 64
    rng = np.random.default_rng(0)
    mesh = jax.make_mesh((dcp,), ("d",), devices=jax.devices()[:dcp])
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("d")))
    rep = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = np.stack([np.asarray(msa_fp4.pack_rows(jnp.asarray(
        rng.integers(0, 16, (pool, S, heads * 2 * D)), jnp.uint8))) for _ in range(dcp)])
    sc = rng.uniform(0.01, 0.05, (dcp, pool, msa_fp4.SCALE_ROWS, S)).astype(np.float32)
    q = rng.standard_normal((dcp, T, Hl, D)).astype(np.float32)
    bt = np.zeros((nseq, C), np.int32); bt[1] = rng.permutation(pool)[:C]
    sel = np.full((dcp, T, K), -1, np.int32)
    for h in range(heads):
        w = (48, 20)[h]
        for t in range(16, TREAL):
            lo = rng.integers(0, nblk - w + 1)
            sel[h * R:(h + 1) * R, t] = np.sort(rng.choice(np.arange(lo, lo + w), K, replace=False))[::-1]
    # Requests 0 and 2: 8 decode rows each (stand-in attention); request 1
    # prefills the rest. With `lead` the decode rows come first.
    lead = sys.argv[1] == "1"
    if lead:
        qlen = np.array([8, TREAL - 16, 8, 0], np.int32)
    else:
        qlen = np.array([8, TREAL - 16, 8, 0], np.int32)[[1, 0, 2, 3]]
        bt = bt[[1, 0, 2, 3]]
        sel = np.concatenate([sel[:, 16:TREAL], sel[:, :16], sel[:, TREAL:]], 1)
    cu = np.concatenate([[0], np.cumsum(qlen)]).astype(np.int32)
    seq_lens = np.where(qlen > 8, nblk * S, qlen).astype(np.int32)
    dist = np.array([0, 0, 3], np.int32)
    H._own_selection = lambda candidates, **kw: candidates
    H._dcp_attend = lambda q, *a, **kw: (q * 3).astype(q.dtype)
    def run(k8):
        os.environ["TPU_MINIMAX_M3_MSA_PREFILL_K8"] = k8
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                           out_specs=P("d"), check_vma=False)
        def f(rank, q, kv, sc, sel, seq_lens, bt, cu, dist):
            out = H.attend_and_merge(
                rank[0], q[0], kv[0], sc[0], sel[0], seq_lens, bt.reshape(-1), cu, dist, axis="d",
                dcp=dcp, kv_head_replicas=R, sm_scale=D ** -0.5, topk=K, decode_rows=8,
                rs_out_precision=True, interpret=True)
            return out[None]
        return np.asarray(f(put(np.arange(dcp, dtype=np.int32)[:, None]), put(jnp.asarray(q, jnp.bfloat16)),
                            put(kv), put(sc), put(sel), rep(seq_lens), rep(bt), rep(cu),
                            rep(dist)).astype(jnp.float32))
    a = run("0")
    assert np.abs(a).sum() > 0
    for k8 in ("1", "mergedus+scat", "merge+cmp"):
        b = run(k8)
        assert np.array_equal(a, b), (k8, float(np.abs(a - b).max()))
    print("OK")
    """
)


@pytest.mark.parametrize("lead", ["1", "0"])
def test_attend_and_merge_same_bits(lead):
    r = subprocess.run(
        [sys.executable, "-c", _E2E, lead],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=3000,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
