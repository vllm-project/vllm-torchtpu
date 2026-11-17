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
"""TPU_MINIMAX_M3_DCP_HP_BALANCE: the head-parallel attention's ring balance
of KV-head work, and (8 host devices, interpret) the same output bits as
without it."""

import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H


def test_tile_unions_matches_numpy():
    rng = np.random.default_rng(0)
    keys = rng.integers(-1, 40, (64, 16)).astype(np.int32)
    keys[16:32] = -1
    got = np.asarray(H.tile_unions(jnp.asarray(keys), 16))
    want = [len(set(t[t >= 0].tolist())) for t in keys.reshape(4, -1)]
    assert got.tolist() == want


@pytest.mark.parametrize("seed", range(5))
def test_ring_offload_balances(seed):
    rng = np.random.default_rng(seed)
    w = np.repeat(rng.uniform(100, 1000, 4), 2)  # 4 heads, 2 replicas each
    f = np.asarray(H.ring_offload(jnp.asarray(w), 2))
    heads = w[::2]
    load = heads - f + np.roll(f, -1)
    assert f.min() == pytest.approx(0.0, abs=1e-3) and (f >= -1e-3).all()
    np.testing.assert_allclose(load, heads.mean(), rtol=1e-4)


def test_offload_split():
    unions = jnp.asarray([10, 10, 10, 10, 0, 0], jnp.int32)  # 16-row tiles
    cost = 10 + H._BALANCE_TILE_COST
    # Two tiles of work fit: hand tiles 2.. (rows 32..95).
    assert int(H._offload_split(unions, 2.0 * cost, 96, 16)) == 2
    # The row cap wins: at most 32 rows -> tiles 4.. only.
    assert int(H._offload_split(unions, 2.0 * cost, 32, 16)) == 4
    assert int(H._offload_split(unions, 0.0, 96, 16)) == 4  # nothing but empty tiles


def test_rows_from_env(monkeypatch):
    monkeypatch.delenv(H.HP_BALANCE_ENV, raising=False)
    assert H.hp_balance_rows(16, 8192) == 0
    monkeypatch.setenv(H.HP_BALANCE_ENV, "4100")
    assert H.hp_balance_rows(16, 8192) == 4096
    assert H.hp_balance_rows(16, 1024) == 512


_E2E = textwrap.dedent(
    """
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "16:16::64"
    import functools, numpy as np, jax, jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4, msa
    dcp, R, heads, Hl, D, S, K = 8, 2, 4, 8, 128, 128, 16
    C, T, TREAL = 8, 128, 112
    nseq, pool, nblk = 2, 20, 64
    rng = np.random.default_rng(0)
    mesh = jax.make_mesh((dcp,), ("d",))
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("d")))
    rep = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = np.stack([np.asarray(msa_fp4.pack_rows(jnp.asarray(
        rng.integers(0, 16, (pool, S, heads * 2 * D)), jnp.uint8))) for _ in range(dcp)])
    sc = rng.uniform(0.01, 0.05, (dcp, pool, msa_fp4.SCALE_ROWS, S)).astype(np.float32)
    q = rng.standard_normal((dcp, T, Hl, D)).astype(np.float32)
    bt = np.zeros((nseq, C), np.int32); bt[0] = rng.permutation(pool)[:C]
    sel = np.full((dcp, T, K), -1, np.int32)
    for h, w in enumerate((64, 16, 32, 16)):  # head 0 scattered, others narrow
        for t in range(TREAL):
            lo = rng.integers(0, nblk - w + 1) if h == 0 else 0
            sel[h * R:(h + 1) * R, t] = np.sort(rng.choice(np.arange(lo, lo + w), K, replace=False))[::-1]
    seq_lens = np.array([nblk * S, 0], np.int32); cu = np.array([0, TREAL, TREAL], np.int32)
    H._own_selection = lambda candidates, **kw: candidates
    splits = []
    _split = H._offload_split
    def spy(*a):
        s = _split(*a); jax.debug.callback(lambda x: splits.append(int(x)), s); return s
    H._offload_split = spy
    def run():
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                           out_specs=P("d"), check_vma=False)
        def f(rank, q, kv, sc, sel, seq_lens, bt, cu, hp):
            tokens = msa.token_positions(T, seq_lens, cu)
            return H._head_parallel_attend(
                q[0], kv[0], sc[0], sel[0], seq_lens, bt.reshape(-1), cu, hp, tokens, axis="d",
                rank=rank[0], dcp=dcp, kv_head_replicas=R, sm_scale=D ** -0.5, topk=K,
                interpret=True)[None]
        return np.asarray(f(put(np.arange(dcp, dtype=np.int32)[:, None]), put(jnp.asarray(q, jnp.bfloat16)),
                            put(kv), put(sc), put(sel), rep(seq_lens), rep(bt), rep(cu),
                            rep(np.array([True, False]))).astype(jnp.float32))
    os.environ["TPU_MINIMAX_M3_DCP_HP_BALANCE"] = "0"; a = run()
    os.environ["TPU_MINIMAX_M3_DCP_HP_BALANCE"] = "64"; b = run()
    assert splits and min(splits) < T // 16 - 1, splits  # some rank handed real rows
    assert np.abs(a).sum() > 0
    assert np.array_equal(a, b), float(np.abs(a - b).max())
    print("OK")
    """
)


def test_balanced_attend_same_bits():
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, "-c", _E2E], capture_output=True, text=True, env=env, timeout=1800
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
