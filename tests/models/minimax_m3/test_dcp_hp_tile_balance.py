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
"""TPU_MINIMAX_M3_DCP_HP_BALANCE: (host devices, interpret) the heavy KV
head's replicas hand tiles to the partner head's replicas and the
head-parallel attention keeps the row split's bits, at DCP 4 / 8 (both
directions of a pair), over two rounds, with the remote-DMA scales and a
shared scale page's row offset; a mutation of the partner pages changes the
output (the tiles did move). Plus the pricing / plan helpers and the flag
parsing (default off)."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_E2E = textwrap.dedent(
    """
    import os, sys
    dcp, nreq = int(sys.argv[1]), int(sys.argv[2])
    SPEC = "0.5:4:0" if nreq == 1 else "0.5:0:-64"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "8:16::64"
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "1"
    import functools, numpy as np, jax, jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4, msa
    R, Hl, D, S, K = 2, 8, 128, 128, 16
    heads = dcp // R
    C, T, TREAL = 8, 128, 104
    nseq, pool, nblk = 3, 20, 64
    rng = np.random.default_rng(0)
    mesh = jax.make_mesh((dcp,), ("d",), devices=jax.devices()[:dcp])
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("d")))
    rep = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = np.stack([np.asarray(msa_fp4.pack_rows(jnp.asarray(
        rng.integers(0, 16, (pool, S, heads * 2 * D)), jnp.uint8))) for _ in range(dcp)])
    sc = rng.uniform(0.01, 0.05, (dcp, pool, msa_fp4.SCALE_ROWS, S)).astype(np.float32)
    q = rng.standard_normal((dcp, T, Hl, D)).astype(np.float32)
    bt = np.zeros((nseq, C), np.int32); bt[1] = rng.permutation(pool)[:C]; bt[2] = rng.permutation(pool)[:C]
    if nreq == 2:
        nblk = 6 * dcp
    sel = np.full((dcp, T, K), -1, np.int32)
    # heads 0 / 3 dispersed (heavy), 1 / 2 narrow: both directions of a pair.
    widths = (64, 16, 16, 48) if nreq == 1 else (24, 16, 16, 20)
    for h in range(heads):
        w = widths[h]
        for t in range(8, TREAL):
            lo = rng.integers(0, nblk - w + 1) if w > 16 else 0
            sel[h * R:(h + 1) * R, t] = np.sort(rng.choice(np.arange(lo, lo + w), K, replace=False))[::-1]
    if nreq == 1:
        seq_lens = np.array([8, nblk * S, 0], np.int32); cu = np.array([0, 8, TREAL, TREAL], np.int32)
        hp = np.array([False, True, False])
    else:
        seq_lens = np.array([8, nblk * S, nblk * S], np.int32); cu = np.array([0, 8, 56, TREAL], np.int32)
        hp = np.array([False, True, True])
    H._own_selection = lambda candidates, **kw: candidates
    def run(row0=0):
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                           out_specs=P("d"), check_vma=False)
        def f(rank, q, kv, sc, sel, seq_lens, bt, cu, hp):
            tokens = msa.token_positions(T, seq_lens, cu)
            out = H._head_parallel_attend(
                q[0], kv[0], msa_fp4.with_row0(sc[0], row0), sel[0], seq_lens, bt.reshape(-1), cu, hp, tokens, axis="d",
                rank=rank[0], dcp=dcp, kv_head_replicas=R, sm_scale=D ** -0.5, topk=K, interpret=True)
            return out[None]
        return np.asarray(f(put(np.arange(dcp, dtype=np.int32)[:, None]), put(jnp.asarray(q, jnp.bfloat16)),
                            put(kv), put(sc), put(sel), rep(seq_lens), rep(bt), rep(cu),
                            rep(hp)).astype(jnp.float32))
    def on(bal="", scales="0"):
        os.environ["TPU_MINIMAX_M3_DCP_HP_BALANCE"] = bal
        os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES"] = scales
    plans = []
    orig_plan = H.balance_plan
    def spy(*a, **kw):
        s = orig_plan(*a, **kw)
        jax.debug.callback(lambda x: plans.append(np.asarray(x)), s[0])
        return s
    H.balance_plan = spy
    on(); a = run()
    assert np.abs(a).sum() > 0
    on(SPEC); b = run()
    assert plans and plans[-1].max() > 0, plans
    print("plan", plans[-1].tolist())
    assert np.array_equal(a, b), float(np.abs(a - b).max())
    on(SPEC, "1"); c = run()
    assert np.array_equal(a, c), float(np.abs(a - c).max())
    on("1"); d = run()
    assert np.array_equal(a, d), float(np.abs(a - d).max())
    if 2 * heads <= msa_fp4.SCALE_ROWS // 2:
        r0 = msa_fp4.SCALE_ROWS // 2
        on(); e0 = run(row0=r0)
        on(SPEC); e1 = run(row0=r0)
        assert np.array_equal(e0, e1), float(np.abs(e0 - e1).max())
    # Mutation: the helper's pages zeroed -> the stolen tiles change.
    orig_wait = H.exchange_partner_pages
    H.exchange_partner_pages = lambda *a, **kw: tuple(jnp.zeros_like(x) for x in orig_wait(*a, **kw))
    on(SPEC); m = run()
    assert not np.array_equal(a, m)
    H.exchange_partner_pages = orig_wait
    print("OK")
    """
)


@pytest.mark.parametrize("dcp,nreq", [(4, 1), (8, 1), (4, 2)])
def test_hp_balance_same_bits(dcp, nreq):
    r = subprocess.run(
        [sys.executable, "-c", _E2E, str(dcp), str(nreq)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=2400,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_tile_costs_count_fresh_slots():
    import jax.numpy as jnp

    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H

    rng = np.random.default_rng(1)
    keys = rng.integers(-1, 12, (64, 4)).astype(np.int32)
    keys[8:16] = -1  # an empty tile
    got = np.asarray(H.tile_costs(jnp.asarray(keys), 8, 4.0))
    want = []
    for t in keys.reshape(8, 8, 4):
        n, prev = 0, set()
        for row in t:
            n += sum(1 for x in row if x >= 0 and x not in prev)
            prev = set(row.tolist())
        want.append(n + 4.0 if n else 0.0)
    np.testing.assert_array_equal(got, np.array(want, np.float32))


def test_balance_plan():
    import jax.numpy as jnp

    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H

    # 2 heads x 2 replicas, 8 tiles (4 per replica); head 0 heavy, its last
    # tile (index 7, replica 1) and tile 6 (replica 0) unpriced.
    c = np.zeros((2, 8), np.float32)
    c[0, :6] = 10.0
    c[1, :8] = 1.0
    steal, end = H.balance_plan(jnp.asarray(c), replicas=2, max_tiles=2, call_cost=0.0)
    np.testing.assert_array_equal(np.asarray(end), [[3, 3], [4, 4]])
    # replica 0: 30 vs 4 -> budget 13 -> one tile; replica 1 likewise.
    np.testing.assert_array_equal(np.asarray(steal), [[1, 1], [0, 0]])
    steal, _ = H.balance_plan(jnp.asarray(c), replicas=2, max_tiles=2, call_cost=30.0)
    np.testing.assert_array_equal(np.asarray(steal), [[0, 0], [0, 0]])
    steal, _ = H.balance_plan(jnp.asarray(c), replicas=2, max_tiles=1, call_cost=-100.0)
    np.testing.assert_array_equal(np.asarray(steal), [[1, 1], [1, 1]])


def test_hp_balance_flag(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H

    monkeypatch.delenv("TPU_MINIMAX_M3_DCP_HP_BALANCE", raising=False)
    assert envs.TPU_MINIMAX_M3_DCP_HP_BALANCE == ""
    assert H.hp_balance() is None
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_HP_BALANCE", "0")
    assert H.hp_balance() is None
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_HP_BALANCE", "1")
    assert H.hp_balance() == (*H._BALANCE_DEFAULTS, True)
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_HP_BALANCE", "0.4::32:sync")
    assert H.hp_balance() == (0.4, H._BALANCE_DEFAULTS[1], 32.0, False)
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_HP_BALANCE", "0.4:1")
    with pytest.raises(ValueError):
        H.hp_balance()
