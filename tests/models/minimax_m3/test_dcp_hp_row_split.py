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
"""TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT and TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC /
_DIRECT: (host devices, interpret + tpu7x AOT) the replicas of a KV head
splitting the query tiles, and the fused KV + scale page exchange (both
two-stage forwarding and direct, plus legacy _IN_FLIGHT compatibility), give
the same output bits as the defaults at DCP 4 and DCP 8."""

import os
import subprocess
import sys
import textwrap

import pytest

_E2E = textwrap.dedent(
    """
    import os, sys
    dcp, nreq = int(sys.argv[1]), int(sys.argv[2])
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "8:16::64"
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
        # Two prefill requests of 6 blocks per rank each: two rounds of an
        # 8-block buffer (the second one in windows).
        nblk = 6 * dcp
    sel = np.full((dcp, T, K), -1, np.int32)
    widths = (64, 16, 32, 16) if nreq == 1 else (24, 16, 20, 16)
    for h in range(heads):
        w = widths[h]
        for t in range(8, TREAL):
            lo = rng.integers(0, nblk - w + 1) if h == 0 else 0
            sel[h * R:(h + 1) * R, t] = np.sort(rng.choice(np.arange(lo, lo + w), K, replace=False))[::-1]
    # Request 0: 8 decode rows (not head-parallel); then the prefill rows.
    if nreq == 1:
        seq_lens = np.array([8, nblk * S, 0], np.int32); cu = np.array([0, 8, TREAL, TREAL], np.int32)
        hp = np.array([False, True, False])
    else:
        seq_lens = np.array([8, nblk * S, nblk * S], np.int32); cu = np.array([0, 8, 56, TREAL], np.int32)
        hp = np.array([False, True, True])
    H._own_selection = lambda candidates, **kw: candidates
    def run(between=None):
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                           out_specs=P("d"), check_vma=False)
        def f(rank, q, kv, sc, sel, seq_lens, bt, cu, hp):
            tokens = msa.token_positions(T, seq_lens, cu)
            out = H._head_parallel_attend(
                q[0], kv[0], sc[0], sel[0], seq_lens, bt.reshape(-1), cu, hp, tokens, axis="d",
                rank=rank[0], dcp=dcp, kv_head_replicas=R, sm_scale=D ** -0.5, topk=K,
                interpret=True, between=between)
            if between is not None:
                out, x = out
                out = out + 0 * x.astype(out.dtype)
            return out[None]
        return np.asarray(f(put(np.arange(dcp, dtype=np.int32)[:, None]), put(jnp.asarray(q, jnp.bfloat16)),
                            put(kv), put(sc), put(sel), rep(seq_lens), rep(bt), rep(cu),
                            rep(hp)).astype(jnp.float32))
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "0"; a = run()
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "1"; b = run()
    assert H._row_split_logged
    assert np.abs(a[:, :, 8:TREAL]).min(axis=(0, 1, 3)).sum() >= 0 and np.abs(a).sum() > 0
    assert np.array_equal(a, b), float(np.abs(a - b).max())
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "0"
    os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT"] = "1"
    os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_IN_FLIGHT"] = "3"
    c = run()
    assert H._exchange_logged
    assert np.array_equal(a, c), float(np.abs(a - c).max())
    # `between` under interpret=True falls back to the synchronous exchange
    # because Pallas interpret mode does not model DMA semaphores returned in
    # out_specs; the tpu7x AOT test below compiles the start/wait pair.
    between = lambda q, cand: jnp.sum(q[:8].astype(jnp.float32)) + jnp.sum(cand[:8])
    os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT"] = "0"
    d = run(between)
    assert np.array_equal(a, d), float(np.abs(a - d).max())

    # Reference check: fused KV + scale_cache exchange matches pure gather/a2a
    # bit-for-bit on every live slot across empty, partial, and full rounds.
    part = C // R
    pages_np = np.arange(C, dtype=np.int32)
    for direct in ("0", "1"):
        os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT"] = direct
        @jax.jit
        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P("d"),) * 3 + (P(),) * 2,
            out_specs=(P("d"), P("d")),
            check_vma=False,
        )
        def exch(rank, kv_in, sc_in, pgs, lv):
            rk, rs = H._exchange_pages(
                kv_in[0], sc_in[0], pgs, lv[0], rank[0, 0],
                dcp=dcp, replicas=R, head_width=2 * D, axis="d", interpret=True,
            )
            return rk[None], rs[None]

        for live_n in (0, C - 3, C):
            got_kv, got_sc = exch(
                put(np.arange(dcp, dtype=np.int32)[:, None, None]),
                put(kv),
                put(sc),
                rep(pages_np),
                rep(np.array([live_n], dtype=np.int32)),
            )
            got_kv, got_sc = np.asarray(got_kv), np.asarray(got_sc)
            assert got_kv.shape == (dcp, C * dcp, S // 8, 4, 2 * D)
            for dst in range(dcp):
                h = dst // R
                for l in range(live_n):
                    j, i = l % R, l // R
                    p = pages_np[j * part + i]
                    for src in range(dcp):
                        slot = (j * dcp + src) * part + i
                        assert np.array_equal(
                            got_kv[dst, slot],
                            kv[src, p, :, :, h * 2 * D : (h + 1) * 2 * D],
                        )
                        assert np.array_equal(
                            got_sc[dst, slot, :2],
                            sc[src, p, 2 * h : 2 * h + 2],
                        )
                        assert np.all(got_sc[dst, slot, 2:] == 0)
    print("OK")
    """
)


@pytest.mark.parametrize("dcp,nreq", [(4, 1), (8, 1), (4, 2)])
def test_row_split_same_bits(dcp, nreq):
    r = subprocess.run(
        [sys.executable, "-c", _E2E, str(dcp), str(nreq)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


@pytest.mark.parametrize(
    "async_env,direct_env", [("1", "0"), ("1", "1"), ("0", "0")]
)
def test_compiles_async_and_sync_for_tpu7x(async_env, direct_env):
    """Default async start/wait exchange (no has_side_effects, fused scales)
    and the sync fallback lower cleanly for tpu7x:2x2x1 in proper overlap order."""
    code = textwrap.dedent(
        """
        import functools, os, sys
        import jax, jax.numpy as jnp, numpy as np
        from jax.experimental import topologies
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads

        async_env, direct_env = sys.argv[1], sys.argv[2]
        os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC"] = async_env
        os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT"] = direct_env
        os.environ["TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR"] = "2"
        os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "1"
        os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "8:64::1024"
        try:
            topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
        except Exception as exc:
            print(f"SKIP_NO_TOPO:{exc}")
            raise SystemExit(0)

        mesh = Mesh(np.array(topo.devices).reshape(2, 4), ("dcp_lane", "dcp"))
        sh, rep = P("dcp"), P()
        s = lambda shape, dt, spec: jax.ShapeDtypeStruct(
            shape, dt, sharding=NamedSharding(mesh, spec)
        )
        dcp, replicas, hl, d, topk, pages, seqs, pps, tokens = (
            4, 2, 8, 128, 16, 4096, 64, 512, 8192
        )
        heads = dcp // replicas
        specs = (sh,) * 5 + (rep,) * 4
        args = [
            s((dcp, 1), jnp.int32, sh),
            s((dcp * tokens, hl, d), jnp.bfloat16, sh),
            s((dcp * pages, 16, 4, heads * 2 * d), jnp.uint8, sh),
            s((dcp * pages, 8, 128), jnp.float32, sh),
            s((dcp * tokens, heads * 2 * topk), jnp.float32, sh),
            s((seqs,), jnp.int32, rep),
            s((seqs * pps,), jnp.int32, rep),
            s((seqs + 1,), jnp.int32, rep),
            s((3,), jnp.int32, rep),
        ]
        body = jax.shard_map(
            functools.partial(
                msa_dcp_heads.attend_and_merge,
                axis="dcp",
                dcp=dcp,
                kv_head_replicas=replicas,
                sm_scale=d**-0.5,
                topk=topk,
                decode_rows=5,
                rs_out_precision=False,
            ),
            mesh=mesh,
            in_specs=specs,
            out_specs=sh,
            check_vma=False,
        )
        hlo = jax.jit(body).lower(*args).compile().as_text()
        entry = hlo[hlo.index("ENTRY ") :]
        if async_env == "1":
            i_start = entry.index("minimax_m3_msa_dcp_exchange_start")
            i_wait = entry.index("minimax_m3_msa_dcp_exchange_wait")
            i_ag = entry.index("all-gather", i_start)
            i_hp = entry.rindex("minimax_m3_msa_sparse_attention_lse")
            assert i_start < i_ag < i_wait < i_hp, (i_start, i_ag, i_wait, i_hp)
        else:
            assert "minimax_m3_msa_dcp_exchange_head_pages" in entry
        assert "f32[4,1024,128]" not in hlo
        print("AOT_OK")
        """
    )
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    r = subprocess.run(
        [sys.executable, "-c", code, async_env, direct_env],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if "SKIP_NO_TOPO:" in r.stdout:
        pytest.skip(r.stdout.strip())
    assert r.returncode == 0 and "AOT_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.multichip
@pytest.mark.parametrize("direct_env", ["0", "1"])
def test_async_and_sync_match_bits_on_tpu(monkeypatch, direct_env):
    """On a multichip TPU, ASYNC=1 start/wait exchange matches ASYNC=0 bit-for-bit."""
    import functools

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads, msa_fp4

    if jax.default_backend() != "tpu" or jax.device_count() < 4:
        pytest.skip("requires >= 4 TPU cores")
    dcp, replicas, hl, d, s, topk = 4, 2, 8, 128, 128, 16
    heads = dcp // replicas
    pool, c_pages, seqs, tokens = 32, 8, 4, 2048
    rng = np.random.default_rng(7)
    mesh = Mesh(np.array(jax.devices()[:dcp]), ("dcp",))

    def put(a):
        return jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("dcp")))

    def rep(a):
        return jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = np.concatenate(
        [
            np.asarray(
                msa_fp4.pack_rows(
                    jnp.asarray(
                        rng.integers(0, 16, (pool, s, heads * 2 * d)), jnp.uint8
                    )
                )
            )
            for _ in range(dcp)
        ],
        axis=0,
    )
    sc = rng.uniform(0.01, 0.05, (dcp * pool, msa_fp4.SCALE_ROWS, s)).astype(np.float32)
    q = rng.standard_normal((dcp * tokens, hl, d)).astype(np.float32)
    cand = np.zeros((dcp, tokens, heads, 2 * topk), np.float32)
    cand[..., :topk] = -np.inf
    cand[..., topk:] = -1.0
    cand[0, :4, :, 0] = rng.standard_normal((4, heads)).astype(np.float32)
    cand[0, :4, :, topk] = 0.0
    for r in range(dcp):
        cand[r, 4:, :, :c_pages] = rng.standard_normal(
            (tokens - 4, heads, c_pages)
        ).astype(np.float32)
        cand[r, 4:, :, topk : topk + c_pages] = (
            np.arange(c_pages, dtype=np.float32) * dcp + r
        )
    cand = cand.reshape(dcp * tokens, heads * 2 * topk)
    bt = np.zeros((seqs, c_pages), np.int32)
    bt[0, :2] = [0, 1]
    bt[1] = rng.permutation(pool)[:c_pages]
    seq_lens = np.array([4, c_pages * s * dcp, 0, 0], np.int32)
    cu = np.array([0, 4, tokens, tokens, tokens], np.int32)
    dist = np.array([1, 1, 2], np.int32)
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT", direct_env)

    def run_mode(async_val: str) -> np.ndarray:
        monkeypatch.setenv("TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC", async_val)
        jax.clear_caches()
        fn = jax.jit(
            jax.shard_map(
                functools.partial(
                    msa_dcp_heads.attend_and_merge,
                    axis="dcp",
                    dcp=dcp,
                    kv_head_replicas=replicas,
                    sm_scale=d**-0.5,
                    topk=topk,
                    decode_rows=4,
                    rs_out_precision=False,
                ),
                mesh=mesh,
                in_specs=(P("dcp"),) * 5 + (P(),) * 4,
                out_specs=P("dcp"),
                check_vma=False,
            )
        )
        out = fn(
            put(np.arange(dcp, dtype=np.int32)[:, None]),
            put(jnp.asarray(q, jnp.bfloat16)),
            put(kv),
            put(sc),
            put(cand),
            rep(seq_lens),
            rep(bt.reshape(-1)),
            rep(cu),
            rep(dist),
        )
        return np.asarray(out.astype(jnp.float32))

    out0, out1 = run_mode("0"), run_mode("1")
    assert np.isfinite(out0).all() and np.abs(out0).sum() > 0
    assert np.array_equal(out0, out1)
