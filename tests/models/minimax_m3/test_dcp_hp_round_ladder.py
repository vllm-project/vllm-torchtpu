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
"""TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER: (host devices, interpret) a
head-parallel step whose prefill requests fit a smaller round buffer gives
the same output bits as the max_model_len-sized buffer, for one and two
prefill requests, at DCP 4 and DCP 8, in the rung and in the full branch;
the rungs parse to multiples of the replicas below the capacity; and the
traced program keeps its inputs and output whatever the step's rows (the
ladder is a `lax.switch` inside the same program)."""

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
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4, msa
    R, Hl, D, S, K = 2, 8, 128, 128, 16
    heads = dcp // R
    C, T, TREAL = 8, 128, 104
    nseq, pool = 3, 20
    rng = np.random.default_rng(1)
    mesh = jax.make_mesh((dcp,), ("d",), devices=jax.devices()[:dcp])
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("d")))
    rep = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = np.stack([np.asarray(msa_fp4.pack_rows(jnp.asarray(
        rng.integers(0, 16, (pool, S, heads * 2 * D)), jnp.uint8))) for _ in range(dcp)])
    sc = rng.uniform(0.01, 0.05, (dcp, pool, msa_fp4.SCALE_ROWS, S)).astype(np.float32)
    q = rng.standard_normal((dcp, T, Hl, D)).astype(np.float32)
    bt = np.zeros((nseq, C), np.int32); bt[1] = rng.permutation(pool)[:C]; bt[2] = rng.permutation(pool)[:C]
    # Logical blocks per rank of each prefill request (at most the block
    # table's C), its context, and the selectable blocks (over the ranks).
    per = 7 if nreq == 1 else 5
    nblk = per * dcp
    sel = np.full((dcp, T, K), -1, np.int32)
    for h in range(heads):
        w = min(nblk, (24, 16, 20, 16)[h])
        for t in range(8, TREAL):
            lo = rng.integers(0, nblk - w + 1)
            sel[h * R:(h + 1) * R, t] = np.sort(rng.choice(np.arange(lo, lo + w), K, replace=False))[::-1]
    # Request 0: 8 decode rows (not head-parallel); then the prefill rows.
    if nreq == 1:
        seq_lens = np.array([8, nblk * S - 37, 0], np.int32); cu = np.array([0, 8, TREAL, TREAL], np.int32)
        hp = np.array([False, True, False])
    else:
        seq_lens = np.array([8, nblk * S, nblk * S - 100], np.int32); cu = np.array([0, 8, 56, TREAL], np.int32)
        hp = np.array([False, True, True])
    need = per * nreq
    H._own_selection = lambda candidates, **kw: candidates
    def run(factor, ladder):
        envs.TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR = factor
        envs.TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER = ladder
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                           out_specs=P("d"), check_vma=False)
        def f(rank, q, kv, sc, sel, seq_lens, bt, cu, hp):
            tokens = msa.token_positions(T, seq_lens, cu)
            out = H._head_parallel_attend(
                q[0], kv[0], sc[0], sel[0], seq_lens, bt.reshape(-1), cu, hp, tokens, axis="d",
                rank=rank[0], dcp=dcp, kv_head_replicas=R, sm_scale=D ** -0.5, topk=K,
                interpret=True)
            return out[None]
        args = (put(np.arange(dcp, dtype=np.int32)[:, None]), put(jnp.asarray(q, jnp.bfloat16)),
                put(kv), put(sc), put(sel), rep(seq_lens), rep(bt), rep(cu), rep(hp))
        txt = f.lower(*args).as_text()
        return np.asarray(f(*args).astype(jnp.float32)), txt
    tok = lambda blocks: blocks * S * dcp  # rung of `blocks` logical blocks per rank
    # Reference: the full buffer of 4 max-length requests (one round), key off.
    ref, txt = run(4, "")
    assert np.abs(ref[:, :, 8:TREAL]).sum() > 0 and np.all(ref[:, :, :8] == 0)
    # Rung that holds the step exactly / with room / two rungs (smallest
    # fitting one taken) / a rung too small (full branch inside the switch).
    for ladder in (str(tok(need)), str(tok(need + 2)), f"{tok(need)}/{tok(need + 6)}",
                   f"{tok(need - 2)}", f"{tok(2)}/{tok(need + 2)}"):
        got, txt = run(4, ladder)
        assert H._round_ladder_logged
        assert np.array_equal(ref, got), (ladder, float(np.abs(ref - got).max()))
    # A rung at or above the full buffer is dropped: the key-off program.
    _, off = run(1, "")
    _, big = run(1, str(tok(8 * 4)))
    assert big == off
    if nreq == 2:
        # Factor 1 key off: two rounds of an 8-block buffer; the ladder's
        # rung of 2 * per blocks takes both requests in one round.
        two, _ = run(1, "")
        one, _ = run(1, str(tok(2 * per)))
        # Rounds attend disjoint rows over the same pages: same bits.
        assert np.array_equal(two, one), float(np.abs(two - one).max())
    print("OK")
    """
)


@pytest.mark.parametrize("dcp,nreq", [(4, 1), (8, 1), (4, 2)])
def test_round_ladder_same_bits(dcp, nreq):
    r = subprocess.run(
        [sys.executable, "-c", _E2E, str(dcp), str(nreq)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_ladder_parse(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H

    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER", "", raising=False)
    assert H.hp_round_ladder(128, 4, 2, 4096) == []
    # 1M at DCP 4, factor 2: capacity 4096 logical blocks per rank.
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER", "262144/524288")
    assert H.hp_round_ladder(128, 4, 2, 4096) == [512, 1024]
    # Rounded up to the replicas; duplicates, commas and too-big rungs dropped.
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER", "1000,1024/2097152/9999999")
    assert H.hp_round_ladder(128, 4, 2, 4096) == [2]


_TRACE = textwrap.dedent(
    """
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "8:64::1024"
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR"] = "2"
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "1"
    import jax, jax.extend, jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4
    # Served c64 full-trace shapes per rank: DCP 4, 2 replicas, 64 seqs, 1M.
    DCP, R, HL, D, K, SEQS, PPS, POOL = 4, 2, 8, 128, 16, 64, 2048, 1024
    mesh = jax.make_mesh((DCP,), ("d",), devices=jax.devices()[:DCP])
    I32, U8, F32, BF16 = jnp.int32, jnp.uint8, jnp.float32, jnp.bfloat16
    def avals(T):
        sh = lambda spec: NamedSharding(mesh, spec)
        S = lambda shape, dt: jax.ShapeDtypeStruct(shape, dt, sharding=sh(P("d")))
        Rp = lambda shape, dt: jax.ShapeDtypeStruct(shape, dt, sharding=sh(P()))
        return (S((DCP, 1), I32), S((DCP * T, HL, D), BF16), S((DCP * POOL, 16, 4, 4 * D), U8),
                S((DCP * POOL, 8, 128), F32), S((DCP * T, 4 * K), F32), Rp((SEQS,), I32),
                Rp((SEQS * PPS,), I32), Rp((SEQS + 1,), I32), Rp((3,), I32))
    def body(rank, q, kv, sc, cand, sl, bt, cu, dist):
        return H.attend_and_merge(rank, q, kv, msa_fp4.with_row0(sc, 4), cand, sl, bt, cu, dist,
                                  axis="d", dcp=DCP, kv_head_replicas=R, sm_scale=D ** -0.5,
                                  topk=K, decode_rows=8, rs_out_precision=False)
    # A fresh function per trace: JAX caches the traced jaxpr per function.
    def make():
        return jax.shard_map(lambda *x: body(*x), mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                             out_specs=P("d"), check_vma=False)
    def count(jaxpr, branches):
        n = 0
        for e in jaxpr.eqns:
            n += e.primitive.name == "cond" and len(e.params["branches"]) == branches
            for sub in jax.extend.core.jaxprs_in_params(e.params):
                n += count(sub, branches)
        return n
    for T in (256, 4096, 8192):
        out = {}
        for ladder in ("", "262144/524288"):
            envs.TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER = ladder
            cj = jax.make_jaxpr(make())(*avals(T))
            out[ladder] = (cj.in_avals, cj.out_avals, count(cj.jaxpr, 3))
        (ia, oa, ca), (ib, ob, cb) = out[""], out["262144/524288"]
        assert ia == ib and oa == ob, (T, ia, ib, oa, ob)
        hp = not H.uses_dcp_attention(T, SEQS, 8)
        # The ladder adds one switch (2 rungs + the full buffer) on
        # head-parallel steps only.
        assert cb == ca + (1 if hp else 0), (T, ca, cb)
        print(T, hp, ca, cb, oa)
    print("OK")
    """
)


def test_round_ladder_trace_signature():
    r = subprocess.run(
        [sys.executable, "-c", _TRACE],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_heads_op_fake_symbolic_with_ladder(monkeypatch):
    """The torch op of `msa_dcp_heads.attend_and_merge` keeps its fake
    schema with the ladder on (the switch lives inside the JAX body)."""
    pytest.importorskip("torch_tpu")
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    monkeypatch.setattr(
        envs, "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER", "262144/524288", raising=False
    )
    op = msa_ops._get_msa_dcp_heads_op(
        128**-0.5 * 1.53125, 16, 4, 2, 8, "fake_hprb", scale_row0=4
    )
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(48),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    e = torch.empty
    with mode:
        num = seed.shape[0]
        args = (
            e((1,), dtype=torch.int32),
            e((num, 8, 128), dtype=torch.bfloat16),
            e((256, 16, 4, 512), dtype=torch.uint8),
            e((256, 8, 128), dtype=torch.float32),
            e((num, 64), dtype=torch.float32),
            e((64,), dtype=torch.int32),
            e((64 * 2048,), dtype=torch.int32),
            e((65,), dtype=torch.int32),
            e((3,), dtype=torch.int32),
        )
        out = op(*[mode.from_tensor(a) for a in args])
    assert out.shape == (num, 8, 128) and out.dtype == torch.bfloat16
    assert isinstance(out.shape[0], torch.SymInt)
