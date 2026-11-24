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
"""TPU_MINIMAX_M3_DCP_SINGLE_HEAD_HP: (host devices, interpret) the one-KV-head
attend-merge op picks its path on the static step size inside JAX. A short
step gives the bits of `msa_dcp.attend_and_merge`; a long step attends
head-parallel (row split or not) and matches it to bf16 rounding."""

import os
import subprocess
import sys
import textwrap

_E2E = textwrap.dedent(
    """
    import os, sys
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = "8:16::64"
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = sys.argv[1]
    import functools, numpy as np, jax, jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_dcp_heads as H, msa_fp4
    from vllm_torchtpu.models.minimax_m3.ops import msa as ops_msa
    dcp, R, Hl, D, S, K = 2, 2, 8, 128, 128, 8
    nblk, C, nseq, pool, T, TREAL, DEC = 16, 8, 3, 20, 128, 104, 8
    rng = np.random.default_rng(0)
    mesh = jax.make_mesh((dcp,), ("d",), devices=jax.devices()[:dcp])
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("d")))
    rep = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = np.concatenate([np.asarray(msa_fp4.pack_rows(jnp.asarray(
        rng.integers(0, 16, (pool, S, 2 * D)), jnp.uint8))) for _ in range(dcp)])
    sc = rng.uniform(0.01, 0.05, (dcp * pool, msa_fp4.SCALE_ROWS, S)).astype(np.float32)
    q = msa_fp4.rotate_queries(jnp.asarray(
        rng.standard_normal((T, dcp * Hl, D)), jnp.bfloat16))
    q = jnp.asarray(q).transpose(1, 0, 2)  # head-major, rank r holds heads [r*Hl, (r+1)*Hl)
    bt = np.zeros((nseq, C), np.int32)
    bt[0] = rng.permutation(pool)[:C]; bt[1] = rng.permutation(pool)[:C]
    seq_lens = np.array([DEC, nblk * S, 0], np.int32)
    cu = np.array([0, DEC, TREAL, TREAL], np.int32)
    dist = np.array([1, 1, 2], np.int32)
    cand = np.zeros((dcp, T, 2 * K), np.float32)
    cand[:, :, :K] = -np.inf; cand[:, :, K:] = -1
    for t in range(DEC, TREAL):
        blocks = rng.choice(nblk, K, replace=False)
        scores = rng.standard_normal(K).astype(np.float32)
        for r in range(dcp):
            m = blocks % dcp == r
            cand[r, t, :m.sum()] = scores[m]; cand[r, t, K:K + m.sum()] = blocks[m]
    cand[:, :DEC, :K] = -np.inf
    had = jnp.asarray(msa_fp4.hadamard(D), jnp.bfloat16)
    def run(fn, n, **kw):
        specs = (P("d"),) * 4 + (P(),) * 3 + (P("d"), P()) + ((P(),) if kw else ())
        body = jax.jit(jax.shard_map(functools.partial(fn, axis="d", dcp=dcp, sm_scale=D ** -0.5,
                       topk=K, interpret=True, **kw), mesh=mesh, in_specs=specs,
                       out_specs=P("d"), check_vma=False))
        args = [put(np.arange(dcp, dtype=np.int32)), put(q[:, :n].reshape(dcp * Hl, n, D)),
                put(kv), put(cand[:, :n].reshape(dcp * n, 2 * K)), rep(seq_lens), rep(bt.reshape(-1)),
                rep(cu if n == T else np.minimum(cu, n)), put(sc), rep(had)]
        if kw:
            args.append(rep(dist))
        return np.asarray(body(*args).astype(jnp.float32))
    # Steps of >= 64 rows head-parallel here (2048 when served).
    real = H.uses_dcp_attention
    H.uses_dcp_attention = lambda n, m, d, h=64: real(n, m, d, 64)
    hp = dict(kv_head_replicas=R, decode_rows=DEC)
    ref_long = run(msa_dcp.attend_and_merge, T)
    n0 = len(ops_msa._SINGLE_HEAD_HP_TRACES)
    got_long = run(ops_msa._single_head_attend_and_merge, T, **hp)
    assert ops_msa._SINGLE_HEAD_HP_TRACES[n0:] == [T], ops_msa._SINGLE_HEAD_HP_TRACES
    rows = slice(0, TREAL)
    ref, got = ref_long.reshape(dcp, T, -1)[:, rows], got_long.reshape(dcp, T, -1)[:, rows]
    err = np.abs(ref - got).max(); scale = np.abs(ref).max()
    print("long", err, scale)
    assert scale > 0 and err <= 2e-2 * scale, (err, scale)
    # The decode rows keep the DCP attention: the same bits.
    assert np.array_equal(ref[:, :DEC], got[:, :DEC])
    n1 = len(ops_msa._SINGLE_HEAD_HP_TRACES)
    short = 32
    a = run(msa_dcp.attend_and_merge, short)
    b = run(ops_msa._single_head_attend_and_merge, short, **hp)
    assert len(ops_msa._SINGLE_HEAD_HP_TRACES) == n1
    assert np.array_equal(a, b)
    print("OK")
    """
)


def test_single_head_hp_choice():
    for row_split in ("0", "1"):
        r = subprocess.run(
            [sys.executable, "-c", _E2E, row_split],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=1800,
        )
        assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
