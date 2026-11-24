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
"""TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT and TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT /
_IN_FLIGHT: (host devices, interpret) the replicas of a KV head splitting the
query tiles, and the page exchange copying straight to every replica with a
different number of pages in flight, give the same output bits as the
defaults, at DCP 2, DCP 4 and DCP 8."""

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
    # `between` without TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC: the synchronous
    # exchange. (The asynchronous start / wait pair needs DMA semaphores that
    # outlive a kernel, which interpret mode does not model: TPU harness.)
    between = lambda q, cand: jnp.sum(q[:8].astype(jnp.float32)) + jnp.sum(cand[:8])
    os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT"] = "0"
    d = run(between)
    assert np.array_equal(a, d), float(np.abs(a - d).max())
    print("OK")
    """
)


@pytest.mark.parametrize("dcp,nreq", [(2, 1), (4, 1), (8, 1), (2, 2), (4, 2)])
def test_row_split_same_bits(dcp, nreq):
    r = subprocess.run(
        [sys.executable, "-c", _E2E, str(dcp), str(nreq)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
