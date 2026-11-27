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
"""TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE: (host devices, interpret) the page
exchange with group-direct sends and chunked forwarding fills every live
slot of the page (and scale-row) buffer with the same bytes as the per-page
kernel, at DCP 4 / 8, partial / full / empty buffers and chunks that do not
divide the part; the head-parallel attention gives the same bits with and
without EXCHANGE_SCALES, the row split and two rounds; the flag is off by
default."""

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
    def run(between=None, row0=0):
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"),) * 5 + (P(),) * 4,
                           out_specs=P("d"), check_vma=False)
        def f(rank, q, kv, sc, sel, seq_lens, bt, cu, hp):
            tokens = msa.token_positions(T, seq_lens, cu)
            out = H._head_parallel_attend(
                q[0], kv[0], msa_fp4.with_row0(sc[0], row0), sel[0], seq_lens, bt.reshape(-1), cu, hp, tokens, axis="d",
                rank=rank[0], dcp=dcp, kv_head_replicas=R, sm_scale=D ** -0.5, topk=K,
                interpret=True, between=between)
            if between is not None:
                out, x = out
                out = out + 0 * x.astype(out.dtype)
            return out[None]
        return np.asarray(f(put(np.arange(dcp, dtype=np.int32)[:, None]), put(jnp.asarray(q, jnp.bfloat16)),
                            put(kv), put(sc), put(sel), rep(seq_lens), rep(bt), rep(cu),
                            rep(hp)).astype(jnp.float32))
    def on(**kw):
        for k in ("SCALES", "DIRECT", "IN_FLIGHT", "COALESCE"):
            os.environ.pop("TPU_MINIMAX_M3_DCP_EXCHANGE_" + k, None)
        os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = kw.pop("split", "0")
        for k, v in kw.items():
            os.environ["TPU_MINIMAX_M3_DCP_EXCHANGE_" + k] = v
    on(SCALES="1"); a = run()
    assert np.abs(a).sum() > 0
    assert not H._exchange_coalesce_logged
    for spec in ("1", "g", "c3", "g:c2"):
        on(SCALES="1", COALESCE=spec); b = run()
        assert np.array_equal(a, b), (spec, float(np.abs(a - b).max()))
    assert H._exchange_coalesce_logged
    on(COALESCE="1", IN_FLIGHT="3"); c = run()
    assert np.array_equal(a, c), float(np.abs(a - c).max())
    on(SCALES="1", COALESCE="1", split="1"); d = run()
    assert np.array_equal(a, d), float(np.abs(a - d).max())
    print("OK")
    """
)

_EXCHANGE = textwrap.dedent(
    """
    import os, sys
    dcp = int(sys.argv[1]); live_n = int(sys.argv[2])
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import functools, numpy as np, jax, jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4
    R, D, S = 2, 128, 128
    heads = dcp // R; R0 = 2 if heads <= 3 else 0
    pool, L = 24, 16  # buffer pages per rank (2 parts of 8)
    rng = np.random.default_rng(1)
    mesh = jax.make_mesh((dcp,), ("d",), devices=jax.devices()[:dcp])
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("d")))
    rep = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    kv = rng.integers(0, 256, (dcp, pool, S // 8, 4, heads * 2 * D)).astype(np.uint8)
    sc = rng.uniform(0.01, 0.05, (dcp, pool, msa_fp4.SCALE_ROWS, S)).astype(np.float32)
    pages = rng.permutation(pool)[:L].astype(np.int32)
    def run(scales, **kw):
        @jax.jit
        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P("d"), P("d"), P("d"), P(), P()), out_specs=P("d"), check_vma=False)
        def f(rank, kv, sc, pages, live):
            r = H.exchange_head_pages(kv[0], pages, live, rank[0], axis="d", dcp=dcp, replicas=R, head_width=2 * D,
                                      interpret=True, in_flight=3, scale_cache=sc[0] if scales else None, scale_row0=R0 if scales else 0, **kw)
            if scales:
                return r[0][None], r[1][None]
            return r[None], jnp.zeros((1, 1), jnp.float32)
        a, b = f(put(np.arange(dcp, dtype=np.int32)[:, None]), put(kv), put(sc), rep(pages), rep(np.array([live_n], np.int32)))
        return np.asarray(a), np.asarray(b)
    part = L // R
    def live_mask():
        m = np.zeros((R, part), bool)
        for j in range(R):
            m[j, : int(np.clip((live_n - j + R - 1) // R, 0, part))] = True
        return m
    m = live_mask()
    for scales in (False, True):
        a, sa = run(scales)
        # reference check of the base too
        for me in range(dcp):
            h = me // R
            for j in range(R):
                for src in range(dcp):
                    for i in range(part):
                        if m[j, i]:
                            assert np.array_equal(a[me, j, src, i], kv[src, pages[j * part + i], :, :, h * 2 * D:(h + 1) * 2 * D])
                            if scales:
                                assert np.array_equal(sa[me, j, src, i, :2], sc[src, pages[j * part + i], R0 + 2 * h: R0 + 2 + 2 * h])
        for kw in (dict(group_direct=True), dict(fwd_chunk=3), dict(fwd_chunk=4), dict(group_direct=True, fwd_chunk=3),
                   dict(group_direct=True, fwd_chunk=32), dict(group_direct=True, fwd_chunk=1)):
            b, sb = run(scales, **kw)
            # a[:, j, src, i]: mask over (j, i) -> index with broadcasting
            A = a.transpose(0, 2, 1, 3, 4, 5, 6)[:, :, m]; B = b.transpose(0, 2, 1, 3, 4, 5, 6)[:, :, m]
            assert np.array_equal(A, B), (scales, kw)
            if scales:
                SA = sa.transpose(0, 2, 1, 3, 4, 5)[:, :, m][..., :2, :]; SB = sb.transpose(0, 2, 1, 3, 4, 5)[:, :, m][..., :2, :]
                assert np.array_equal(SA, SB), (scales, kw)
            print("ok", scales, kw, flush=True)
    print("OK")
    """
)


@pytest.mark.parametrize("dcp,nreq", [(4, 1), (4, 2)])
def test_exchange_coalesce_attention_same_bits(dcp, nreq):
    r = subprocess.run(
        [sys.executable, "-c", _E2E, str(dcp), str(nreq)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


@pytest.mark.parametrize("dcp,live", [(4, 13), (8, 11), (4, 16), (4, 1), (4, 0)])
def test_exchange_coalesce_same_slots(dcp, live):
    r = subprocess.run(
        [sys.executable, "-c", _EXCHANGE, str(dcp), str(live)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_exchange_coalesce_spec(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H

    monkeypatch.delenv("TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE", raising=False)
    assert envs.TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE == ""
    assert H.exchange_coalesce() == (False, 0)
    for spec, want in (("0", (False, 0)), ("1", (True, H._EXCHANGE_FWD_CHUNK)),
                       ("g", (True, 0)), ("c16", (False, 16)), ("g:c64", (True, 64))):
        monkeypatch.setenv("TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE", spec)
        assert H.exchange_coalesce() == want, spec
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE", "x")
    with pytest.raises(ValueError):
        H.exchange_coalesce()
