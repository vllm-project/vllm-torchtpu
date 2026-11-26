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
"""kernels/collectives/pair_lse_merge.py on 8 CPU devices: the merge math
against cp_lse_ag_out_rs's (torch, two ranks), the collective-permute merge
and the Pallas exchange (interpret mode) against the reference over disjoint
rank pairs (equal up to a rare one-ulp bf16 rounding), and a deviceless tpu7x compile of the kernel."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import pair_lse_merge as plm  # noqa: E402

# Head pairs of a DCP-4 draft head group over TP 8 (any disjoint pairs work).
PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
FAR_PAIRS = ((0, 4), (1, 5), (2, 6), (3, 7))


def _perm(pairs):
    return [(a, b) for p in pairs for a, b in (p, p[::-1])]


def _meta(pairs):
    """Per mesh index (peer, block) as the arming writes it."""
    meta = np.zeros((8, 2), np.int32)
    for p in pairs:
        for k in range(2):
            meta[p[k]] = (p[1 - k], k)
    return jnp.asarray(meta)


def _inputs(t, two_h, d, seed=0, dead_rows=()):
    """Every rank's partial [T, 2H, D] (bf16) and LSE [T, 2H] (f32),
    rank-major; ``dead_rows`` hold -inf LSEs on even ranks (a rank that
    holds none of the row's window) and NaN outs there."""
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((8, t, two_h, d)).astype(np.float32)
    lse = (rng.standard_normal((8, t, two_h)) * 3).astype(np.float32)
    for r in dead_rows:
        lse[::2, r] = -np.inf
        out[::2, r] = np.nan
    return jnp.asarray(out, jnp.bfloat16), jnp.asarray(lse)


def _want(out, lse, pairs):
    """Per rank, `reference` of its pair."""
    t, two_h, d = out.shape[1:]
    want_o = np.zeros((8, t, two_h // 2, d), np.float32)
    want_l = np.zeros((8, t, two_h // 2), np.float32)
    for p in pairs:
        for k in range(2):
            o, ol = plm.reference([out[p[0]], out[p[1]]], [lse[p[0]], lse[p[1]]], k)
            want_o[p[k]] = np.asarray(o.astype(jnp.float32))
            want_l[p[k]] = np.asarray(ol)
    return want_o, want_l


def _assert_near_exact(got_o, got_l, want_o, want_l):
    """The reference's result up to one bf16 rounding of a rare element: the
    same f32 formula, but exp / log under XLA's fusion (or Mosaic) may land
    one f32 ulp off, which flips a bf16 rounding about once in 1e5-1e6."""
    np.testing.assert_allclose(got_l, want_l, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_o, want_o, rtol=2**-7, atol=1e-6)
    assert np.mean(got_o != want_o) < 1e-4, np.mean(got_o != want_o)


def _mesh():
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices (XLA_FLAGS host device count)")
    return Mesh(np.array(jax.devices()[:8]), ("x",))


def _run(body, out, lse, meta):
    mesh = _mesh()
    f = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("x"),) * 3,
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
    )
    t, two_h, d = out.shape[1:]
    o, ol = f(out.reshape(8 * t, two_h, d), lse.reshape(8 * t, two_h), meta)
    return (
        np.asarray(o.astype(jnp.float32)).reshape(8, t, two_h // 2, d),
        np.asarray(ol).reshape(8, t, two_h // 2),
    )


def test_reference_matches_cp_lse_ag_out_rs_math():
    """`reference` is cp_lse_ag_out_rs's math (dcp_comm, torch) for a group of
    two: logsumexp of the LSEs, ``where(w > 0, out * w, 0)``, the f32 sum of
    the two terms, rank k keeping heads [k H, (k + 1) H)."""
    torch = pytest.importorskip("torch")
    out, lse = _inputs(24, 16, 128, dead_rows=(3,))
    outs = [torch.from_numpy(np.asarray(out[r].astype(jnp.float32))) for r in (0, 1)]
    lses = [torch.from_numpy(np.asarray(lse[r])) for r in (0, 1)]
    merged_lse = torch.logsumexp(torch.stack(lses), dim=0)
    terms = []
    for o, ls in zip(outs, lses):
        w = torch.exp(ls - merged_lse).unsqueeze(-1)
        terms.append(torch.where(w > 0, o * w, torch.zeros_like(o)))
    total = (terms[0] + terms[1]).to(torch.bfloat16).float().numpy()
    for k in range(2):
        o, ol = plm.reference([out[0], out[1]], [lse[0], lse[1]], k)
        sl = slice(8 * k, 8 * (k + 1))
        np.testing.assert_allclose(
            np.asarray(o.astype(jnp.float32)), total[:, sl], rtol=2**-7, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(ol), merged_lse.numpy()[:, sl], rtol=1e-6, atol=1e-6
        )


def test_merge_is_reference_bitwise():
    """The kernel's `_merge` (either operand order) equals `reference`."""
    out, lse = _inputs(24, 16, 128, seed=1, dead_rows=(0, 5))
    for k in range(2):
        want_o, want_l = plm.reference([out[0], out[1]], [lse[0], lse[1]], k)
        me, peer = k, 1 - k
        o, ol = plm._merge(
            out[me].astype(jnp.float32),
            lse[me][..., None],
            out[peer].astype(jnp.float32),
            lse[peer][..., None],
            jnp.int32(k),
            8,
        )
        np.testing.assert_array_equal(
            np.asarray(o.astype(jnp.bfloat16).astype(jnp.float32)),
            np.asarray(want_o.astype(jnp.float32)),
        )
        np.testing.assert_array_equal(np.asarray(ol[..., 0]), np.asarray(want_l))


@pytest.mark.parametrize("pairs", [PAIRS, FAR_PAIRS])
def test_exchange_xla_matches_reference(pairs):
    out, lse = _inputs(40, 16, 128, seed=2, dead_rows=(1,))
    got_o, got_l = _run(
        lambda o, ls, m: plm.exchange_xla(o, ls, m.reshape(-1)[1], "x", _perm(pairs)),
        out,
        lse,
        _meta(pairs),
    )
    _assert_near_exact(got_o, got_l, *_want(out, lse, pairs))


@pytest.fixture
def tpu7x(monkeypatch):
    # Mosaic tiling queries the TPU generation even in interpret mode.
    from jax._src import tpu_info

    monkeypatch.setattr(tpu_info, "get_device_kind", lambda: "TPU7x")
    monkeypatch.setattr(tpu_info, "get_num_device_cores", lambda: 1)


@pytest.mark.parametrize("pairs", [PAIRS, FAR_PAIRS])
@pytest.mark.parametrize("t", [16, 64])
def test_exchange_interpret_matches_reference(tpu7x, pairs, t):
    """The Pallas exchange (remote copies to the peer, local merge) in
    interpret mode gives every rank the reference result."""
    out, lse = _inputs(t, 16, 128, seed=3, dead_rows=(2,))
    assert plm.supported(out.shape[1:], out.dtype)
    try:
        got_o, got_l = _run(
            lambda o, ls, m: plm.exchange(o, ls, m, interpret=pltpu.InterpretParams()),
            out,
            lse,
            _meta(pairs),
        )
    except NotImplementedError as e:  # interpreter without remote DMA
        pytest.skip(f"interpret mode cannot run the remote copies: {e}")
    _assert_near_exact(got_o, got_l, *_want(out, lse, pairs))


def test_supported_cap():
    assert plm.supported((64, 16, 128), jnp.bfloat16)  # 256 KiB: a decode block
    assert not plm.supported((72, 16, 128), jnp.bfloat16)
    assert not plm.supported((16, 12, 128), jnp.bfloat16)
    assert not plm.supported((16, 16, 96), jnp.bfloat16)


@pytest.mark.parametrize("t", [16, 64])
def test_compiles_for_tpu7x(t):
    """Mosaic lowers the exchange for the served shard_map (deviceless tpu7x
    2x2x1, 8 TensorCores, adjacent pairs)."""
    from jax.experimental import topologies
    from jax.sharding import NamedSharding

    try:
        topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    mesh = Mesh(np.array(topo.devices), ("x",))
    f = jax.jit(
        jax.shard_map(
            plm.exchange,
            mesh=mesh,
            in_specs=(P("x"),) * 3,
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
    )
    sh = NamedSharding(mesh, P("x"))
    args = (
        jax.ShapeDtypeStruct((8 * t, 16, 128), jnp.bfloat16, sharding=sh),
        jax.ShapeDtypeStruct((8 * t, 16), jnp.float32, sharding=sh),
        jax.ShapeDtypeStruct((8, 2), jnp.int32, sharding=sh),
    )
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        f.lower(*args).compile()


# ---------------------------------------------------------------------------
# exchange_block (TPU_DRAFT_DCP_PAIR_MERGE_BLOCK): the exchange plus the draft
# block's own attention and the final merge in one kernel, against the unfused
# path (token-order partials, collective-permute merge, cache round trip of
# the block's K/V, cp_draft_attention.attend_block + merge).
# ---------------------------------------------------------------------------
Q_BLOCK = 8


def _block_inputs(num_reqs, t, two_h, d, seed, scaled):
    rng = np.random.default_rng(seed)
    h = two_h // 2

    def bf(*shape, scale=1.0):
        x = rng.standard_normal((8, *shape)) * scale
        return jnp.asarray(x, jnp.bfloat16)

    out = bf(t, two_h, d)
    lse = jnp.asarray(
        rng.standard_normal((8, num_reqs, 1, Q_BLOCK * two_h)) * 3, jnp.float32
    )
    q, k, v = bf(t, h, d, scale=0.3), bf(t, 1, d), bf(t, 1, d)
    if scaled:
        return out, lse, q, k, v, dict(k_scale=0.05, v_scale=0.03,
                                       cache_dtype=jnp.float8_e4m3fn)
    return out, lse, q, k, v, dict(k_scale=None, v_scale=None, cache_dtype=None)


def _unfused(out, lse, meta, q, k, v, *, num_reqs, perm, sm_scale, causal, quant):
    from vllm_torchtpu.layers.adapter import cp_draft_attention as cda

    t, two_h, _ = out.shape
    valid = jnp.arange(t) < num_reqs * Q_BLOCK
    out = jnp.where(valid[:, None, None], out, 0).astype(out.dtype)
    lse = cda._pad_rows(lse.reshape(-1, two_h), t, -jnp.inf)
    if quant["k_scale"] is not None:
        k = cda._round_trip(k, quant["k_scale"], quant["cache_dtype"])
        v = cda._round_trip(v, quant["v_scale"], quant["cache_dtype"])
    ctx, ctx_lse = plm.exchange_xla(out, lse, meta.reshape(-1)[1], "x", perm)
    seq_lens = jnp.zeros((num_reqs,), jnp.int32)
    out_b, lse_b = cda.attend_block(
        q, k, v, seq_lens, None, max_query_len=Q_BLOCK, sm_scale=sm_scale,
        causal=causal, uniform_blocks=True,
    )
    return cda.merge(ctx.astype(out_b.dtype), ctx_lse, out_b, lse_b)[0]


def _run_block(body, args):
    mesh = _mesh()
    f = jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=(P("x"),) * 6, out_specs=P("x"),
                      check_vma=False)
    )
    flat = [a.reshape(8 * a.shape[1], *a.shape[2:]) for a in args]
    got = f(*flat)
    return np.asarray(got.astype(jnp.float32)).reshape(8, -1, *got.shape[1:])


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_reqs,t", [(2, 18), (4, 32)])
def test_exchange_block_interpret_matches_unfused(tpu7x, scaled, causal, num_reqs, t):
    """Every rank's merged block output from the fused kernel (interpret mode)
    matches the unfused ops to bf16 roundings (the self-test's tolerance):
    the kernel computes the block attention in f32 over all heads at once and
    rounds where the unfused path hands bf16 tensors between ops."""
    pairs = PAIRS
    out, lse, q, k, v, quant = _block_inputs(num_reqs, t, 16, 128, 4, scaled)
    meta = _meta(pairs)
    sm_scale = 128**-0.5
    kw = dict(q_block=Q_BLOCK, sm_scale=sm_scale, causal=causal, **quant)
    try:
        got = _run_block(
            lambda o, ls, m, q_, k_, v_: plm.exchange_block(
                o, ls.reshape(num_reqs, 1, -1), m, q_, k_, v_, **kw,
                interpret=pltpu.InterpretParams(),
            ),
            (out, lse, meta.reshape(8, 1, 2), q, k, v),
        )
    except NotImplementedError as e:  # interpreter without remote DMA
        pytest.skip(f"interpret mode cannot run the remote copies: {e}")
    want = _run_block(
        lambda o, ls, m, q_, k_, v_: _unfused(
            o, ls.reshape(num_reqs, 1, -1), m, q_, k_, v_, num_reqs=num_reqs,
            perm=_perm(pairs), sm_scale=sm_scale, causal=causal, quant=quant,
        ),
        (out, lse, meta.reshape(8, 1, 2), q, k, v),
    )
    live = num_reqs * Q_BLOCK
    np.testing.assert_allclose(got[:, :live], want[:, :live], rtol=2**-6, atol=1e-2)
    err = np.abs(got[:, :live] - want[:, :live])
    # For the record: mostly equal, the rest bf16 roundings.
    assert np.mean(err == 0) > 0.5, np.mean(err == 0)


def test_exchange_block_compiles_for_tpu7x():
    """Mosaic lowers the fused block kernel (deviceless tpu7x 2x2x1) at a
    64-row draft bucket of 8 requests, int8-like cache round trip off."""
    from jax.experimental import topologies
    from jax.sharding import NamedSharding

    try:
        topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    mesh = Mesh(np.array(topo.devices), ("x",))
    t, num_reqs = 64, 8

    def body(o, ls, m, q_, k_, v_):
        return plm.exchange_block(o, ls, m, q_, k_, v_, q_block=Q_BLOCK,
                                  sm_scale=0.088, causal=False)

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("x"),) * 6,
                              out_specs=P("x"), check_vma=False))
    sh = NamedSharding(mesh, P("x"))

    def s(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=sh)

    args = (
        s((8 * t, 16, 128), jnp.bfloat16),
        s((8 * num_reqs, 1, Q_BLOCK * 16), jnp.float32),
        s((8, 2), jnp.int32),
        s((8 * t, 8, 128), jnp.bfloat16),
        s((8 * t, 1, 128), jnp.bfloat16),
        s((8 * t, 1, 128), jnp.bfloat16),
    )
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        f.lower(*args).compile()
