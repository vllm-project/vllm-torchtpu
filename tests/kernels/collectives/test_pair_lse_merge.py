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
