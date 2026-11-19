# SPDX-License-Identifier: Apache-2.0
"""DEC-COLL: the DCP head-slice all-to-all against today's all-gather + slice combine (CPU, bitwise).

Today's path (``qsa_op._dcp_finish`` + ``qsa_rpa.dcp_combine``): all-gather every member's partial output
[T, size*H, D] f32 and LSE [T, size*H] over the DCP group, then slice this rank's H heads and merge. New paths:
pack each destination's head slice (output + LSE in one block), all-to-all (XLA ``lax.all_to_all``, or the Pallas
remote-DMA kernel in the Pallas TPU interpreter with remote DMAs and semaphores modelled), unpack, and run the
shared merge (``dcp_combine_heads``). Every member's output must be bitwise today's.

Mesh: 8 host devices as ("kv", "dcp") = (2, 4) in the engine's group order ({0,1,4,5}, {6,7,2,3}), so two DCP
groups run side by side, as on O1 (TP8 = 2 KV-head groups x DCP4). Geometry: H = 3 heads per rank, D = 256
(O1: 12 group heads), T = 16 / 64 / 128 rows.

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import dcp_head_a2a as A  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import dcp_combine, dcp_combine_heads  # noqa: E402

KV, SIZE, H, D = 2, 4, 3, 256
ENGINE_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)
SPEC = P("kv", "dcp")


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < 8:
        pytest.skip("needs 8 host devices (see module docstring)")
    devs = {d.id: d for d in jax.devices()}
    return Mesh(np.asarray([devs[i] for i in ENGINE_ORDER]).reshape(KV, SIZE), ("kv", "dcp"))


def _inputs(tokens, seed, kind="nrm"):
    """Per device [kv, dcp, ...]: out [T, size*H, D] f32, lse [T, size*H] f32, gate [T, H, D] bf16."""
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((KV, SIZE, tokens, SIZE * H, D)).astype(np.float32)
    lse = (rng.standard_normal((KV, SIZE, tokens, SIZE * H)) * 4).astype(np.float32)
    gate = rng.standard_normal((KV, SIZE, tokens, H, D)).astype(np.float32)
    if kind == "adv":
        # A member that owns no key of a row (-inf), rows nobody owns (all -inf -> 0 output), a large LSE spread,
        # ties, +-0 and huge / tiny partials.
        lse[:, 1, ::3, :] = -np.inf
        lse[:, :, 5, 4] = -np.inf
        lse[:, :, 7, :] = 80.0
        lse[:, 2, 9, :] = -80.0
        lse[:, :, 11, 2] = 3.0
        out[:, :, 13, :, :] = 0.0
        out[:, 3, 13, :, ::2] = -0.0
        out[:, 0, 6, :, :] *= 1e30
        out[:, 1, 6, :, :] *= 1e-30
    return out, lse, gate


def _put(mesh, a, dtype=None):
    a = jnp.asarray(a if dtype is None else a.astype(dtype))
    return jax.device_put(a, NamedSharding(mesh, SPEC))


def _today(mesh):
    def body(out, lse, gate):
        out, lse, gate = out[0, 0], lse[0, 0], gate[0, 0]
        out_pair = jax.lax.all_gather(out, "dcp")
        lse_pair = jax.lax.all_gather(lse, "dcp")
        rank = jax.lax.axis_index("dcp").reshape(1).astype(jnp.int32)
        return dcp_combine(out_pair, lse_pair, gate, rank)[None, None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(SPEC,) * 3, out_specs=SPEC, check_vma=False))


def _a2a(mesh, transport, calls=1):
    def one(out, lse, gate):
        x = A.pack_head_slices(out, lse, SIZE)
        if transport == "xla":
            recv = A.xla_head_a2a_local(x, axis="dcp")
        else:
            recv = A.dcp_head_a2a_local(x, axis="dcp", size=SIZE, interpret=True)
        out_sel, lse_sel = A.unpack_head_slices(recv, H, D)
        return dcp_combine_heads(out_sel, lse_sel, gate)

    def body(out, lse, gate):
        out, lse, gate = out[0, 0], lse[0, 0], gate[0, 0]
        res = [one(out, lse, gate)]
        for k in range(1, calls):  # back-to-back calls: barrier and semaphore reuse in one program
            res.append(one(out * (k + 1), lse, gate))
        return jnp.stack(res)[None, None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(SPEC,) * 3, out_specs=SPEC, check_vma=False))


def _recv_only(mesh, transport):
    def body(out, lse):
        x = A.pack_head_slices(out[0, 0], lse[0, 0], SIZE)
        if transport == "xla":
            recv = A.xla_head_a2a_local(x, axis="dcp")
        else:
            recv = A.dcp_head_a2a_local(x, axis="dcp", size=SIZE, interpret=True)
        o, s = A.unpack_head_slices(recv, H, D)
        return o[None, None], s[None, None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(SPEC,) * 2, out_specs=(SPEC, SPEC), check_vma=False))


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint16) if a.dtype == jnp.bfloat16 else a.astype(np.float32).view(np.uint32)


def _run_today(mesh, out, lse, gate):
    return np.asarray(_today(mesh)(_put(mesh, out), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16)))


@pytest.mark.parametrize("transport", ["xla", "dma"])
@pytest.mark.parametrize("tokens", [16, 64, 128])
def test_delivers_exactly_the_slice_today_selects(mesh, transport, tokens):
    """Pure data movement: member r of group g receives, at index i, member i's heads [r*H, (r+1)*H), bit for bit."""
    out, lse, _ = _inputs(tokens, seed=tokens, kind="adv")
    o, s = _recv_only(mesh, transport)(_put(mesh, out), _put(mesh, lse))
    o, s = np.asarray(o), np.asarray(s)
    for g in range(KV):
        for r in range(SIZE):
            want_o = out[g, :, :, r * H:(r + 1) * H, :]
            want_s = lse[g, :, :, r * H:(r + 1) * H]
            assert np.array_equal(_bits(o[g, r]), _bits(want_o)), f"output slice differs at group {g} member {r}"
            assert np.array_equal(_bits(s[g, r]), _bits(want_s)), f"LSE slice differs at group {g} member {r}"


@pytest.mark.parametrize("transport", ["xla", "dma"])
@pytest.mark.parametrize("kind", ["nrm", "adv"])
@pytest.mark.parametrize("tokens", [16, 64, 128])
def test_combine_bitwise_today(mesh, transport, kind, tokens):
    out, lse, gate = _inputs(tokens, seed=1000 + tokens, kind=kind)
    want = _run_today(mesh, out, lse, gate)
    got = np.asarray(_a2a(mesh, transport)(_put(mesh, out), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16)))[:, :, 0]
    assert got.shape == want.shape == (KV, SIZE, tokens, H, D)
    bad = int((_bits(got) != _bits(want)).sum())
    assert bad == 0, f"{bad} of {want.size} outputs differ from today's combine"
    assert np.isfinite(np.asarray(want, np.float32)).all()
    if kind == "adv":  # rows nobody owns stay exactly 0 (and some exist)
        dead = np.all(~np.isfinite(lse), axis=1)  # [kv, T, size*H]: no member owns a key
        assert dead.any()
        for g in range(KV):
            for r in range(SIZE):
                m = dead[g][:, r * H:(r + 1) * H]
                assert not np.asarray(got[g, r], np.float32)[m].any()


def test_back_to_back_calls_one_program(mesh):
    out, lse, gate = _inputs(64, seed=7)
    args = (_put(mesh, out), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16))
    got = np.asarray(_a2a(mesh, "dma", calls=3)(*args))
    for k in range(3):
        want = _run_today(mesh, out * (k + 1), lse, gate)
        assert np.array_equal(_bits(got[:, :, k]), _bits(want)), f"call {k} differs"


@pytest.mark.parametrize("transport", ["xla", "dma"])
def test_positive_control(mesh, transport):
    """One moved partial changes exactly its head's owner, on both paths alike."""
    out, lse, gate = _inputs(64, seed=11)
    base = np.asarray(_a2a(mesh, transport)(_put(mesh, out), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16)))[:, :, 0]
    moved = out.copy()
    moved[0, 2, 5, 7, 11] += 64.0  # group 0, member 2's partial, row 5, head 7 (owned by member 7 // H = 2), lane 11
    got = np.asarray(_a2a(mesh, transport)(_put(mesh, moved), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16)))[:, :, 0]
    want = _run_today(mesh, moved, lse, gate)
    assert np.array_equal(_bits(got), _bits(want))
    changed = [(g, r) for g in range(KV) for r in range(SIZE) if not np.array_equal(_bits(got[g, r]), _bits(base[g, r]))]
    assert changed == [(0, 2)], changed
    diff = np.argwhere(_bits(got[0, 2]) != _bits(base[0, 2]))
    assert diff.tolist() == [[5, 7 - 2 * H, 11]], diff.tolist()


def test_pack_layout():
    rng = np.random.default_rng(3)
    out = rng.standard_normal((8, SIZE * H, D)).astype(np.float32)
    lse = rng.standard_normal((8, SIZE * H)).astype(np.float32)
    x = np.asarray(A.pack_head_slices(jnp.asarray(out), jnp.asarray(lse), SIZE))
    assert x.shape == (SIZE, 8, A.packed_width(H, D)) == (SIZE, 8, H * D + 128)
    for j in range(SIZE):
        assert np.array_equal(x[j, :, : H * D], out[:, j * H:(j + 1) * H, :].reshape(8, H * D))
        assert np.array_equal(x[j, :, H * D: H * D + H], lse[:, j * H:(j + 1) * H])
        assert not x[j, :, H * D + H:].any()


def test_shape_guards():
    with pytest.raises(ValueError):
        A.packed_width(3, 100)  # 300 lanes: not whole tiles
    with pytest.raises(ValueError):
        A.pack_head_slices(jnp.zeros((4, 10, 256)), jnp.zeros((4, 10)), 4)  # 10 heads over 4 members


def _kernel_eqns(mesh, size=SIZE):
    f = jax.shard_map(lambda x: A.dcp_head_a2a_local(x[0, 0], axis="dcp", size=size, interpret=True)[None, None],
                      mesh=mesh, in_specs=SPEC, out_specs=SPEC, check_vma=False)
    jp = jax.make_jaxpr(f)(jnp.zeros((KV, SIZE, size, 8, A.packed_width(H, D)), jnp.float32))

    def find(j):
        for e in j.eqns:
            if e.primitive.name == "pallas_call":
                return e.params["jaxpr"]
            for p in e.params.values():
                jj = getattr(p, "jaxpr", p)
                if hasattr(jj, "eqns"):
                    r = find(jj)
                    if r is not None:
                        return r
        return None

    k = find(jp.jaxpr)
    assert k is not None, "no pallas_call in the traced program"
    return k.eqns


# Scalar ops the barrier-target evaluator below folds; every other primitive is left unevaluated.
_SCALAR_OPS = {"add", "sub", "mul", "rem", "neg", "max", "min", "convert_element_type"}


def _signal_targets(eqns, me):
    """Fold the kernel's scalar integer program for member ``me`` (``axis_index`` -> me) and return the device id
    of every ``semaphore_signal`` in program order. jax 0.11 binds semaphore_signal on the flattened
    ``[ref, transforms, inc, device_id, core_index]`` (``args_tree``). DEC-O4-4 [plan-v4 DEC-COLL CP2]: the Pallas
    interpreter completes a barrier whose signals go to the member itself, so only this static check sees it."""
    env = {}

    def val(v):
        return np.asarray(v.val, v.aval.dtype) if type(v).__name__ == "Literal" else env.get(v)

    out = []
    for e in eqns:
        name = e.primitive.name
        if name == "axis_index":
            env[e.outvars[0]] = np.asarray(me, e.outvars[0].aval.dtype)
        elif name == "semaphore_signal":
            _, _, _, dev, _ = jax.tree_util.tree_unflatten(e.params["args_tree"], list(e.invars))
            leaves = jax.tree_util.tree_leaves(dev)
            assert len(leaves) == 1, f"one mesh-axis device id expected, got {dev}"
            v = val(leaves[0])
            assert v is not None, "a signal's device id does not fold to a constant"
            out.append(int(v))
        elif name in _SCALAR_OPS and len(e.outvars) == 1:
            ins = [val(v) for v in e.invars]
            if all(x is not None and np.ndim(x) == 0 for x in ins):
                env[e.outvars[0]] = np.asarray(e.primitive.bind(*ins, **e.params))
    return out


def test_schedule_protocol(mesh):
    """The barrier round precedes every DMA: one signal per peer, one wait for exactly that many, then 1 local +
    (size-1) remote DMAs, then (size-1) send waits, (size-1) receive waits and the local wait."""
    names = [e.primitive.name for e in _kernel_eqns(mesh)]
    sig = [i for i, n in enumerate(names) if n == "semaphore_signal"]
    wait = [i for i, n in enumerate(names) if n == "semaphore_wait"]
    starts = [i for i, n in enumerate(names) if n == "dma_start"]
    dwaits = [i for i, n in enumerate(names) if n == "dma_wait"]
    assert len(sig) == SIZE - 1 and len(wait) == 1, (len(sig), len(wait))
    assert max(sig) < wait[0] < min(starts), "a DMA starts before the barrier round completes"
    assert len(starts) == SIZE and len(dwaits) == 2 * (SIZE - 1) + 1, (len(starts), len(dwaits))
    assert min(dwaits) > max(starts)
    eq = _kernel_eqns(mesh)[wait[0]]
    counts = [int(np.asarray(getattr(v, "val"))) for v in eq.invars if type(v).__name__ == "Literal"
              and np.asarray(getattr(v, "val")).dtype == np.int32]
    assert counts == [SIZE - 1], counts


def test_barrier_signals_go_to_every_peer(mesh):
    """Every member signals each of its size-1 peers once and never itself (the interpreter cannot see a self-signal:
    the barrier still completes, so the bitwise tests pass while the device would race)."""
    eqns = _kernel_eqns(mesh)
    for me in range(SIZE):
        t = _signal_targets(eqns, me)
        assert me not in t, f"member {me} signals itself: {t}"
        assert sorted(t) == sorted((me + i) % SIZE for i in range(1, SIZE)), (me, t)


def test_combine_matches_independent_model(mesh):
    """Guards the shared merge itself (an edit to it would change both paths alike): the new path's output is
    within bf16 rounding of a numpy model of LSE-weighted merge + sigmoid gate."""
    out, lse, gate = _inputs(64, seed=21)
    gate_b = np.asarray(jnp.asarray(gate, jnp.bfloat16).astype(jnp.float32))
    got = np.asarray(_a2a(mesh, "xla")(_put(mesh, out), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16)))[:, :, 0]
    for g in range(KV):
        for r in range(SIZE):
            o = out[g, :, :, r * H:(r + 1) * H, :].astype(np.float64)  # [size, T, H, D]
            s = lse[g, :, :, r * H:(r + 1) * H].astype(np.float64)
            w = np.exp(s - s.max(axis=0))
            ref = (w[..., None] * o).sum(0) / w.sum(0)[..., None] / (1 + np.exp(-gate_b[g, r]))
            np.testing.assert_allclose(np.asarray(got[g, r], np.float32), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens", [16, 64])
def test_engine_steps_bitwise_today(mesh, tokens):
    """The two jax_op bodies the engine wiring uses (pack -> flat all-to-all -> merge) equal today's combine."""
    out, lse, gate = _inputs(tokens, seed=300 + tokens, kind="adv")

    def body(o, s, g):
        o, s, g = o[0, 0], s[0, 0], g[0, 0]
        flat = A.dcp_a2a_pack_step(o, s, g)
        recv = jax.lax.all_to_all(flat, "dcp", split_axis=0, concat_axis=0, tiled=True)
        return A.dcp_a2a_merge_step(recv, g)[None, None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(SPEC,) * 3, out_specs=SPEC, check_vma=False))
    got = np.asarray(f(_put(mesh, out), _put(mesh, lse), _put(mesh, gate, jnp.bfloat16)))
    want = _run_today(mesh, out, lse, gate)
    assert np.array_equal(_bits(got), _bits(want))
