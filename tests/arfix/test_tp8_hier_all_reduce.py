# SPDX-License-Identifier: Apache-2.0
"""DEC-FB-2 lever 12: the hierarchical TP8 all-reduce kernel (H8) against its declared summation order (CPU).

The kernel runs in the Pallas TPU interpreter on 8 host devices under shard_map (remote DMAs, DMA and barrier
semaphores modelled), at Qwen3.8-Flash's hidden 2560 and the served decode row counts (MTP k=3: 64 / 112 / 192 / 256
rows at B16 / B28 / B48 / B64, plus 16). Variants: all-gather by forwarding (fwd) or direct, pair sums on a bf16 or an
f32 wire. Checks: every member's output is bitwise the numpy model of the order (``reference``), all 8 members are
bitwise equal, adversarial columns (order-sensitive exponent spread, large cancellation, +-0, subnormals, overflow),
back-to-back calls in one program (semaphore and barrier reuse), an arbitrary chip placement, a positive control and
the XLA fallback. The f32 wire must also be bitwise the single-rounding f32 reference.

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8  # noqa: E402

H = 2560
VARIANTS = [("fwd", "bf16"), ("direct", "bf16"), ("fwd", "f32")]
VIDS = [f"{a}-{w}" for a, w in VARIANTS]
SCRAMBLED = ((3, 6), (0, 5), (7, 2), (1, 4))


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < 8:
        pytest.skip("needs 8 host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:8]), ("tp",))


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def _run(mesh, x, ag="fwd", wire="bf16", pairs=h8.DEFAULT_PAIRS, interpret=True, max_rows=h8.MAX_ROWS):
    xs = jax.device_put(jnp.asarray(x, jnp.bfloat16), NamedSharding(mesh, P("tp")))
    out = jax.jit(lambda a: h8.tp8_hier_all_reduce(a, mesh=mesh, axis="tp", pairs=pairs, ag=ag, wire=wire,
                                                   max_rows=max_rows, interpret=interpret))(xs)
    return np.asarray(out).astype(np.float32)


def _check(out, ref):
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8)), "members disagree"
    bad = int((_bits(out[0]) != _bits(ref)).sum())
    assert bad == 0, f"{bad} of {ref.size} output elements differ from the declared order"


def _normal(rows, seed, hidden=H):
    rng = np.random.default_rng(seed)
    return h8._bf16_round(rng.standard_normal((8, rows, hidden)).astype(np.float32))


def _adversarial(rows, pairs=h8.DEFAULT_PAIRS, seed=7):
    """Random partials with planted columns (each pattern in every row, on disjoint columns col % 13 == 0..6,
    spread over all shards)."""
    x = _normal(rows, seed)
    member = {(c, k): pairs[c][k] for c in range(4) for k in range(2)}

    def put(col, vals):  # vals[c][k] -> member (c, k)'s value in column col
        for c in range(4):
            for k in range(2):
                x[member[(c, k)], :, col] = vals[c][k]

    cols = np.arange(H)
    for col in cols[cols % 13 == 0]:  # order-sensitive exponent spread: in chip order (((2^24 + 3) - 2^24) + 1/8) = 4.125
        put(col, [[2.0 ** 23, 2.0 ** 23], [1.0, 2.0], [-(2.0 ** 23), -(2.0 ** 23)], [0.0625, 0.0625]])
    for col in cols[cols % 13 == 1]:  # big terms on chips 1 and 3: ((1/8 + 2^24) + 3) - 2^24 = 4 in chip order
        put(col, [[0.0625, 0.0625], [2.0 ** 23, 2.0 ** 23], [1.0, 2.0], [-(2.0 ** 23), -(2.0 ** 23)]])
    for col in cols[cols % 13 == 2]:  # large cancellation across chips, small remainder on the last chip
        put(col, [[1000.0, 24.0], [-1024.0, 0.0], [3.0, -1.0], [0.0078125, 0.001953125]])
    for col in cols[cols % 13 == 3]:  # every member -0: the sum is -0
        put(col, [[-0.0, -0.0]] * 4)
    for col in cols[cols % 13 == 4]:  # mixed signed zeros: the sum is +0
        put(col, [[-0.0, 0.0], [-0.0, -0.0], [0.0, -0.0], [-0.0, -0.0]])
    for col in cols[cols % 13 == 5]:  # subnormals (bf16 keeps f32's exponent range); flushed by the adds -> 2^-126
        put(col, [[2.0 ** -130, 2.0 ** -133], [-(2.0 ** -131), 2.0 ** -129], [2.0 ** -126, -(2.0 ** -127)],
                  [2.0 ** -133, 2.0 ** -133]])
    for col in cols[cols % 13 == 6]:  # chip 0's pair sum overflows to +inf; the sum is +inf
        put(col, [[1.875 * 2.0 ** 127, 1.875 * 2.0 ** 127], [1.0, 1.0], [2.0, -1.0], [0.5, 0.25]])
    return h8._bf16_round(x)


def _ref(x, wire, pairs=h8.DEFAULT_PAIRS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return h8.reference(x, pairs, wire)


def test_shard_table():
    t = h8.shard_table(2560)
    assert t == (((0, 384), (384, 384), (768, 256), (1024, 256)), ((1280, 256), (1536, 256), (1792, 384), (2176, 384)))
    for hidden in (1024, 1280, 2560, 4096):
        t = h8.shard_table(hidden)
        flat = sorted(sw for row in t for sw in row)
        assert flat[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in zip(flat, flat[1:]))
        assert flat[-1][0] + flat[-1][1] == hidden and all(s % 128 == 0 and w % 128 == 0 and w > 0 for s, w in flat)
        assert all(t[k][c][0] // (hidden // 2) == k for k in range(2) for c in range(4))
    per_chip = [h8.shard_table(2560)[0][c][1] + h8.shard_table(2560)[1][c][1] for c in range(4)]
    assert per_chip == [640] * 4  # every chip finishes 5 lane tiles
    for bad in (768, 2560 + 128, 1000):
        with pytest.raises(ValueError):
            h8.shard_table(bad)


def test_supports():
    assert all(h8.supports(r, 2560, jnp.bfloat16) for r in (16, 64, 112, 192, 256))
    assert not any(h8.supports(r, 2560, jnp.bfloat16) for r in (0, 8, 100, 272))
    assert not h8.supports(64, 2560, jnp.float32) and not h8.supports(64, 2688, jnp.bfloat16)


class _Dev:
    def __init__(self, i, coords=None, core=None):
        self.id = i
        if coords is not None:
            self.coords, self.core_on_chip = coords, core


def test_placement_from_devices():
    # The deviceless tpu7x:2x2x1 topology (VM, jax 0.11.2 / libtpu 0.0.50): ids 2c, 2c+1 share coords.
    coords = [(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 1, 0), (1, 1, 0)]
    devs = [_Dev(i, coords[i], i % 2) for i in range(8)]
    assert h8.placement_from_devices(devs) == ((0, 1), (4, 5), (2, 3), (6, 7))
    perm = [5, 2, 7, 0, 3, 6, 1, 4]  # mesh index i holds device perm[i]
    pairs = h8.placement_from_devices([devs[j] for j in perm])
    assert sorted(i for p in pairs for i in p) == list(range(8))
    assert all(coords[perm[a]] == coords[perm[b]] and perm[a] % 2 == 0 and perm[b] % 2 == 1 for a, b in pairs)
    assert h8.placement_from_devices([_Dev(i) for i in range(8)]) == h8.DEFAULT_PAIRS  # id fallback
    with pytest.raises(ValueError):
        h8.placement_from_devices([_Dev(i, (i // 4, 0, 0), i % 4) for i in range(8)])  # 4 cores per chip
    with pytest.raises(ValueError):
        h8.check_pairs(((0, 1), (2, 3), (4, 5), (6, 6)))


@pytest.mark.parametrize("ag,wire", VARIANTS, ids=VIDS)
@pytest.mark.parametrize("rows", [16, 64, 112, 192, 256])
def test_kernel_bitwise_declared_order(mesh, rows, ag, wire):
    x = _normal(rows, rows)
    out = _run(mesh, x, ag, wire)
    _check(out, _ref(x, wire))
    if wire == "f32":
        _check(out, h8.reference_f32(x))
    assert (out[0] != 0).mean() > 0.99


@pytest.mark.parametrize("ag,wire", VARIANTS, ids=VIDS)
def test_adversarial_columns(mesh, ag, wire):
    x = _adversarial(64)
    out = _run(mesh, x, ag, wire)
    ref = _ref(x, wire)
    _check(out, ref)
    # the planted patterns really are what the test claims
    c = np.arange(H) % 13
    assert np.all(ref[:, c == 0] == 4.125) and np.all(ref[:, c == 1] == 4.0)  # exact sum 3.125 in both
    assert np.all(np.signbit(ref[:, c == 3])) and np.all(ref[:, c == 3] == 0)
    assert np.all(~np.signbit(ref[:, c == 4])) and np.all(ref[:, c == 4] == 0) and np.all(np.isposinf(ref[:, c == 6]))
    # subnormals: the adds flush them (XLA:CPU, measured), so only chip 2's normal 2^-126 survives; without the
    # flush the IEEE sum differs, so this column pins the flush model
    assert np.all(ref[:, c == 5] == 2.0 ** -126)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.all(h8.reference(x, wire=wire, ftz=False)[:, c == 5] != 2.0 ** -126)


def test_order_patterns_distinguish_orders():
    """The planted columns change under another summation order, so an order bug cannot pass silently."""
    x = _adversarial(16)
    ref = _ref(x, "bf16")
    xr = x[[6, 7, 4, 5, 2, 3, 0, 1]]  # chips in reverse order
    assert int((_bits(_ref(xr, "bf16")) != _bits(ref)).sum()) > 0
    chain = h8._bf16_round(x)
    acc = np.zeros_like(chain[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in range(8):  # a plain member-order chain in bf16
            acc = h8._bf16_round(acc + chain[r])
    assert int((_bits(acc) != _bits(ref)).sum()) > 0


@pytest.mark.parametrize("ag,wire", VARIANTS, ids=VIDS)
def test_back_to_back_calls(mesh, ag, wire):
    """Three calls in one program, each input = previous output + a fresh per-member partial (barrier and
    semaphore reuse across calls)."""
    rows = 32
    xs = [_normal(rows, 100 + i) for i in range(3)]
    sh = NamedSharding(mesh, P("tp"))

    def prog(a0, a1, a2):
        out = []
        cur = a0
        for nxt in (a1, a2, None):
            y = h8.tp8_hier_all_reduce(cur, mesh=mesh, axis="tp", ag=ag, wire=wire, interpret=True)
            out.append(y)
            if nxt is not None:
                cur = (y.astype(jnp.float32) + nxt.astype(jnp.float32)).astype(jnp.bfloat16)
        return out

    outs = jax.jit(prog)(*(jax.device_put(jnp.asarray(a, jnp.bfloat16), sh) for a in xs))
    cur = xs[0]
    for i, y in enumerate(outs):
        ref = _ref(cur, wire)
        _check(np.asarray(y).astype(np.float32), ref)
        if i < 2:
            cur = h8._bf16_round(ref[None] + xs[i + 1])


@pytest.mark.parametrize("ag,wire", [("fwd", "bf16"), ("direct", "f32")], ids=["fwd-bf16", "direct-f32"])
def test_scrambled_placement(mesh, ag, wire):
    x = _adversarial(48, SCRAMBLED, seed=3)
    out = _run(mesh, x, ag, wire, pairs=SCRAMBLED)
    _check(out, _ref(x, wire, SCRAMBLED))


@pytest.mark.parametrize("hidden", [1024, 1280])
def test_other_hidden(mesh, hidden):
    x = _normal(32, hidden, hidden)
    _check(_run(mesh, x, "fwd", "bf16"), _ref(x, "bf16"))


def test_positive_control(mesh):
    x = _normal(64, 9)
    base = _run(mesh, x)
    x2 = x.copy()
    x2[5, 7, 1000] += 64.0
    moved = _run(mesh, x2)
    diff = _bits(base) != _bits(moved)
    assert diff[:, 7, 1000].all() and int(diff.sum()) == 8


def test_fallback_is_xla_sum(mesh):
    x = _normal(272, 4)  # above max rows
    out = _run(mesh, x)
    assert all(np.array_equal(_bits(out[0]), _bits(out[r])) for r in range(8))
    np.testing.assert_allclose(out[0], x.astype(np.float64).sum(0), rtol=0, atol=0.25)


def test_precision_of_the_two_wires():
    """Characterisation (numpy): the f32 wire is within 1 ulp of bf16(exact sum) everywhere on N(0, 1) partials;
    the bf16 wire's pair rounding puts a sizeable fraction of elements more than 1 ulp off the single-rounding
    f32 reference (the measured fraction is reported to DEC-FB-2: about 12%)."""
    x = _normal(256, 1)
    exact = h8._bf16_round(x.astype(np.float64).sum(0))
    d32 = h8.ulp_distance(h8.reference(x, wire="f32"), exact)
    d16 = h8.ulp_distance(h8.reference(x, wire="bf16"), h8.reference_f32(x))
    assert d32.max() <= 1
    assert 0.05 < (d16 > 1).mean() < 0.20


def test_xla_arms_compute_the_sum(mesh):
    x = _normal(64, 11)
    pairs = h8.placement_from_devices(list(mesh.devices.flat))
    m2 = h8.chip_core_mesh(list(mesh.devices.flat), pairs)
    grid = np.stack([np.stack([x[a], x[b]]) for a, b in pairs])  # [4, 2, R, H]
    sh2 = NamedSharding(m2, P("chip", "core"))
    exact = x.astype(np.float64).sum(0)
    for fn in (h8.xla_two_level, h8.xla_hier):
        f = jax.jit(jax.shard_map(lambda a, fn=fn: fn(a[0, 0], chip_axis="chip", core_axis="core")[None, None],
                                  mesh=m2, in_specs=P("chip", "core"), out_specs=P("chip", "core"), check_vma=False))
        out = np.asarray(f(jax.device_put(jnp.asarray(grid, jnp.bfloat16), sh2))).astype(np.float32)
        out = out.reshape(8, 64, H)
        np.testing.assert_allclose(out[0], exact, rtol=0, atol=0.25)
        assert all(np.array_equal(out[0], out[r]) for r in range(8))
    f = jax.jit(jax.shard_map(lambda a: h8.xla_rs_ag(a[0], axis="tp")[None], mesh=mesh, in_specs=P("tp"),
                              out_specs=P("tp"), check_vma=False))
    out = np.asarray(f(jax.device_put(jnp.asarray(x, jnp.bfloat16), NamedSharding(mesh, P("tp"))))).astype(np.float32)
    np.testing.assert_allclose(out[0], exact, rtol=0, atol=0.25)


def _protocol_violations(trace):
    """Check one traced kernel's schedule (``h8._SCHEDULE_TRACE`` events). Returns a list of violations."""
    bad = []
    members = sorted({e[0] for e in trace})
    if members != list(range(8)):
        return [f"members traced: {members}"]
    signalled = {m: 0 for m in members}
    sent, waited, s_started, s_waited = {}, {}, {}, {}
    for m in members:
        ev = [e[1:] for e in trace if e[0] == m]
        sig = [e[1] for e in ev if e[0] == "signal"]
        waits = [i for i, e in enumerate(ev) if e[0] == "barrier_wait"]
        starts = [i for i, e in enumerate(ev) if e[0] == "start"]
        if len(waits) != 1:
            bad.append(f"member {m}: {len(waits)} barrier waits")
            continue
        w = waits[0]
        if ev[w][1] != len(sig) or len(set(sig)) != len(sig) or m in sig:
            bad.append(f"member {m}: waits {ev[w][1]} for signals to {sig}")
        if any(e[0] == "signal" for e in ev[w:]):
            bad.append(f"member {m}: signals after its barrier wait")
        if not starts or starts[0] < w:
            bad.append(f"member {m}: a remote copy starts before the barrier wait")
        for p in sig:
            signalled[p] += 1
        for i in starts:
            _, peer, send, recv, nb = ev[i]
            if peer not in sig:
                bad.append(f"member {m}: writes to {peer}, which it did not pass a barrier with")
            sent[(peer, recv)] = sent.get((peer, recv), 0) + nb
            s_started[(m, send)] = s_started.get((m, send), 0) + nb
        for e in ev:
            if e[0] == "wait_recv":
                waited[(m, e[1])] = waited.get((m, e[1]), 0) + e[2]
            if e[0] == "wait_send":
                s_waited[(m, e[1])] = s_waited.get((m, e[1]), 0) + e[2]
    for m in members:
        w = [e[2] for e in trace if e[0] == m and e[1] == "barrier_wait"]
        if w and w[0] != signalled[m]:
            bad.append(f"member {m}: waits for {w[0]} barrier signals, receives {signalled[m]}")
    if sent != waited:
        diff = sorted(set(sent.items()) ^ set(waited.items()))[:4]
        bad.append(f"receive semaphores unbalanced (sent vs waited bytes): {diff}")
    if s_started != s_waited:
        bad.append("send semaphores unbalanced")
    return bad


PROTO = [("fwd", "bf16", h8.DEFAULT_PAIRS), ("direct", "bf16", h8.DEFAULT_PAIRS), ("fwd", "f32", h8.DEFAULT_PAIRS),
         ("fwd", "bf16", SCRAMBLED), ("direct", "f32", SCRAMBLED)]


@pytest.mark.parametrize("ag,wire,pairs", PROTO, ids=["fwd-bf16", "direct-bf16", "fwd-f32", "fwd-bf16-scr", "direct-f32-scr"])
@pytest.mark.parametrize("rows", [16, 112])
def test_schedule_protocol(mesh, monkeypatch, rows, ag, wire, pairs):
    """The barrier and the DMA semaphores, from the real kernel code traced (the interpreter starts all members
    together, so a missing barrier never shows in its outputs): every member passes one barrier with exactly the
    members it writes to, before its first remote copy; every barrier wait count equals the signals it receives;
    every receive semaphore is waited for exactly the bytes sent to it; every send semaphore balances."""
    trace = []
    monkeypatch.setattr(h8, "_SCHEDULE_TRACE", trace)
    jax.make_jaxpr(lambda a: h8.tp8_hier_all_reduce(a, mesh=mesh, axis="tp", pairs=pairs, ag=ag, wire=wire,
                                                    interpret=True))(jax.ShapeDtypeStruct((8, rows, H), jnp.bfloat16))
    assert trace, "nothing traced"
    bad = _protocol_violations(trace)
    assert not bad, bad
