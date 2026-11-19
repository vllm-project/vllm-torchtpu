# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 GDN-COMMIT1: the whole-step call's plumbing (CPU; kernel, QSA inputs and writers stubbed with the
D07-b1 test's stubs, tests/megakernel_runtime/test_l1d07b1_gdn_meta_shared.py).

  - the flag is registered, off by default;
  - flag off: `call` and D07-b1's `call_shared` return (residual, pool) with the scatter called once, specs (tp, tp);
  - flag on (writing call): the scatter is never called, the residual is bit-identical to the flag-off residual, the pool
    holds only the QSA writer's bytes, and the 4 extra outputs are the kernel's conv / rec checkpoints, the call's write
    ids (= gdn_blocks' wb) and the state ids (each layer's ids fitted to NS rows), tiled per rank, specs + 4 x tp;
  - flag on with write=False (the wholecheck shadow): the flag-off program (scatter called, no extra outputs);
  - refused together with VLLM_TPU_MK_L47_FOLD and VLLM_TPU_MK_GDN_CKPT_INK (ValueError at build);
  - the torch op's stash pops exactly the 4 staging outputs into the kind's pending list (flag on, writing kind) and is
    the identity otherwise.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src:. pytest tests/q38_l1g1/test_g1_wholestep_call.py
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from tests.megakernel_runtime import test_l1d07b1_gdn_meta_shared as d7  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_commit as gc  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import T_VERIFY, gdn_blocks  # noqa: E402

FLAG = gc.ENV
R = d7.R


def _build(mesh, write: bool):
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache

    layers = [QSALayerCache(compressed_slot=o % 2, main_slot=(o + 1) % 2) for o in range(d7.N_ATT)]
    return wsc.build_wholestep_call(mesh, SimpleNamespace(att_every=4), d7.B, None, layers, L=d7.L, budget=64,
                                    max_groups=1024, norm_eps=1e-6, rotary_dim=64, write=write, interpret=True)


def _operands():
    md, layers, gdn_ids = d7._metadata_objects()
    per_layer = [np.asarray(t.numpy()) for t in d7._per_layer_operands(md, layers, gdn_ids)]
    shared = [np.asarray(t.numpy()) for t in d7._module(layers, gdn_ids)._gdn_meta_shared(md)]
    rng = np.random.default_rng(11)
    hidden = rng.standard_normal((R * d7.B, d7.H)).astype(np.float32)
    w0 = rng.standard_normal((R * 4, d7.H)).astype(np.float32)
    pool = np.zeros((R * d7.POOL,), np.int32)
    tabs = [rng.integers(0, 50, (48 * n,)).astype(np.int32) for _ in range(d7.N_ATT) for n in (8, 32, 32)]
    tail = [*tabs, rng.integers(0, 300, (d7.B,)).astype(np.int32), np.arange(d7.B + 1, dtype=np.int32),
            rng.integers(0, 300, (d7.B,)).astype(np.int32), rng.standard_normal((16, 128)).astype(np.float32),
            *[rng.standard_normal((n,)).astype(np.float32) for _ in range(d7.N_ATT) for n in (256, 128)]]
    return per_layer, shared, hidden, w0, pool, tail


def _counting_scatter(monkeypatch):
    calls = {"n": 0}
    stub = wsc.scatter_gdn_ckpts_fast

    def scatter(*a, **k):
        calls["n"] += 1
        return stub(*a, **k)

    monkeypatch.setattr(wsc, "scatter_gdn_ckpts_fast", scatter)
    return calls


def _set(monkeypatch, name: str, on: bool):
    if on:
        monkeypatch.setenv(name, "1")
    else:
        monkeypatch.delenv(name, raising=False)


def test_flag_registered_and_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    from vllm_torchtpu import envs

    assert envs.VLLM_TPU_MK_GDN_COMMIT1 is False and gc.commit1_on() is False
    monkeypatch.setenv(FLAG, "1")
    assert envs.VLLM_TPU_MK_GDN_COMMIT1 is True and gc.commit1_on() is True


@pytest.mark.parametrize("shared", [False, True])
def test_call_flag_off_scatters_and_flag_on_stages(monkeypatch, shared):
    d7._stub(monkeypatch)
    calls = _counting_scatter(monkeypatch)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    _set(monkeypatch, "VLLM_TPU_MK_GDN_META_DEDUP", shared)
    per_layer, shared_ops, hidden, w0, pool, tail = _operands()
    meta = shared_ops if shared else per_layer
    tp, rep = P("tp"), P()

    _set(monkeypatch, FLAG, False)
    call0, specs0, outs0, don0 = _build(mesh, True)
    assert call0.__name__ == ("call_shared" if shared else "call") and outs0 == (tp, tp)
    calls["n"] = 0
    res0, pool0 = jax.jit(call0)(hidden, w0, pool, *meta, *tail)
    assert calls["n"] == 1, "flag off: one scatter per call"

    _set(monkeypatch, FLAG, True)
    call1, specs1, outs1, don1 = _build(mesh, True)
    assert outs1 == (tp, tp, tp, tp, tp, tp) and specs1 == specs0 and don1 == don0
    calls["n"] = 0
    res1, pool1, conv1, rec1, wb1, st1 = jax.jit(call1)(hidden, w0, pool, *meta, *tail)
    assert calls["n"] == 0, "flag on: the scatter must not run"
    np.testing.assert_array_equal(np.asarray(res1), np.asarray(res0))

    NS = d7.B // T_VERIFY
    rb, wb, hi = (np.asarray(x) for x in jax.jit(lambda *m: gdn_blocks(m, NS, True))(*per_layer))
    p0, p1 = np.asarray(pool0).reshape(R, d7.POOL), np.asarray(pool1).reshape(R, d7.POOL)
    n = rb.size + hi.size + wb.size
    assert (p0[:, :n] != 0).any(), "control: the flag-off scatter wrote the pool prefix"
    assert (p1[:, :-1] == 0).all(), "flag on: only the QSA writer touches the pool"
    np.testing.assert_array_equal(p1[:, -1], p0[:, -1])
    # the staging outputs: the kernel's conv / rec (the stub returns rb / hi per rank), wb and the state ids
    np.testing.assert_array_equal(np.asarray(conv1), np.broadcast_to(rb[None], (R,) + rb.shape))
    np.testing.assert_array_equal(np.asarray(rec1), np.broadcast_to(hi[None], (R,) + hi.shape))
    # the ids leave tiled per rank (tp-sharded): each rank's shard is the step's whole wb / st
    np.testing.assert_array_equal(np.asarray(wb1).reshape(R, *wb.shape), np.broadcast_to(wb[None], (R,) + wb.shape))
    assert np.asarray(wb1).dtype == np.int32 and (wb != 0).any()
    st_want = np.stack([per_layer[5 * ln][:NS] for ln in range(d7.N_GDN)]).astype(np.int32)
    assert np.asarray(st1).shape == (R * d7.N_GDN, NS) and not (st_want == st_want[:1]).all()
    np.testing.assert_array_equal(np.asarray(st1).reshape(R, d7.N_GDN, NS), np.broadcast_to(st_want[None], (R,) + st_want.shape))


def test_write_false_keeps_the_flag_off_program(monkeypatch):
    d7._stub(monkeypatch)
    calls = _counting_scatter(monkeypatch)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    monkeypatch.delenv("VLLM_TPU_MK_GDN_META_DEDUP", raising=False)
    per_layer, _, hidden, w0, pool, tail = _operands()
    outs = {}
    for on in (False, True):
        _set(monkeypatch, FLAG, on)
        call, specs, out_specs, don = _build(mesh, False)
        assert out_specs == (P("tp"), P("tp"))
        calls["n"] = 0
        outs[on] = jax.jit(call)(hidden, w0, pool, *per_layer, *tail)
        assert calls["n"] == 1 and len(outs[on]) == 2
        if on:
            assert str(jax.make_jaxpr(call)(hidden, w0, pool, *per_layer, *tail)) == jaxpr_off
        else:
            jaxpr_off = str(jax.make_jaxpr(call)(hidden, w0, pool, *per_layer, *tail))
    for a, b in zip(outs[False], outs[True]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("other", ["VLLM_TPU_MK_L47_FOLD", "VLLM_TPU_MK_GDN_CKPT_INK"])
def test_refused_compositions(monkeypatch, other):
    d7._stub(monkeypatch)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(other, "1")
    assert gc.refusal() is not None and other in gc.refusal()
    with pytest.raises(ValueError, match="not composed with " + other):
        gc.out_specs((P("tp"), P("tp")), True, P("tp"), P())
    monkeypatch.delenv(other)
    assert gc.refusal() is None


def test_stash_pops_the_staging_outputs(monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep as qw

    outs = [f"o{i}" for i in range(6)]
    kind = SimpleNamespace(kw={"write": True})
    monkeypatch.delenv(FLAG, raising=False)
    assert qw._g1_stash(kind, list(outs), 16) == outs and "_g1_pending" not in kind.__dict__
    monkeypatch.setenv(FLAG, "1")
    n0 = gc.Counters.stashed
    assert qw._g1_stash(kind, list(outs), 16) == outs[:2]
    (s,) = kind._g1_pending
    assert (s.conv, s.rec, s.wb, s.st, s.rows) == ("o2", "o3", "o4", "o5", 16) and gc.Counters.stashed == n0 + 1
    shadow = SimpleNamespace(kw={"write": False})
    assert qw._g1_stash(shadow, list(outs), 16) == outs and "_g1_pending" not in shadow.__dict__
    stats = [f"s{i}" for i in range(7)]  # union stats on: (res, pool, stats) + 4
    assert qw._g1_stash(kind, list(stats), 32) == stats[:3] and kind._g1_pending[-1].conv == "s3"
