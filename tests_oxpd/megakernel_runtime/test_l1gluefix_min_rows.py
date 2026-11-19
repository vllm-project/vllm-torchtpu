# SPDX-License-Identifier: Apache-2.0
"""q38 L1-GLUEFIX: VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS, the per-bucket gate of D07-b1 (CPU).

T98i (j-4e05e0ae, glue arm = D07-b1 + D07-b2 mode 1) lost +0.42 ms at 8 rows (2 live requests in the b16 whole-step
bucket), +0.10 at 12 rows (3) and gained -0.17 at 16 rows (4); 32 / 48 / 64 rows gained. With the gate, a bucket below
the min rows runs the flag-off whole-step program, so b16 is the base program again while b32+ keep D07-b1:

  - the gate's truth table (flag off: never shared; flag on, min 0: every bucket, the D07-b1 behaviour; min 32: shared
    from b32), the env default 0 and its refusal of negatives;
  - below the min rows `build_wholestep_call` returns the flag-off `call` with the flag-off operand specs, and its
    lowering is the flag-off lowering text; at / above it, `call_shared` with the ungated specs and lowering; the shared
    lowering at b16 differs from the flag-off one (the comparison is not vacuous);
  - `expand_gdn_meta` turns the module's shared operand list into the flag-off list (same values in the flag-off
    order; the shared step tensors are the same objects), and the flag-off program on it equals `call_shared` on the
    shared list bit for bit;
  - `_WholeKind.callable_for` wraps exactly the buckets below the min rows: the program built there receives the
    expanded list, the others the shared list unchanged.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel_runtime/test_l1gluefix_min_rows.py
"""

from __future__ import annotations

import importlib.util
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
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import GDN_SHARED, T_VERIFY  # noqa: E402

# The D07-b1 suite's stubs and runner-shaped metadata (loaded from its file, not copied).
_spec = importlib.util.spec_from_file_location(
    "q38_l1d07b1_suite", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_l1d07b1_gdn_meta_shared.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

FLAG, MINR = "VLLM_TPU_MK_GDN_META_DEDUP", "VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS"
R, L, N_ATT, H, N_GDN = D.R, D.L, D.N_ATT, D.H, D.N_GDN


def _set(monkeypatch, flag, minr):
    for k, v in ((FLAG, flag), (MINR, minr)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_gate_truth_table_and_default(monkeypatch):
    from vllm_torchtpu import envs

    _set(monkeypatch, None, None)
    assert envs.VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS == 0
    for minr in (None, "32"):
        _set(monkeypatch, None, minr)
        assert [wsc.gdn_meta_shared(b) for b in (16, 32, 48, 64)] == [False] * 4
        assert wsc.gdn_meta_shared() is False
    _set(monkeypatch, "1", None)
    assert [wsc.gdn_meta_shared(b) for b in (8, 16, 32, 48, 64)] == [True] * 5  # the D07-b1 behaviour
    _set(monkeypatch, "1", "32")
    assert [wsc.gdn_meta_shared(b) for b in (8, 16, 32, 48, 64)] == [False, False, True, True, True]
    assert wsc.gdn_meta_shared() is True  # the module's operand list stays the shared one
    _set(monkeypatch, "1", "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        wsc.gdn_meta_shared(16)


def _build(mesh, b):
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache

    layers = [QSALayerCache(compressed_slot=o % 2, main_slot=(o + 1) % 2) for o in range(N_ATT)]
    return wsc.build_wholestep_call(mesh, SimpleNamespace(att_every=4), b, None, layers, L=L, budget=64,
                                    max_groups=1024, norm_eps=1e-6, rotary_dim=64, interpret=True)


def _operands(b, seed):
    """(hidden, w0, pool, per-layer meta, shared meta, tail) for bucket b, from runner-shaped metadata."""
    NS = b // T_VERIFY
    off, qsl, seq, sts, cks = D._meta(N_GDN, NS, 48, seed)
    per_layer = D._full(off, qsl, seq, sts, cks)
    shared = [off, qsl, seq, *sts, *cks]
    rng = np.random.default_rng(seed + 1)
    hidden = rng.standard_normal((R * b, H)).astype(np.float32)
    w0 = rng.standard_normal((R * 4, H)).astype(np.float32)
    pool = np.zeros((R * D.POOL,), np.int32)
    tabs = [rng.integers(0, 50, (48 * n,)).astype(np.int32) for _ in range(N_ATT) for n in (8, 32, 32)]
    positions = rng.integers(0, 300, (b,)).astype(np.int32)
    starts = np.arange(b + 1, dtype=np.int32)
    context = rng.integers(0, 300, (b,)).astype(np.int32)
    trig = rng.standard_normal((16, 128)).astype(np.float32)
    norms = [rng.standard_normal((n,)).astype(np.float32) for _ in range(N_ATT) for n in (256, 128)]
    return hidden, w0, pool, per_layer, shared, [*tabs, positions, starts, context, trig, *norms]


def _lowered(call, args):
    return jax.jit(call).lower(*args).as_text()


@pytest.mark.parametrize("minr,flag_off", [("32", True), ("17", True), ("16", False), ("8", False)])
def test_below_min_rows_builds_the_flag_off_program(monkeypatch, minr, flag_off):
    """Flag on, bucket 16: below the min rows (17, 32) the flag-off program (name, specs, lowering text); at / above it
    (16, 8) the ungated shared program. The boundary is inclusive (min 16 at b16 is shared)."""
    D._stub(monkeypatch)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    b = D.B
    hidden, w0, pool, per_layer, shared, tail = _operands(b, seed=3)
    _set(monkeypatch, None, None)
    off_call, off_specs, off_outs, off_don = _build(mesh, b)
    _set(monkeypatch, "1", None)
    sh_call, sh_specs, sh_outs, sh_don = _build(mesh, b)
    _set(monkeypatch, "1", minr)
    g_call, g_specs, g_outs, g_don = _build(mesh, b)
    assert off_call.__name__ == "call" and sh_call.__name__ == "call_shared"
    off_txt = _lowered(off_call, (hidden, w0, pool, *per_layer, *tail))
    sh_txt = _lowered(sh_call, (hidden, w0, pool, *shared, *tail))
    assert off_txt != sh_txt  # not vacuous: the two programs differ
    if flag_off:
        assert g_call.__name__ == "call", g_call.__name__
        assert (g_specs, g_outs, g_don) == (off_specs, off_outs, off_don)
        assert _lowered(g_call, (hidden, w0, pool, *per_layer, *tail)) == off_txt
    else:
        assert g_call.__name__ == "call_shared", g_call.__name__
        assert (g_specs, g_outs, g_don) == (sh_specs, sh_outs, sh_don)
        assert _lowered(g_call, (hidden, w0, pool, *shared, *tail)) == sh_txt


def test_expand_gdn_meta_gives_the_flag_off_operands(monkeypatch):
    """The module's shared list, expanded, is the flag-off module's list (values, order; the shared step tensors are
    the same objects), and the flag-off program on it equals `call_shared` on the shared list bit for bit."""
    import torch

    md, layers, gdn_ids = D._metadata_objects(seed=21)
    mod = D._module(layers, gdn_ids)
    shared_t = mod._gdn_meta_shared(md)
    per_t = D._per_layer_operands(md, layers, gdn_ids)
    w0, pool = torch.zeros(3), torch.ones(5)
    tail = [torch.arange(4), torch.arange(6)]
    args = [w0, pool, *shared_t, *tail]
    got = wsc.expand_gdn_meta(args, 2, len(gdn_ids))
    want = [w0, pool, *per_t, *tail]
    assert len(got) == len(want) == 2 + 5 * len(gdn_ids) + 2
    assert got[0] is w0 and got[1] is pool and got[-2] is tail[0] and got[-1] is tail[1]
    for j, (g, w) in enumerate(zip(got, want)):
        assert g.dtype == w.dtype and torch.equal(g, w), j
    for k in range(len(gdn_ids)):  # off / qsl / seq: the shared objects at each layer's slots
        assert got[2 + 5 * k + 1] is shared_t[0] and got[2 + 5 * k + 3] is shared_t[1] and got[2 + 5 * k + 4] is shared_t[2]
        assert got[2 + 5 * k] is shared_t[GDN_SHARED + k] and got[2 + 5 * k + 2] is shared_t[GDN_SHARED + len(gdn_ids) + k]
    # End to end on the stubbed call (b16): flag-off program on the expanded list == call_shared on the shared list.
    D._stub(monkeypatch)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    _set(monkeypatch, None, None)
    call0 = _build(mesh, D.B)[0]
    _set(monkeypatch, "1", None)
    call1 = _build(mesh, D.B)[0]
    hidden, w0n, pooln, _, _, tailn = _operands(D.B, seed=4)
    shared_n = [np.asarray(t.numpy()) for t in shared_t]
    expanded_n = wsc.expand_gdn_meta([w0n, pooln, *shared_n, *tailn], 2, len(gdn_ids))
    res0, pool0 = jax.jit(call0)(hidden, *expanded_n)
    res1, pool1 = jax.jit(call1)(hidden, w0n, pooln, *shared_n, *tailn)
    np.testing.assert_array_equal(np.asarray(res0), np.asarray(res1))
    np.testing.assert_array_equal(np.asarray(pool0), np.asarray(pool1))
    assert (np.asarray(pool0) != 0).any()


@pytest.mark.parametrize("minr,wrapped", [(None, ()), ("32", (16,)), ("64", (16, 32, 48))])
def test_callable_for_expands_exactly_below_min_rows(monkeypatch, minr, wrapped):
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep as ws

    _set(monkeypatch, "1", minr)
    seen = {}

    def fake_build(mesh, g, B, *a, **k):
        def full(*args):
            return args
        full.B = B
        return full, ("spec",), ("out",), (2,)

    bridge = types.ModuleType("vllm_torchtpu.megakernel_runtime.exported_bridge")
    bridge.make_typed_step_wrapper = lambda name, full, n_in, n_out: full
    bridge.sharded_jax_callable = lambda name, typed, **kw: (lambda hidden, *args: seen.setdefault(typed.B, list(args)))
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.megakernel_runtime.exported_bridge", bridge)
    monkeypatch.setattr(wsc, "build_wholestep_call", fake_build)
    monkeypatch.setattr(ws, "wholestep_tiles", lambda B, mg: None)
    kw = dict(budget=512, max_groups=8192, norm_eps=1e-6, rotary_dim=64, write=True)
    kind = ws._WholeKind(SimpleNamespace(size=R), None, None, [None] * N_ATT, tag=f"l1gluefix_{minr}", **kw)
    assert kind.gdn_shared is True and kind.n_gdn == 0
    kind.pool_index, kind.n_gdn = 1, N_GDN  # one weight operand, then the pool, as bind_pool sets them
    shared = ["w0", "pool", "off", "qsl", "seq", *[f"st{k}" for k in range(N_GDN)], *[f"ck{k}" for k in range(N_GDN)],
              "tab", "pos"]
    flag_off = ["w0", "pool", *[x for k in range(N_GDN) for x in (f"st{k}", "off", f"ck{k}", "qsl", "seq")], "tab", "pos"]
    for b in (16, 32, 48, 64):
        kind.callable_for(b)("hidden", *shared)
        assert seen[b] == (flag_off if b in wrapped else shared), (b, minr)


def test_callable_for_refuses_an_unset_layer_count(monkeypatch):
    """Below the min rows the expansion needs the bind-time layer count; a kind that was never bound refuses."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep as ws

    _set(monkeypatch, "1", "32")
    bridge = types.ModuleType("vllm_torchtpu.megakernel_runtime.exported_bridge")
    bridge.make_typed_step_wrapper = lambda name, full, n_in, n_out: full
    bridge.sharded_jax_callable = lambda name, typed, **kw: typed
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.megakernel_runtime.exported_bridge", bridge)
    monkeypatch.setattr(wsc, "build_wholestep_call", lambda *a, **k: ((lambda *x: x), (), (), ()))
    monkeypatch.setattr(ws, "wholestep_tiles", lambda B, mg: None)
    kw = dict(budget=512, max_groups=8192, norm_eps=1e-6, rotary_dim=64, write=True)
    kind = ws._WholeKind(SimpleNamespace(size=R), None, None, [None] * N_ATT, tag="l1gluefix_unbound", **kw)
    with pytest.raises(AssertionError):
        kind.callable_for(16)
    kind.callable_for(32)  # at / above the min rows nothing is expanded
