# SPDX-License-Identifier: Apache-2.0
"""q38 H8-FIX: the function H8 hands to torch_tpu's ``jax_op`` must survive jax_op's signature evaluation (CPU).

Why (labels M / C / I as in reports/runs.md):
- On TPU (M, J0b allonk arm, job j-db896a2a, 2026-11-13 09:03Z) every ``VLLM_TPU_DECODE_TP_AR=h8`` arm on 5d368f166
  failed at engine init: ``tp_ar_decode.arm`` -> torch_tpu ``jax_op`` -> ``inspect.signature(..., eval_str=True)`` ->
  ``NameError: name 'jax' is not defined``.
- Cause (C): ``tp8_ar_decode_fn.py`` has ``from __future__ import annotations`` (annotations are stored as strings) and
  ``global_fn`` imported jax only locally, so its inner ``fn(x: jax.Array) -> jax.Array`` carried the strings
  ``'jax.Array'``, which ``eval_str=True`` evaluates in the module's globals, where ``jax`` was not bound.
- torch_tpu's ``jax_op`` (torch_tpu/_internal/pallas/pallas.py, ``dec``) runs exactly
  ``inspect.signature(inspect.unwrap(fn), eval_str=True)`` and then maps the return annotation to a schema with
  ``_map_jax_return_to_schema_str``, which accepts ``jax.Array`` / ``torch.Tensor`` and raises for an empty annotation
  (C, read from the torch_tpu source), so the annotations must stay and must evaluate to ``jax.Array``.
- The older wiring test (test_tp_ar_decode_wiring.py) replaced ``jax_op`` with a stub that never looked at the
  signature, so it passed with this bug present.

These tests call the same evaluation on the function ``global_fn`` returns, and drive ``tp_ar_decode.arm`` with a
``jax_op`` stand-in that performs torch_tpu's evaluation (the stand-in is installed as ``torch_tpu._internal.pallas``
when the real package is absent, as on CPU), so the arm path is exercised end to end up to the op build.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import typing
from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")

from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as fnmod  # noqa: E402

PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))


def _jax_op_signature(fn):
    """torch_tpu jax_op's own call (pallas.py ``dec``): unwrap, then evaluate string annotations."""
    return inspect.signature(inspect.unwrap(fn), eval_str=True)


@pytest.mark.parametrize("wire", ["bf16", "f32"])
def test_global_fn_signature_evaluates_like_jax_op(wire):
    fn = fnmod.global_fn(object(), axis="ep", pairs=PAIRS, wire=wire, max_rows_=256)
    sig = _jax_op_signature(fn)  # raised NameError: name 'jax' is not defined on 5d368f166
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["x"]
    assert params[0].annotation is jax.Array
    assert sig.return_annotation is jax.Array  # what jax_op maps to the schema return "Tensor"


def test_global_fn_type_hints_resolve():
    fn = fnmod.global_fn(object(), axis="ep", pairs=PAIRS, wire="bf16", max_rows_=256)
    assert typing.get_type_hints(fn) == {"x": jax.Array, "return": jax.Array}


def test_jax_is_a_module_global_of_the_op_module():
    # eval_str evaluates in fn.__globals__ (the module dict); the inner fn must find ``jax`` there.
    fn = fnmod.global_fn(object(), axis="ep", pairs=PAIRS, wire="bf16", max_rows_=256)
    assert fn.__globals__ is vars(fnmod)
    assert vars(fnmod).get("jax") is jax


# ---------------------------------------------------------------- arm() through a jax_op that evaluates the signature


def _install_torch_tpu_pallas_standin(monkeypatch):
    """Return the module object ``arm`` imports jax_op from: the real torch_tpu one when importable, else a stand-in
    module registered in sys.modules (CPU)."""
    if importlib.util.find_spec("torch_tpu") is not None:
        import torch_tpu._internal.pallas as ttp

        return ttp
    pkg = types.ModuleType("torch_tpu")
    pkg.__path__ = []
    internal = types.ModuleType("torch_tpu._internal")
    internal.__path__ = []
    pallas = types.ModuleType("torch_tpu._internal.pallas")
    pallas.jax_op = None  # replaced below
    pkg._internal = internal
    internal.pallas = pallas
    monkeypatch.setitem(sys.modules, "torch_tpu", pkg)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", internal)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", pallas)
    return pallas


def test_arm_hands_jax_op_an_evaluable_signature(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm.distributed")
    import vllm.distributed as vd

    from vllm_torchtpu.distributed import ep_mesh
    from vllm_torchtpu.layers.adapter import tp_ar_decode as ar

    monkeypatch.setenv(fnmod.ENV, "h8")
    monkeypatch.delenv(fnmod.MAX_ROWS_ENV, raising=False)
    monkeypatch.setattr(ar, "_ARMED", None)
    ttp = _install_torch_tpu_pallas_standin(monkeypatch)

    tp = SimpleNamespace(world_size=8, ranks=list(range(8)), cpu_group=None)
    ep = SimpleNamespace(world_size=8, ranks=list(range(8)))
    monkeypatch.setattr(vd, "get_tp_group", lambda: tp)
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: ep)
    order = (0, 1, 4, 5, 6, 7, 2, 3)

    class _Dev:
        def __init__(self, i):
            self.id = i

    class _Mesh:
        shape = {ep_mesh.EP_AXIS_NAME: 8}
        axis_names = (ep_mesh.EP_AXIS_NAME,)

        class devices:  # noqa: N801
            flat = [_Dev(i) for i in order]

    monkeypatch.setattr(ep_mesh, "build_ep_mesh", lambda: _Mesh())
    monkeypatch.setattr(ep_mesh, "ep_device_ids", lambda: order)
    monkeypatch.setattr(ep_mesh, "_ep_mesh_order", lambda ids: tuple(range(len(ids))))

    def gather(out, sig, group=None):
        for i in range(len(out)):
            out[i] = sig

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    seen = []

    class _JaxOp:
        """torch_tpu jax_op's first steps on ``fn``: the signature evaluation that raised on TPU."""

        def __init__(self, name, fn, **kw):
            sig = _jax_op_signature(fn)
            seen.append((name, sig, kw))

        def register_fake(self, f):
            self.fake = f

        def __call__(self, x):
            return ("kernel", x)

    monkeypatch.setattr(ttp, "jax_op", _JaxOp)
    runner = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=2560)))
    assert ar.arm(runner) is True
    assert len(seen) == 1
    name, sig, kw = seen[0]
    assert name.startswith("pallas::q38_tp_ar_decode")
    assert sig.return_annotation is jax.Array
    assert [p.annotation for p in sig.parameters.values()] == [jax.Array]
    assert ar._ARMED is not None and ar._ARMED[1] == 2560
