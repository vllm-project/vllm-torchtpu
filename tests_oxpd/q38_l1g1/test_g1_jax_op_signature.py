# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1FIX: G1 GDN-COMMIT1's (and the G1 CHECK's) jax ops pass the REAL torch_tpu jax_op construction (no fake).

Why: the G1 device check T94c (j-7ca9fba7, 2026-11-15 19:37:37Z) died at engine init in qwen38_gdn_commit._ops: torch_tpu
0.2.0's pallas.jax_op runs `inspect.signature(fn, eval_str=True)` and `_verify_signature` when the op is built and raised
"ValueError: Missing argument type annotation for JAX function: (pool, offsets, conv, rec, wb, st)." The CPU proofs used a
fake torch_tpu whose jax_op checks nothing. Here the image's own torch_tpu._internal.pallas.jax_op builds every op: the
signature check, the static-argnum inference, the torch schema and the torch.library registration all run, and none of them
needs a TPU (a CDK CPU pod has torch_tpu; elsewhere the module skips, and the q38 proof harness fails on any skip).
  J1 `_ops(layout)` builds commit1 / commit2 / full with the real jax_op (a spy records each call, then calls the real one).
  J2 each op's torch schema: Tensor arguments only (no static argument), returns (Tensor, Tensor) like its register_fake.
  J3 each recorded function's signature passes the real `_verify_signature` directly (the check that raised in T94c).
  J4 [trees with the CHECK] `_check_ops(layout)` builds the check op the same way; schema returns (Tensor, Tensor, Tensor).
  J5 coverage: the recorded jax_op calls == the `jax_op(` call sites in qwen38_gdn_commit.py (a new site must be built here).
Run on a CDK CPU pod: JAX_PLATFORMS=cpu PYTHONPATH=src:. pytest -s tests/q38_l1g1/test_g1_jax_op_signature.py
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

pytest.importorskip("torch_tpu")

import torch  # noqa: E402
from torch_tpu._internal import pallas as tt_pallas  # noqa: E402
from torch_tpu._internal.pallas import pallas as tt_impl  # noqa: E402

import vllm_torchtpu.megakernel_runtime.qwen38_gdn_commit as gc  # noqa: E402

MISSING = "Missing argument type annotation for JAX function"
_REAL_JAX_OP = tt_pallas.jax_op  # captured once: a spy must never wrap another spy
COMMIT_ARGS = {
    "q38_g1_gdn_commit1": ["pool", "offsets", "conv", "rec", "wb", "st"],
    "q38_g1_gdn_commit2": ["pool", "offsets", "conv_a", "rec_a", "wb_a", "st_a", "conv_b", "rec_b", "wb_b", "st_b"],
    "q38_g1_gdn_full": ["pool", "conv", "rec", "wb"],
}
CHECK_ARGS = {"q38_g1_gdn_check": ["pool", "offsets", "conv", "rec", "wb", "st", "fault"]}


class _Layout:
    """A hashable stand-in: building the ops only closes over the layout (it is read when the op is traced)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


def _build(monkeypatch, builder, tag: str) -> list:
    """Calls `builder(layout)` with the real jax_op behind a recording spy; returns [(name, fn, kwargs, op)]."""
    calls = []

    def spy(name, fn=None, /, **kw):
        op = _REAL_JAX_OP(name, fn, **kw)
        calls.append((name, fn, kw, op))
        return op

    monkeypatch.setattr(tt_pallas, "jax_op", spy)
    builder(_Layout(tag))
    return calls


def _schema(name: str):
    ns, op = name.split("::")
    return getattr(getattr(torch.ops, ns), op).default._schema


def _base(name: str) -> str:
    return re.sub(r"_[0-9]+$", "", name.split("::")[1])


def _check_ops_built(calls: list, want: dict, nret: int) -> None:
    assert sorted(_base(n) for n, *_ in calls) == sorted(want), [n for n, *_ in calls]
    for name, fn, kw, op in calls:
        s = _schema(name)
        args = [a.name for a in s.arguments]
        types = {str(a.type) for a in s.arguments}
        rets = [str(r.type) for r in s.returns]
        print(f"Q38G1FIX J2 {name}: {len(args)} args {sorted(types)} -> ({', '.join(rets)}); donate {kw.get('donate_argnums')}")
        assert args == want[_base(name)], (name, args)
        assert types == {"Tensor"}, (name, s)
        assert rets == ["Tensor"] * nret, (name, s)
        sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
        tt_impl._verify_signature(sig, bound_names=tt_impl._get_bound_names(fn))  # J3: raises on the T94c defect
        assert tt_impl._infer_static_argnums(sig) == (), (name, sig)
        print(f"Q38G1FIX J3 {name}: real _verify_signature OK {sig}")


def test_j1_j2_j3_commit_ops_real_jax_op(monkeypatch):
    k0 = gc._ops.cache_info().currsize
    calls = _build(monkeypatch, gc._ops, "commit")
    assert gc._ops.cache_info().currsize == k0 + 1
    assert [kw.get("donate_argnums") for *_, kw, _ in calls] == [(0,), (0,), (0,)]
    _check_ops_built(calls, COMMIT_ARGS, 2)


def test_j4_check_op_real_jax_op(monkeypatch):
    if not hasattr(gc, "_check_ops"):
        n = len(_jax_op_sites())
        assert n == 3, f"no _check_ops but {n} jax_op call sites"
        print("Q38G1FIX J4 no CHECK in this tree (3 jax_op sites, all in _ops)")
        return
    calls = _build(monkeypatch, gc._check_ops, "check")
    assert [kw.get("donate_argnums") for *_, kw, _ in calls] == [None]
    _check_ops_built(calls, CHECK_ARGS, 3)


def _jax_op_sites() -> list:
    tree = ast.parse(inspect.getsource(gc))
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Attribute) and n.func.attr == "jax_op")
                 or (isinstance(n.func, ast.Name) and n.func.id == "jax_op"))]


def test_j5_every_jax_op_site_is_built(monkeypatch):
    sites = _jax_op_sites()
    built = _build(monkeypatch, gc._ops, "cover")
    if hasattr(gc, "_check_ops"):
        built += _build(monkeypatch, gc._check_ops, "cover")
    print(f"Q38G1FIX J5 jax_op call sites {len(sites)} (lines {sites}); built {len(built)}")
    assert len(sites) == len(built) and len(sites) in (3, 4), (sites, [n for n, *_ in built])


def test_j0_the_real_check_rejects_an_unannotated_twin():
    """Control: the real check is live in this process (an unannotated copy of commit1's signature raises the T94c error)."""

    def commit1_fn(pool, offsets, conv, rec, wb, st):
        return pool, wb

    with pytest.raises(ValueError, match=re.escape(f"{MISSING}: (pool, offsets, conv, rec, wb, st).")):
        tt_pallas.jax_op("pallas::q38_g1fix_unannotated_twin", commit1_fn, donate_argnums=(0,))
    print("Q38G1FIX J0 real jax_op rejects the unannotated twin with the T94c message")
