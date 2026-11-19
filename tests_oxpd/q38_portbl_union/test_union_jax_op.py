# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL UNION (rule 12): the jax op that carries the union route and the HBM-typed schedule,
``qwen4_exp::pooled_qsa_request_attend_dcp`` (body ``pooled_qsa_request_attend_dcp_step``), builds through the REAL
torch_tpu ``pallas.jax_op`` (signature check, static-argnum inference, torch schema, torch.library registration),
with both PORT-BL UNION keys set. The port adds no jax_op of its own: the SparseCore metadata kernel and the
selection-mask RPA run inside this op's body (J3 checks no jax_op call site was added). Template:
tests/q38_l1g1/test_g1_jax_op_signature.py (L1-G1FIX). Run on a CDK CPU pod (torch_tpu present; skips elsewhere).
"""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("torch_tpu")

import torch  # noqa: E402
from torch_tpu._internal import pallas as tt_pallas  # noqa: E402
from torch_tpu._internal.pallas import pallas as tt_impl  # noqa: E402

_REAL_JAX_OP = tt_pallas.jax_op
ATTEND_ARGS = ["q", "gate", "lengths", "req", "starts", "main_table", "context", "pool", "gathered", "dcp_rank"]


def _build(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    calls = []

    def spy(name, fn=None, /, **kw):
        op = _REAL_JAX_OP(name, fn, **kw)
        calls.append((name, fn, kw, op))
        return op

    monkeypatch.setattr(tt_pallas, "jax_op", spy)
    qsa_op._request_dcp_ops.cache_clear()
    qsa_op._request_dcp_ops()
    return calls


def test_j1_j2_attend_op_builds_with_the_real_jax_op(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_RPA_UNION_O1S", "640")
    monkeypatch.setenv("VLLM_TPU_RPA_SCHEDULE_HBM_O1S", "1")
    calls = _build(monkeypatch)
    names = [n for n, *_ in calls]
    assert any(n.endswith("pooled_qsa_request_attend_dcp") for n in names), names
    for name, fn, kw, op in calls:
        sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
        tt_impl._verify_signature(sig, bound_names=tt_impl._get_bound_names(fn))
        ns, opname = name.split("::")
        schema = getattr(getattr(torch.ops, ns), opname).default._schema
        args = [a.name for a in schema.arguments]
        print(f"PBLU J {name}: real _verify_signature OK; torch schema {len(args)} args; donate {kw.get('donate_argnums')}")
        if opname == "pooled_qsa_request_attend_dcp":
            assert args[: len(ATTEND_ARGS)] == ATTEND_ARGS, args
            assert kw.get("donate_argnums") == (7,)


def test_j3_the_port_adds_no_jax_op_site():
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa, qsa_union_meta

    for mod in (qsa_union_meta, qsa_rpa):
        tree = ast.parse(inspect.getsource(mod))
        sites = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and ((isinstance(n.func, ast.Attribute) and n.func.attr == "jax_op")
                      or (isinstance(n.func, ast.Name) and n.func.id == "jax_op"))]
        print(f"PBLU J3 {mod.__name__}: jax_op call sites {sites}")
        assert not sites
