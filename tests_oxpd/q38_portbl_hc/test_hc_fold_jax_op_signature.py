# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL HC, rule 12: the jax ops the port adds are built by the REAL torch_tpu jax_op (no fake).

Template: tests/q38_l1g1/test_g1_jax_op_signature.py (q38 L1-G1FIX). The port adds one new jax_op function, the final
mixer's no-injection decode combine-mix (vllm_adapter.build_decode_combine_mix_op(..., inject=False); ported from
Benliang Wang <benliangw@google.com>, commit 28419c4a5 on benliangw/q38fh-integ-pin-5b44702aea), and the ``_t<N>``
variants of the existing decode ops under TPU_QWEN4_HC_DECODE_TILE_ROWS. torch_tpu 0.2.0's pallas.jax_op runs
``inspect.signature(fn, eval_str=True)`` and ``_verify_signature`` when the op is built (the check that killed G1 at
engine init, T94c: "Missing argument type annotation for JAX function"); none of it needs a TPU.
  J1 each op builds through the real jax_op (a spy records the call, then calls the real one).
  J2 its torch schema: Tensor arguments only (named as the core's parameters), Tensor returns as its register_fake.
  J3 the recorded function's signature passes the real ``_verify_signature`` directly; no static argument.
  J4 the default (unset-key) ops keep v8's names; the module stays on tpu_compiler's runtime compile-key path.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch_tpu")

import torch  # noqa: E402
from torch_tpu._internal import pallas as tt_pallas  # noqa: E402
from torch_tpu._internal.pallas import pallas as tt_impl  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va  # noqa: E402

_REAL_JAX_OP = tt_pallas.jax_op
EPS = 1.2345e-6   # an eps no other test uses: unique op names in this process
COMBINE = ["residual", "block", "prev_injection", "weight", "down", "up"]
MIX = ["x", "weight", "down", "up"]


@pytest.fixture(autouse=True)
def _fresh_caches():
    va.build_decode_mix_op.cache_clear()
    va.build_decode_combine_mix_op.cache_clear()
    yield
    va.build_decode_mix_op.cache_clear()
    va.build_decode_combine_mix_op.cache_clear()


def _build(monkeypatch, builder, *args, **kwargs):
    calls = []

    def spy(name, fn=None, /, **kw):
        op = _REAL_JAX_OP(name, fn, **kw)
        calls.append((name, fn, kw, op))
        return op

    monkeypatch.setattr(tt_pallas, "jax_op", spy)
    builder(*args, **kwargs)
    monkeypatch.setattr(tt_pallas, "jax_op", _REAL_JAX_OP)
    assert len(calls) == 1, calls
    return calls[0]


def _schema(name: str):
    ns, op = name.split("::")
    return getattr(getattr(torch.ops, ns), op).default._schema


def _check(call, args_want, nret, suffix):
    name, fn, kw, op = call
    assert name.endswith(suffix), name
    s = _schema(name)
    args = [a.name for a in s.arguments]
    types = {str(a.type) for a in s.arguments}
    rets = [str(r.type) for r in s.returns]
    print(f"PBL_HC J2 {name}: {len(args)} args {sorted(types)} -> ({', '.join(rets)})")
    assert args == args_want, (name, args)
    assert types == {"Tensor"}, (name, s)
    assert rets == ["Tensor"] * nret, (name, s)
    sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
    tt_impl._verify_signature(sig, bound_names=tt_impl._get_bound_names(fn))   # J3: raises on the T94c defect
    assert tt_impl._infer_static_argnums(sig) == (), (name, sig)
    print(f"PBL_HC J3 {name}: real _verify_signature OK {sig}")


@pytest.mark.parametrize("up_t", [True, False])
def test_j_noinj_combine_mix_op_real_jax_op(monkeypatch, up_t):
    call = _build(monkeypatch, va.build_decode_combine_mix_op, 4, 2560, EPS, up_t, inject=False)
    _check(call, COMBINE, 2, f"_noinj{'_upt' if up_t else ''}_v1")


def test_j_tile_variants_real_jax_op(monkeypatch):
    _check(_build(monkeypatch, va.build_decode_combine_mix_op, 4, 2560, EPS, True, inject=False, max_tile=512),
           COMBINE, 2, "_noinj_upt_t512_v1")
    _check(_build(monkeypatch, va.build_decode_combine_mix_op, 4, 2560, EPS, True, max_tile=512),
           COMBINE + ["inject_weight"], 3, "_upt_t512_v1")
    _check(_build(monkeypatch, va.build_decode_mix_op, 4, 2560, EPS, True, True, max_tile=512),
           MIX + ["inject_weight"], 2, "_inj_upt_t512_v1")
    _check(_build(monkeypatch, va.build_decode_mix_op, 4, 2560, EPS, False, True, max_tile=512),
           MIX, 1, "_upt_t512_v1")


def test_j4_default_ops_keep_v8_names_and_compile_key_path(monkeypatch):
    name = _build(monkeypatch, va.build_decode_combine_mix_op, 4, 2560, EPS, True)[0]
    assert "_noinj" not in name and name.endswith("_upt_v1") and "_t5" not in name, name
    name = _build(monkeypatch, va.build_decode_mix_op, 4, 2560, EPS, True, True)[0]
    assert name.endswith("_inj_upt_v1"), name
    from vllm_torchtpu.compilation.tpu_compiler import _RUNTIME_CACHE_KEY_PATHS

    rel = "vllm_torchtpu/kernels/qwen4_exp/vllm_adapter.py"
    assert any(rel == k or rel.startswith(k.rstrip("/") + "/") for k in _RUNTIME_CACHE_KEY_PATHS)
    print("PBL_HC J4 default names unchanged; vllm_adapter.py on _RUNTIME_CACHE_KEY_PATHS")
