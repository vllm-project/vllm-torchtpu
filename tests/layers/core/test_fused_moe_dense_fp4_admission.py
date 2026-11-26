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
"""Which token counts fused_moe_func hands to the dense fp4 kernel (all rows
or sparse rows) and which keep sort + GMM + combine, traced abstractly."""

import contextlib

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.layers.core import fused_moe_gmm as g

_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
FP4 = jnp.float4_e2m1fn
H, INTER, E, TOPK = 256, 256, 8, 4


class _Gmm(Exception):
    pass


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _INFO)
    calls = []

    def dense(*args, sparse_rows=False, **kwargs):
        calls.append(("dense", args[0].shape[0], sparse_rows))
        return jnp.zeros(args[0].shape, jnp.bfloat16)

    def gmm(*args, **kwargs):
        calls.append(("gmm",))
        raise _Gmm

    monkeypatch.setattr(g, "dense_expert_moe_fp4", dense)
    monkeypatch.setattr(g, "prepare_routed_gmm_inputs", gmm)
    g.fused_moe_func.clear_cache()
    yield calls
    g.fused_moe_func.clear_cache()


def _trace(m, use_ep):
    args = (
        jax.ShapeDtypeStruct((m, H), jnp.bfloat16),
        jax.ShapeDtypeStruct((E, H, 2 * INTER), FP4),
        jax.ShapeDtypeStruct((E, INTER, H), FP4),
        jax.ShapeDtypeStruct((E, H // 128, 1, 2 * INTER), jnp.float32),
        jax.ShapeDtypeStruct((E, INTER // 128, 1, H), jnp.float32),
        None,
        None,
        jax.ShapeDtypeStruct((m, TOPK), jnp.bfloat16),
        jax.ShapeDtypeStruct((m, TOPK), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    with contextlib.suppress(_Gmm):
        jax.eval_shape(
            lambda *a: g.fused_moe_func(
                *a, topk=TOPK, activation="swigluoai", use_ep=use_ep
            ),
            *args,
        )
    g.fused_moe_func.clear_cache()


def _route(spy, m, use_ep):
    spy.clear()
    _trace(m, use_ep)
    return spy[-1]


def test_defaults_keep_main_admission(spy, monkeypatch):
    monkeypatch.setattr(g, "_DENSE_FP4_MAX_TOKENS", 16)
    monkeypatch.setattr(g, "_DENSE_FP4_SPARSE_MAX_TOKENS", 0)
    monkeypatch.setattr(g, "_DENSE_FP4_TP_MOE", False)
    assert _route(spy, 16, True) == ("dense", 16, False)
    assert _route(spy, 17, True) == ("gmm",)
    # TP-MoE ranks keep the GMM path with the r15-merge levers unset.
    assert _route(spy, 8, False) == ("gmm",)


@pytest.mark.parametrize("use_ep", [True, False])
def test_sparse_rows_between_the_caps(spy, monkeypatch, use_ep):
    monkeypatch.setattr(g, "_DENSE_FP4_MAX_TOKENS", 16)
    monkeypatch.setattr(g, "_DENSE_FP4_SPARSE_MAX_TOKENS", 128)
    monkeypatch.setattr(g, "_DENSE_FP4_TP_MOE", True)
    for m in (1, 8, 16):
        assert _route(spy, m, use_ep) == ("dense", m, False)
    for m in (17, 24, 32, 48, 64, 96, 128):
        assert _route(spy, m, use_ep) == ("dense", m, True)
    for m in (129, 256, 512, 8192):
        assert _route(spy, m, use_ep) == ("gmm",)


def test_sparse_rows_alone(spy, monkeypatch):
    """TPU_MOE_DENSE_FP4_MAX_TOKENS=0: only the sparse range takes the kernel."""
    monkeypatch.setattr(g, "_DENSE_FP4_MAX_TOKENS", 0)
    monkeypatch.setattr(g, "_DENSE_FP4_SPARSE_MAX_TOKENS", 64)
    monkeypatch.setattr(g, "_DENSE_FP4_TP_MOE", True)
    assert _route(spy, 8, True) == ("dense", 8, True)
    assert _route(spy, 64, True) == ("dense", 64, True)
    assert _route(spy, 65, True) == ("gmm",)


# ------------------------------------------------------------ shared expert
# (TPU_MOE_DENSE_FP4_SHARED): exactly one shared-expert add on every path.

import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.experimental.moe_dense_expert_fp4 import (  # noqa: E402
    SharedExpert,
)

I_S = 128


def _shared_expert(rng):
    e4m3 = jnp.float8_e4m3fn
    return SharedExpert(
        jnp.asarray(rng.normal(size=(H, 2 * I_S)), jnp.float32).astype(e4m3),
        jnp.full((1, 2 * I_S), 2.0**-5, jnp.float32),
        jnp.asarray(rng.normal(size=(I_S, H)), jnp.float32).astype(e4m3),
        jnp.full((1, H), 2.0**-6, jnp.float32),
    )


@pytest.fixture
def stub(monkeypatch):
    """The dense kernel and the GMM path both return ``routed``; records
    what each was handed."""
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _INFO)
    seen = {}

    def dense(*args, sparse_rows=False, shared=None, **kwargs):
        seen["path"] = "dense"
        seen["kernel_shared"] = shared is not None
        return seen["routed"] + (0 if shared is None else seen["shared_out"])

    def prepare(hidden, *args, **kwargs):
        t = hidden.shape[0]
        z = jnp.zeros((t * TOPK,), jnp.int32)
        return hidden, jnp.zeros((E,), jnp.int32), z, z, z, z

    def gmm(x, *args, shared=None, num_tokens, **kwargs):
        seen["path"] = "gmm"
        seen["combine_shared"] = shared is not None
        out = seen["routed"]
        if shared is not None:
            out = g.tc_combine_mod._shared_epilogue(out, shared)
        return out

    monkeypatch.setattr(g, "dense_expert_moe_fp4", dense)
    monkeypatch.setattr(g, "prepare_routed_gmm_inputs", prepare)
    monkeypatch.setattr(g, "moe_gmm", gmm)
    monkeypatch.setattr(g, "_DENSE_FP4_MAX_TOKENS", 16)
    monkeypatch.setattr(g, "_DENSE_FP4_SPARSE_MAX_TOKENS", 128)
    monkeypatch.setattr(g, "_DENSE_FP4_TP_MOE", True)
    g.fused_moe_func.clear_cache()
    yield seen
    g.fused_moe_func.clear_cache()


def _call(m, sh, rows=None, use_ep=False):
    rng = np.random.default_rng(m)
    x = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
    args = (
        x,
        jnp.zeros((E, H, 2 * INTER), FP4),
        jnp.zeros((E, INTER, H), FP4),
        jnp.ones((E, H // 128, 1, 2 * INTER), jnp.float32),
        jnp.ones((E, INTER // 128, 1, H), jnp.float32),
        None,
        None,
        jnp.ones((m, TOPK), jnp.bfloat16),
        jnp.zeros((m, TOPK), jnp.int32),
        jnp.zeros((), jnp.int32),
    )
    out = g.fused_moe_func(
        *args,
        topk=TOPK,
        activation="swigluoai",
        use_ep=use_ep,
        shared_expert=sh,
        shared_rows=rows,
    )
    g.fused_moe_func.clear_cache()
    return x, np.asarray(out, np.float32)


@pytest.mark.parametrize("m", [8, 16, 17, 64, 128, 129, 512])
def test_shared_added_once(stub, m):
    rng = np.random.default_rng(0)
    sh = _shared_expert(rng)
    stub["routed"] = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
    x = jnp.asarray(np.random.default_rng(m).normal(size=(m, H)), jnp.bfloat16)
    want = np.asarray(g.shared_expert_mlp(x, sh, stub["routed"]), np.float32)
    stub["shared_out"] = g.shared_expert_mlp(x, sh, jnp.zeros_like(x))
    _, got = _call(m, sh)
    assert stub["path"] == ("dense" if m <= 128 else "gmm")
    if m <= 128:
        # The kernel's result as returned: the op adds nothing after it.
        assert stub["kernel_shared"]
        kernel_out = stub["routed"] + stub["shared_out"]
        np.testing.assert_array_equal(got, np.asarray(kernel_out, np.float32))
    else:
        assert not stub["combine_shared"]
        np.testing.assert_array_equal(got, want)


def test_shared_e4m3_rows_feed_gate_up(stub):
    m = 256
    rng = np.random.default_rng(1)
    sh = _shared_expert(rng)
    stub["routed"] = jnp.zeros((m, H), jnp.bfloat16)
    x = jnp.asarray(np.random.default_rng(m).normal(size=(m, H)), jnp.bfloat16)
    scale = (
        jnp.maximum(jnp.max(jnp.abs(x.astype(jnp.float32)), 1, keepdims=True), 1e-6)
        / 448
    )
    q = (x.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn)
    rows = ((q[:100], scale[:100]), (q[100:], scale[100:]))
    _, got = _call(m, sh, rows=rows)
    act = jnp.concatenate(
        [g.shared_activation_e4m3(a, b, sh.w1, sh.s1) for a, b in rows]
    )
    want = np.asarray(g._shared_projection(act, sh.w2, sh.s2), np.float32)
    np.testing.assert_array_equal(got, want)


def test_shared_in_combine_only_on_its_steps(stub, monkeypatch):
    monkeypatch.setattr(g, "_tc_shared_applies", lambda *a, **k: True)
    monkeypatch.setattr(g.envs, "TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS", 512)
    rng = np.random.default_rng(2)
    sh = _shared_expert(rng)
    for on, m, fused in ((False, 512, False), (True, 256, False), (True, 512, True)):
        monkeypatch.setattr(g.envs, "TPU_MINIMAX_M3_SHARED_IN_COMBINE", on)
        stub["routed"] = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
        x = jnp.asarray(np.random.default_rng(m).normal(size=(m, H)), jnp.bfloat16)
        _, got = _call(m, sh, use_ep=True)
        assert stub["combine_shared"] == fused
        want = np.asarray(g.shared_expert_mlp(x, sh, stub["routed"]), np.float32)
        # Power-of-two scales: the combine's bf16 scale is exact, same value.
        np.testing.assert_array_equal(got, want)
