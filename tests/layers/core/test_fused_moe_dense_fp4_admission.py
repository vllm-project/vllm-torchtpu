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

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.layers.core import fused_moe_gmm as g

_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
FP4 = jnp.float4_e2m1fn
H, I, E, TOPK = 256, 256, 8, 4


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
        jax.ShapeDtypeStruct((E, H, 2 * I), FP4),
        jax.ShapeDtypeStruct((E, I, H), FP4),
        jax.ShapeDtypeStruct((E, H // 128, 1, 2 * I), jnp.float32),
        jax.ShapeDtypeStruct((E, I // 128, 1, H), jnp.float32),
        None,
        None,
        jax.ShapeDtypeStruct((m, TOPK), jnp.bfloat16),
        jax.ShapeDtypeStruct((m, TOPK), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    try:
        jax.eval_shape(
            lambda *a: g.fused_moe_func(
                *a, topk=TOPK, activation="swigluoai", use_ep=use_ep
            ),
            *args,
        )
    except _Gmm:
        pass
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
