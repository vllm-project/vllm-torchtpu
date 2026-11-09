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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.mhc import shifted_post_pre_rms as mhc

M, D = 4, 5120


def _inputs(tokens: int, seed: int = 0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 8)
    x = (jax.random.normal(ks[0], (tokens, D)) * 0.5).astype(jnp.bfloat16)
    flat = (jax.random.normal(ks[1], (tokens, M * D)) * 0.5).astype(jnp.bfloat16)
    post = jax.nn.sigmoid(jax.random.normal(ks[2], (tokens, M, 1))) * 2.0
    comb = jax.nn.softmax(jax.random.normal(ks[3], (tokens, M, M)), axis=-1)
    pre = jax.nn.sigmoid(jax.random.normal(ks[4], (tokens, M))) + 1e-6
    weight = mhc.pack_mix_weight(jax.random.normal(ks[5], (M * (M + 2), M * D)) * 0.01)
    scale = jnp.array([0.1, 0.2, 0.3], jnp.float32)
    base = jax.random.normal(ks[6], (M * (M + 2),)) * 0.1
    norm = (1.0 + 0.1 * jax.random.normal(ks[7], (D,))).astype(jnp.bfloat16)
    return x, flat, post, comb, pre, weight, scale, base, norm


def _layers(layout: dict, x, flat, post, comb, pre, weight, scale, base, norm):
    """Three chained layers, as the decoder passes residual and gates on."""
    outs = []
    for _ in range(3):
        flat, post, comb, x, pre = mhc.shifted_post_pre_rms(
            x, flat, post, comb, weight, scale, base, pre, norm_weight=norm, **layout
        )
        outs += [flat, post, comb, x, pre]
    return outs


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
@pytest.mark.parametrize("tokens", [96, 288, 2048])
def test_resident_gates_are_bitwise_identical(tokens):
    args = _inputs(tokens)
    resident = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.COMPILER,
    }
    ref = jax.jit(lambda *a: _layers({}, *a))(*args)
    got = jax.jit(lambda *a: _layers(resident, *a))(*args)
    for r, g in zip(ref, got, strict=True):
        assert r.shape == g.shape and r.dtype == g.dtype
        np.testing.assert_array_equal(
            np.ascontiguousarray(r).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
@pytest.mark.parametrize("tokens", [96, 288, 2048])
def test_early_prep_is_bitwise_identical(tokens):
    args = _inputs(tokens)
    resident = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.COMPILER,
    }
    ref = jax.jit(lambda *a: _layers(resident, *a))(*args)
    got = jax.jit(lambda *a: _layers({**resident, "early_prep": True}, *a))(*args)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(
            np.ascontiguousarray(r).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
def test_t240_matches_unfused_fallback_within_rounding(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_mhc_op as op,
    )

    x, flat, post, comb, pre, weight, scale, base, norm = _inputs(240)
    consts = (1e-20, 1e-6, 1e-6, 2.0, 20, 1e-6)

    def layer(*args):
        return op._post_pre_rms_jax(*args, *consts)

    args = (x, flat, post, comb, weight, scale, base, pre, norm)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_MHC_SHIFTED_T16", "0")
    ref = jax.jit(layer)(*args)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_MHC_SHIFTED_T16", "1")
    got = jax.jit(layer)(*args)
    # residual, value: one BF16 rounding step; gates: FP32 reassociation.
    for r, g, rtol in zip(ref, got, (2.0**-7, 1e-4, 1e-4, 2.0**-6, 1e-4)):
        r, g = np.asarray(r, np.float32), np.asarray(g, np.float32)
        bound = np.maximum(np.abs(r), np.abs(g)) * rtol + 1e-6
        assert np.all(np.abs(r - g) <= bound)
