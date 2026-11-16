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
"""Fused decode HC mix (interpret mode) against a JAX reference of the
rounding boundaries in hc_decode's module docstring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.hc_decode import (
    decode_token_tile,
    hc_decode_mix,
)

# Qwen3.8-Flash: 4 streams of hidden 2560, low rank 320; HC is TP-replicated,
# so the per-rank shapes are the full shapes.
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
F32, BF16 = jnp.float32, jnp.bfloat16


def _bf16(rng, shape, scale=1.0):
    return jnp.asarray(rng.normal(size=shape) * scale, BF16)


def _weights(seed):
    rng = np.random.default_rng(seed)
    return (
        _bf16(rng, (WIDTH,), 0.3),
        _bf16(rng, (LOWRANK, WIDTH), 0.02),
        _bf16(rng, (WIDTH, LOWRANK), 0.05),
        _bf16(rng, (BRANCHES, WIDTH), 0.02),
    )


def _norm(x, weight):
    values = x.astype(F32).reshape(len(x), BRANCHES, HIDDEN)
    variance = jnp.mean(values * values, axis=-1, keepdims=True)
    scale = 1.0 + weight.astype(F32).reshape(BRANCHES, HIDDEN)
    return (values * jax.lax.rsqrt(variance + EPS) * scale).astype(BF16)


def _reference(x, weight, down, up, inject):
    """(mixed, injection): the boundaries listed in hc_decode's docstring."""
    norm = _norm(x, weight)  # [T, hc, H]
    flat = norm.reshape(len(x), WIDTH).astype(F32)
    mid = (flat @ down.astype(F32).T).astype(BF16)
    gate = jax.nn.silu((mid / BRANCHES).astype(BF16).astype(F32)).astype(BF16)
    up_b = up.astype(F32).reshape(BRANCHES, HIDDEN, LOWRANK)
    logits = jnp.einsum("tr,bhr->tbh", gate.astype(F32), up_b).astype(BF16)
    weighted = jax.nn.sigmoid(logits.astype(F32)) * norm.astype(F32)
    mixed = (weighted.sum(axis=1) / BRANCHES).astype(BF16)
    inj_logits = (flat @ inject.astype(F32).T).astype(BF16).astype(F32)
    return mixed, 2.0 * jax.nn.sigmoid(inj_logits / BRANCHES)


def _combine(residual, block, injection):
    out = residual.astype(F32).reshape(len(residual), BRANCHES, HIDDEN)
    out = out + block.astype(F32)[:, None] * injection[:, :, None]
    return out.reshape(residual.shape).astype(BF16)


def _check_bf16(actual, expected, name):
    actual = np.asarray(jnp.asarray(actual, jnp.float32))
    expected = np.asarray(jnp.asarray(expected, jnp.float32))
    assert np.isfinite(actual).all(), name
    diff = np.abs(actual - expected)
    # Only the matmul accumulation order (and FMA contraction) differ from
    # the reference, so a BF16 rounding boundary can move by one ulp on a
    # small fraction of elements and propagate through the gate. The mixed
    # output is a sum of branch terms, so a one-ulp term change is bounded
    # by the tensor's scale rather than by the (possibly cancelled) result.
    scale = np.abs(expected).max()
    assert diff.max() <= 2.0**-7 * scale, (name, diff.max(), scale)
    assert np.linalg.norm(diff) <= 1e-3 * np.linalg.norm(expected), name
    assert (diff > 0).mean() <= 0.02, (name, (diff > 0).mean())


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 128])
def test_tile_choice(tokens):
    assert decode_token_tile(tokens) == tokens


def test_tile_choice_large():
    assert decode_token_tile(256) == 128
    assert decode_token_tile(512) == 128
    assert decode_token_tile(192) == 64
    assert decode_token_tile(136) is None
    assert decode_token_tile(1024) is None
    assert decode_token_tile(0) is None


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 128, 256])
def test_decode_mix_matches_reference(tokens):
    weight, down, up, inject = _weights(tokens)
    hyper = _bf16(np.random.default_rng(100 + tokens), (tokens, WIDTH))
    expected_mixed, expected_inj = _reference(hyper, weight, down, up, inject)
    mixed, inj = hc_decode_mix(
        hyper,
        weight,
        down,
        up,
        inject,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        interpret=pltpu.InterpretParams(),
    )
    _check_bf16(mixed, expected_mixed, "mixed")
    np.testing.assert_allclose(
        np.asarray(inj), np.asarray(expected_inj), rtol=2e-2, atol=1e-6
    )


@pytest.mark.parametrize("tokens", [4, 32, 128])
def test_decode_combine_mix_matches_reference(tokens):
    weight, down, up, inject = _weights(1000 + tokens)
    rng = np.random.default_rng(200 + tokens)
    residual = _bf16(rng, (tokens, WIDTH))
    block = _bf16(rng, (tokens, HIDDEN))
    prev_injection = jnp.asarray(rng.uniform(0, 2, (tokens, BRANCHES)), F32)
    expected_h = _combine(residual, block, prev_injection)
    expected_mixed, _ = _reference(expected_h, weight, down, up, inject)
    mixed, inj, h = hc_decode_mix(
        residual,
        weight,
        down,
        up,
        inject,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        block=block,
        prev_injection=prev_injection,
        interpret=pltpu.InterpretParams(),
    )
    # The combine is elementwise FP32 with one BF16 rounding; only FMA
    # contraction may move an element by one ulp.
    _check_bf16(h, expected_h, "h")
    _check_bf16(mixed, expected_mixed, "mixed")
    assert inj.shape == (tokens, BRANCHES)


def test_decode_mix_without_injection():
    weight, down, up, inject = _weights(7)
    hyper = _bf16(np.random.default_rng(7), (8, WIDTH))
    expected_mixed, _ = _reference(hyper, weight, down, up, inject)
    mixed, inj = hc_decode_mix(
        hyper,
        weight,
        down,
        up,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        interpret=pltpu.InterpretParams(),
    )
    assert inj is None
    _check_bf16(mixed, expected_mixed, "mixed")


def test_decode_mix_rejects_unsupported_shapes():
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN
    args = (
        shape((136, width), jnp.bfloat16),
        shape((width,), jnp.bfloat16),
        shape((LOWRANK, width), jnp.bfloat16),
        shape((width, LOWRANK), jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="decode-sized"):
        jax.eval_shape(
            lambda *a: hc_decode_mix(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS),
            *args,
        )


@pytest.mark.parametrize("missing", ["block", "prev_injection"])
def test_decode_combine_requires_block_and_injection(missing):
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN
    args = (
        shape((8, width), jnp.bfloat16),
        shape((width,), jnp.bfloat16),
        shape((LOWRANK, width), jnp.bfloat16),
        shape((width, LOWRANK), jnp.bfloat16),
    )
    prologue = {
        "block": shape((8, HIDDEN), jnp.bfloat16),
        "prev_injection": shape((8, BRANCHES), jnp.float32),
    }
    del prologue[missing]
    with pytest.raises(ValueError, match="both block and prev_injection"):
        jax.eval_shape(
            lambda *a, **k: hc_decode_mix(
                *a, branches=BRANCHES, hidden=HIDDEN, eps=EPS, **k
            ),
            *args,
            **prologue,
        )
