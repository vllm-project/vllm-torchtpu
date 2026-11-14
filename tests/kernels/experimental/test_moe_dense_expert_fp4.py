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
"""TPU parity and routing edge cases for the FP4 (W4A16) dense-expert path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_dense_expert as kernel
from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper

pytestmark = pytest.mark.tpu_test

E, H, INTER, BLOCK, TOPK = 8, 2048, 256, 128, 4


def _weights(h):
    k = jax.random.split(jax.random.key(7), 4)
    fp4 = jnp.float4_e2m1fn
    w1 = jax.random.normal(k[0], (E, h, 2 * INTER), jnp.bfloat16).astype(fp4)
    w2 = jax.random.normal(k[1], (E, INTER, h), jnp.bfloat16).astype(fp4)
    s1 = jax.random.uniform(
        k[2], (E, h // BLOCK, 1, 2 * INTER), jnp.float32, 0.004, 0.008
    )
    s2 = jax.random.uniform(k[3], (E, INTER // BLOCK, 1, h), jnp.float32, 0.004, 0.008)
    return w1, w2, s1, s2


@pytest.fixture(scope="module")
def tensors():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    return _weights(H)


def _routing(m, valid, case):
    rng = np.random.default_rng(m)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(m)])
    weights = rng.uniform(0.1, 1.0, (m, TOPK)).astype(np.float32)
    if case == "single":
        ids[:, 0] = 5
        weights[:, 1:] = 0
    elif case == "duplicate":
        ids[:, :] = [3, 3, 6, 6]
    elif case == "zero":
        weights[:] = 0
    weights[valid:] = 0  # padded tokens carry no route weight
    return jnp.asarray(ids, jnp.int32), jnp.asarray(weights, jnp.bfloat16)


def _run(tensors, monkeypatch, m, valid, case, activation, ep, kernel_name=None):
    w1, w2, s1, s2 = tensors
    x = jax.random.normal(jax.random.key(m), (m, w1.shape[1]), jnp.bfloat16)
    ids, weights = _routing(m, valid, case)
    kwargs = dict(topk=TOPK, activation=activation, skip_padded_tokens=True)
    if ep:
        # This shard owns experts [4, 8): routes to 0..3 belong to another rank.
        w1, w2, s1, s2 = (t[4:] for t in (w1, w2, s1, s2))
        kwargs.update(
            experts_start=jnp.array(4, jnp.int32),
            use_ep=True,
            onehot_moe_permute_threshold=32768,
        )
    args = (x, w1, w2, s1, s2, None, None, weights, ids)
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 0)
    wrapper.fused_moe_func.clear_cache()
    expected = np.asarray(wrapper.fused_moe_func(*args, **kwargs), np.float32)
    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", 1024)
    wrapper.fused_moe_func.clear_cache()
    try:
        actual = np.asarray(wrapper.fused_moe_func(*args, **kwargs), np.float32)
        hlo = wrapper.fused_moe_func.lower(*args, **kwargs).as_text()
        assert (kernel_name or "dense_expert_moe_fp4-e_") in hlo
    finally:
        wrapper.fused_moe_func.clear_cache()
    return actual, expected


@pytest.mark.parametrize("activation", ["swigluoai", "silu"])
@pytest.mark.parametrize("case", ["mixed", "single", "duplicate"])
@pytest.mark.parametrize(("m", "valid"), [(8, 8), (16, 9), (24, 24), (64, 32)])
@pytest.mark.parametrize("ep", [False, True])
def test_matches_gmm_path(tensors, monkeypatch, m, valid, case, activation, ep):
    # Same weights and per-block arithmetic as gmm_v2; only the order of the
    # top-k weighted sum and its rounding differ.
    actual, expected = _run(tensors, monkeypatch, m, valid, case, activation, ep)
    _check(actual, expected, valid)


def _check(actual, expected, valid):
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[valid:], 0)
    rel = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert rel < 0.008, rel


def test_hidden_with_several_down_chunks(monkeypatch):
    """Hidden 3072 runs two 1536-column down-projection chunks."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    assert kernel._down_cols(3072, 128) == 1536
    actual, expected = _run(
        _weights(3072), monkeypatch, 16, 9, "mixed", "swigluoai", False
    )
    _check(actual, expected, 9)


@pytest.mark.parametrize("num_buffers", [2, 3])
def test_short_rings(tensors, monkeypatch, num_buffers):
    """The software pipeline with fewer ring slots than active experts."""
    monkeypatch.setattr(kernel, "_MAX_FP4_EXPERT_BUFFERS", num_buffers)
    kernel.dense_expert_moe_fp4.clear_cache()
    try:
        actual, expected = _run(
            tensors,
            monkeypatch,
            24,
            24,
            "mixed",
            "silu",
            False,
            kernel_name=f"-buffers_{num_buffers}",
        )
    finally:
        kernel.dense_expert_moe_fp4.clear_cache()
    _check(actual, expected, 24)


def test_zero_routes_return_zero(tensors, monkeypatch):
    actual, _ = _run(tensors, monkeypatch, 16, 16, "zero", "swigluoai", False)
    np.testing.assert_array_equal(actual, 0)


def test_wrong_scale_block_is_caught(tensors, monkeypatch):
    """Fault control: the tolerance above rejects a one-block scale shift."""
    w1, w2, s1, s2 = tensors
    x = jax.random.normal(jax.random.key(3), (16, H), jnp.bfloat16)
    ids, weights = _routing(16, 16, "mixed")
    good = kernel.dense_expert_moe_fp4(
        x, w1, w2, s1, s2, weights, ids, activation="swigluoai"
    )
    bad = kernel.dense_expert_moe_fp4(
        x, w1, w2, jnp.roll(s1, 1, axis=1), s2, weights, ids, activation="swigluoai"
    )
    good, bad = np.asarray(good, np.float32), np.asarray(bad, np.float32)
    assert np.linalg.norm(bad - good) / np.linalg.norm(good) > 0.05


@pytest.mark.parametrize(
    "unsupported", ["activation", "quant_dtype", "bias", "scale", "fp8_weights"]
)
def test_unsupported_inputs_keep_standard_path(tensors, unsupported):
    w1, w2, s1, s2 = tensors
    x = jnp.zeros((16, H), jnp.bfloat16)
    b1, activation, quant_dtype = None, "swigluoai", None
    if unsupported == "activation":
        activation = "gelu"
    elif unsupported == "quant_dtype":
        quant_dtype = jnp.int4
    elif unsupported == "bias":
        b1 = jnp.zeros((E, 1, 2 * INTER), jnp.float32)
    elif unsupported == "scale":
        s1 = None
    else:
        w1 = w1.astype(jnp.float8_e4m3fn)
    assert not kernel.can_use_dense_expert_fp4(
        x, w1, w2, s1, s2, b1, None, activation=activation, rhs_quant_dtype=quant_dtype
    )
