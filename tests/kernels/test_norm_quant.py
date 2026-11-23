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
"""Pallas interpret and TPU tests for sandwich and MoE norm-quant kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import norm_quant


def _rms(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    return (
        x32
        * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
        * w.astype(jnp.float32)
    ).astype(x.dtype)


def _fp8_quant(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    finfo, x32 = jnp.finfo(jnp.float8_e4m3fn), x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) / float(finfo.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    return (
        jnp.clip(x32 / scale, float(finfo.min), float(finfo.max)).astype(
            jnp.float8_e4m3fn
        ),
        scale,
    )


def _assert_fp8_pair(
    q: jax.Array,
    s: jax.Array,
    ref_q: jax.Array,
    ref_s: jax.Array,
    cpu_tol: float = 2e-2,
) -> None:
    deq = np.asarray(q.astype(jnp.float32) * s)
    ref_deq = np.asarray(ref_q.astype(jnp.float32) * ref_s)
    if jax.default_backend() == "cpu":
        np.testing.assert_allclose(
            np.asarray(s), np.asarray(ref_s), rtol=2e-2, atol=1e-6
        )
        np.testing.assert_allclose(deq, ref_deq, rtol=cpu_tol, atol=cpu_tol)
    else:
        np.testing.assert_allclose(
            np.asarray(s), np.asarray(ref_s), rtol=2.5e-2, atol=1e-6
        )
        assert float(np.linalg.norm(deq - ref_deq) / np.linalg.norm(ref_deq)) < 3e-2
        np.testing.assert_allclose(deq, ref_deq, rtol=0.16, atol=4e-2)


@pytest.mark.parametrize("with_scalar", [False, True])
def test_add_rms_norm_quantize_matches_reference(with_scalar: bool) -> None:
    t, h, eps = 16, 256, 1e-6
    keys = jax.random.split(jax.random.key(11 + int(with_scalar)), 5)
    x = jax.random.normal(keys[0], (t, h), dtype=jnp.bfloat16)
    r = jax.random.normal(keys[1], (t, h), dtype=jnp.bfloat16)
    w_post = jax.random.uniform(keys[2], (h,), dtype=jnp.bfloat16) + 0.5
    w_next = jax.random.uniform(keys[3], (h,), dtype=jnp.bfloat16) + 0.5
    scalar = jnp.array([0.75], dtype=jnp.bfloat16) if with_scalar else None

    h_out, q_bytes, scale = norm_quant.add_rms_norm_quantize(
        x,
        r,
        w_post,
        w_next,
        scalar,
        eps=eps,
        quant_dtype=jnp.float8_e4m3fn,
        bytes_out=True,
        interpret=True,
    )
    q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
    ref_h = (_rms(x, w_post, eps) + r).astype(jnp.bfloat16)
    if scalar is not None:
        ref_h = (ref_h.astype(jnp.float32) * scalar.astype(jnp.float32)).astype(
            jnp.bfloat16
        )
    ref_q, ref_s = _fp8_quant(_rms(ref_h, w_next, eps))
    np.testing.assert_allclose(
        np.asarray(h_out, np.float32),
        np.asarray(ref_h, np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    _assert_fp8_pair(q, scale, ref_q, ref_s, cpu_tol=2e-2)


def test_moe_tail_and_head_rms_norm_quantize_match_reference() -> None:
    t, h, e, eps = 16, 256, 128, 1e-6
    keys = jax.random.split(jax.random.key(42), 10)
    mlp, moe, res = (jax.random.normal(keys[i], (t, h), jnp.bfloat16) for i in range(3))
    w_mlp, w_moe, w_post, w_next = (
        jax.random.uniform(keys[i], (h,), jnp.bfloat16) + 0.5 for i in range(3, 7)
    )
    scalar = jnp.array([0.5], dtype=jnp.bfloat16)

    h_tail, q_tail, s_tail = norm_quant.moe_tail_rms_norm_quantize(
        mlp,
        moe,
        res,
        w_mlp,
        w_moe,
        w_post,
        scalar,
        w_next,
        eps=eps,
        quant_dtype=jnp.float8_e4m3fn,
        bytes_out=False,
        interpret=True,
    )
    combined = (
        _rms(mlp, w_mlp, eps).astype(jnp.float32)
        + _rms(moe, w_moe, eps).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    ref_h_tail = (
        (_rms(combined, w_post, eps).astype(jnp.float32) + res.astype(jnp.float32))
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
        * scalar.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    ref_q_tail, ref_s_tail = _fp8_quant(_rms(ref_h_tail, w_next, eps))
    np.testing.assert_allclose(
        np.asarray(h_tail, np.float32),
        np.asarray(ref_h_tail, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    _assert_fp8_pair(q_tail, s_tail, ref_q_tail, ref_s_tail, cpu_tol=3e-2)

    w_router = jax.random.normal(keys[7], (e, h), dtype=jnp.bfloat16) * 0.1
    router_scale = jax.random.uniform(keys[8], (h,), dtype=jnp.bfloat16) + 0.5
    root = h**-0.5
    h_head, q_mlp, s_mlp, m_moe, logits = norm_quant.moe_head_rms_norm_quantize(
        mlp,
        res,
        w_post,
        w_next,
        w_moe,
        w_router,
        router_scale,
        eps=eps,
        router_root=root,
        quant_dtype=jnp.float8_e4m3fn,
        bytes_out=False,
        interpret=True,
    )
    assert (
        h_head.shape == (t, h)
        and q_mlp.shape == (t, h)
        and s_mlp.shape == (t, 1)
        and m_moe.shape == (t, h)
        and logits.shape == (t, e)
    )
    ref_h_head = (
        _rms(mlp, w_post, eps).astype(jnp.float32) + res.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    ref_q_mlp, ref_s_mlp = _fp8_quant(_rms(ref_h_head, w_next, eps))
    ref_m_moe = _rms(ref_h_head, w_moe, eps)
    h32 = ref_h_head.astype(jnp.float32)
    xr = (
        (h32 * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + eps)).astype(
            jnp.bfloat16
        )
        * jnp.asarray(root, jnp.bfloat16)
        * router_scale
    )
    ref_logits = (
        jnp.dot(xr, w_router.T, preferred_element_type=jnp.float32)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    for got, ref, tol in (
        (h_head, ref_h_head, 1e-2),
        (m_moe, ref_m_moe, 1e-2),
        (logits, ref_logits, 2e-2),
    ):
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(ref, np.float32), rtol=tol, atol=tol
        )
    _assert_fp8_pair(q_mlp, s_mlp, ref_q_mlp, ref_s_mlp, cpu_tol=3e-2)


@pytest.mark.tpu_test
def test_norm_quant_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    t, h, eps = 16, 512, 1e-6
    k1, k2, k3, k4 = jax.random.split(jax.random.key(707), 4)
    x = jax.random.normal(k1, (t, h), dtype=jnp.bfloat16)
    r = jax.random.normal(k2, (t, h), dtype=jnp.bfloat16)
    w = jax.random.uniform(k3, (h,), dtype=jnp.bfloat16) + 0.5
    w_next = jax.random.uniform(k4, (h,), dtype=jnp.bfloat16) + 0.5

    h_out, q_nq, s_nq = norm_quant.add_rms_norm_quantize(
        x, r, w, w_next, None, eps=eps, quant_dtype=jnp.float8_e4m3fn
    )
    ref_h = (_rms(x, w, eps) + r).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(h_out, np.float32),
        np.asarray(ref_h, np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    _assert_fp8_pair(q_nq, s_nq, *_fp8_quant(_rms(ref_h, w_next, eps)))
