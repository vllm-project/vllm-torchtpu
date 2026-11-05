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
"""Unit tests for the parameterized QKV norm/RoPE prologue kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import prologue
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper

_FP8 = jnp.float8_e4m3fn


def _rms_ref(x: jax.Array, w: jax.Array | None, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return x32 if w is None else x32 * w


def _rotary_ref(y: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    half = cos.shape[-1]
    y1, y2 = y[..., :half], y[..., half : 2 * half]
    parts = [y1 * cos[:, None, :] - y2 * sin[:, None, :]]
    parts.append(y2 * cos[:, None, :] + y1 * sin[:, None, :])
    if 2 * half < y.shape[-1]:
        parts.append(y[..., 2 * half :])
    return jnp.concatenate(parts, axis=-1)


def _quant_ref(x: jax.Array, scale: float, dtype: jnp.dtype) -> jax.Array:
    info = jnp.finfo(dtype)
    val = x / scale if scale != 1.0 else x
    return jnp.clip(val, float(info.min), float(info.max)).astype(dtype)


def _reference_prologue(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    k_scale: float,
    v_scale: float,
    v_rms_norm: bool,
):
    tokens = qkv.shape[0]
    heads = qkv.reshape(tokens, num_q_heads + 2 * num_kv_heads, head_dim)
    q_raw = heads[:, :num_q_heads]
    k_raw = heads[:, num_q_heads : num_q_heads + num_kv_heads]
    v_raw = heads[:, num_q_heads + num_kv_heads :]

    cos32 = cos.astype(jnp.float32)
    sin32 = sin.astype(jnp.float32)
    wq32 = wq.astype(jnp.float32)
    wk32 = wk.astype(jnp.float32)

    q = _rotary_ref(_rms_ref(q_raw, wq32, eps), cos32, sin32).astype(jnp.bfloat16)
    k = _rotary_ref(_rms_ref(k_raw, wk32, eps), cos32, sin32).astype(jnp.bfloat16)
    v32 = _rms_ref(v_raw, None, eps) if v_rms_norm else v_raw.astype(jnp.float32)

    k_q = _quant_ref(k.astype(jnp.float32), k_scale, _FP8)
    v_q = _quant_ref(v32, v_scale, _FP8)
    ref_q_hbm, ref_kv_hbm = rpa_wrapper.prepare_inputs(
        q, k_q, v_q, jnp.bfloat16, _FP8, page_size=16
    )
    return ref_q_hbm, ref_kv_hbm


def _make_inputs(
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    width = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv = jnp.asarray(rng.standard_normal((tokens, width)), jnp.bfloat16)
    angles = rng.uniform(-1.0, 1.0, (tokens, rotary_dim // 2))
    cos = jnp.asarray(np.cos(angles), jnp.bfloat16)
    sin = jnp.asarray(np.sin(angles), jnp.bfloat16)
    wq = jnp.asarray(rng.uniform(0.5, 1.5, (head_dim,)), jnp.bfloat16)
    wk = jnp.asarray(rng.uniform(0.5, 1.5, (head_dim,)), jnp.bfloat16)
    return qkv, cos, sin, wq, wk


@pytest.mark.parametrize("v_rms_norm", [False, True])
@pytest.mark.parametrize(
    "tokens,nq,nkv,head_dim,rotary_dim",
    [
        (32, 8, 2, 128, 128),
        (20, 4, 2, 256, 128),
    ],
)
def test_prologue_lowers_for_tpu7x(
    tokens: int,
    nq: int,
    nkv: int,
    head_dim: int,
    rotary_dim: int,
    v_rms_norm: bool,
):
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    width = (nq + 2 * nkv) * head_dim
    qkv = jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16)
    cos = jax.ShapeDtypeStruct((tokens, rotary_dim // 2), jnp.bfloat16)
    sin = jax.ShapeDtypeStruct((tokens, rotary_dim // 2), jnp.bfloat16)
    wq = jax.ShapeDtypeStruct((head_dim,), jnp.bfloat16)
    wk = jax.ShapeDtypeStruct((head_dim,), jnp.bfloat16)
    emit_fp8 = (nq // nkv) % 4 == 0
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.jit(
            lambda a, c, s, q_w, k_w: prologue.qkv_to_rpa_inputs(
                a,
                c,
                s,
                q_w,
                k_w,
                num_q_heads=nq,
                num_kv_heads=nkv,
                head_dim=head_dim,
                eps=1e-6,
                k_scale=0.5,
                v_scale=0.75,
                v_rms_norm=v_rms_norm,
                emit_fp8_queries=emit_fp8,
                interpret=False,
            )
        ).trace(qkv, cos, sin, wq, wk)
        hlo = traced.lower(lowering_platforms=("tpu",)).as_text()
    assert len(traced.out_info) == (3 if emit_fp8 else 2)
    assert traced.out_info[0].dtype == jnp.bfloat16
    assert traced.out_info[1].dtype == _FP8
    if emit_fp8:
        assert traced.out_info[2].dtype == _FP8
    assert "tpu_custom_call" in hlo or "custom_call" in hlo


def test_prologue_validation():
    with jax.default_device(jax.devices("cpu")[0]):
        qkv, cos, sin, wq, wk = _make_inputs(16, 4, 2, 128, 128)
    with pytest.raises(ValueError, match="multiples of 4"):
        prologue.qkv_to_rpa_inputs(
            qkv,
            cos,
            sin,
            wq,
            wk,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=128,
            eps=1e-6,
            k_scale=1.0,
            v_scale=1.0,
            emit_fp8_queries=True,
            interpret=False,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("v_rms_norm", [False, True])
@pytest.mark.parametrize(
    "tokens,nq,nkv,head_dim,rotary_dim",
    [
        (32, 8, 2, 128, 128),
        (20, 4, 2, 256, 128),
        (64, 16, 2, 256, 128),
    ],
)
def test_prologue_tpu_matches_reference(
    tokens: int,
    nq: int,
    nkv: int,
    head_dim: int,
    rotary_dim: int,
    v_rms_norm: bool,
):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU runtime")
    qkv, cos, sin, wq, wk = _make_inputs(tokens, nq, nkv, head_dim, rotary_dim)
    emit_fp8 = (nq // nkv) % 4 == 0
    out = prologue.qkv_to_rpa_inputs(
        qkv,
        cos,
        sin,
        wq,
        wk,
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=head_dim,
        eps=1e-6,
        k_scale=0.5,
        v_scale=0.75,
        v_rms_norm=v_rms_norm,
        emit_fp8_queries=emit_fp8,
        interpret=False,
    )
    ref_q, ref_kv = _reference_prologue(
        qkv,
        cos,
        sin,
        wq,
        wk,
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=head_dim,
        eps=1e-6,
        k_scale=0.5,
        v_scale=0.75,
        v_rms_norm=v_rms_norm,
    )
    q_hbm, kv_hbm, *extra = out
    np.testing.assert_allclose(
        np.asarray(q_hbm, dtype=np.float32),
        np.asarray(ref_q, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(kv_hbm, dtype=np.float32),
        np.asarray(ref_kv, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    if emit_fp8:
        assert len(extra) == 1
        ref_q_fp8 = rpa_wrapper.quantize_queries(q_hbm, _FP8)
        np.testing.assert_array_equal(
            np.asarray(extra[0].view(jnp.uint8)),
            np.asarray(ref_q_fp8.view(jnp.uint8)),
        )
    else:
        assert not extra
