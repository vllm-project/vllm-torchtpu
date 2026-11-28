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
"""Fused q/gate split, per-head RMS norm, partial NeoX RoPE and gate copy for
Qwen3-Next and Qwen3.5 attention, as one Pallas call per layer for a few rows.

Inputs:

  q_gate   [rows, H * 2 * D] bf16  per head: [q | gate]
  k        [rows, Hk * D]    bf16
  q_w, k_w [D] f32                 the norm weights, already including the + 1
  cos_sin  [rows, R]               the rope cache rows for these positions:
                                   [cos (R/2) | sin (R/2)]

Outputs: q [rows, H*D] and k [rows, Hk*D] (normed, rotated on the first R
dims) and gate [rows, H*D] (pre-sigmoid), all in the input dtype. The math
matches vLLM's eager path: norm in f32 (x * rsqrt(mean(x^2) + eps) * w), rope
on the normed values, NeoX rotation of the halves of the first R dims. Rope is
done in f32 before one cast, so results can differ from the eager path by one
bf16 rounding.

All rows are held in VMEM at once, so callers keep the row count small.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _rms_norm(x, weight, eps):
    variance = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * weight


def _neox_rope(x, cos, sin, rotary_dim):
    """x [rows, D] f32; rotates the first rotary_dim dims NeoX-style with
    cos/sin [rows, rotary_dim / 2]. The remaining dims pass through."""
    half = rotary_dim // 2
    x1, x2 = x[:, :half], x[:, half:rotary_dim]
    parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    if rotary_dim < x.shape[1]:
        parts.append(x[:, rotary_dim:])
    return jnp.concatenate(parts, axis=1)


def _kernel(
    q_gate_ref,
    k_ref,
    q_weight_ref,
    k_weight_ref,
    cos_sin_ref,
    q_out_ref,
    k_out_ref,
    gate_out_ref,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
):
    half = rotary_dim // 2
    cos_sin = cos_sin_ref[...]
    cos = cos_sin[:, :half]
    sin = cos_sin[:, half:rotary_dim]
    q_weight = q_weight_ref[...]
    k_weight = k_weight_ref[...]
    for head in range(num_heads):
        q_start = head * 2 * head_dim
        q = q_gate_ref[:, q_start : q_start + head_dim].astype(jnp.float32)
        gate = q_gate_ref[:, q_start + head_dim : q_start + 2 * head_dim]
        q = _neox_rope(_rms_norm(q, q_weight, eps), cos, sin, rotary_dim)
        out_cols = slice(head * head_dim, (head + 1) * head_dim)
        q_out_ref[:, out_cols] = q.astype(q_out_ref.dtype)
        gate_out_ref[:, out_cols] = gate.astype(gate_out_ref.dtype)
    for head in range(num_kv_heads):
        cols = slice(head * head_dim, (head + 1) * head_dim)
        k = k_ref[:, cols].astype(jnp.float32)
        k = _neox_rope(_rms_norm(k, k_weight, eps), cos, sin, rotary_dim)
        k_out_ref[:, cols] = k.astype(k_out_ref.dtype)


_STATIC_ARGNAMES = (
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "rotary_dim",
    "eps",
    "interpret",
)


@functools.partial(jax.jit, static_argnames=_STATIC_ARGNAMES)
def qk_norm_rope_gate(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    interpret=False,
):
    """Gathers the rope rows by position, then runs qk_norm_rope_gate_rows.

    cos_sin_cache is [max_position, rotary_dim]; positions is [rows] int."""
    cos_sin = jnp.take(cos_sin_cache, positions.astype(jnp.int32), axis=0)
    return qk_norm_rope_gate_rows(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        interpret=interpret,
    )


@functools.partial(jax.jit, static_argnames=_STATIC_ARGNAMES)
def qk_norm_rope_gate_rows(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    interpret=False,
):
    """Runs the kernel on rope rows already gathered by position.

    q_gate [rows, num_heads * 2 * head_dim], k [rows, num_kv_heads * head_dim],
    q_weight and k_weight [head_dim] (already including the + 1), cos_sin
    [rows, rotary_dim] in any float dtype. The rows go through whole, with no
    padding. Returns q, k (normed, rotated) and the raw gate in the input dtype.
    """
    rows = q_gate.shape[0]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    kernel = functools.partial(
        _kernel,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, num_heads * head_dim), q_gate.dtype),
            jax.ShapeDtypeStruct((rows, num_kv_heads * head_dim), k.dtype),
            jax.ShapeDtypeStruct((rows, num_heads * head_dim), q_gate.dtype),
        ),
        in_specs=[vmem] * 5,
        out_specs=(vmem, vmem, vmem),
        name="qk_norm_rope_gate",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(
        q_gate,
        k,
        q_weight.reshape(1, head_dim).astype(jnp.float32),
        k_weight.reshape(1, head_dim).astype(jnp.float32),
        cos_sin.astype(jnp.float32),
    )


def reference(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
):
    """vLLM's eager path in plain JAX: cos/sin gather, q/gate split, per-head
    norm in f32, NeoX rope on the first rotary_dim dims."""
    cos_sin = jnp.take(cos_sin_cache, positions.astype(jnp.int32), axis=0)
    rows = q_gate.shape[0]
    q_and_gate = q_gate.reshape(rows, num_heads, 2 * head_dim)
    q, gate = q_and_gate[:, :, :head_dim], q_and_gate[:, :, head_dim:]
    k_heads = k.reshape(rows, num_kv_heads, head_dim)

    def norm(x, weight):
        x = x.astype(jnp.float32)
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)

    half = rotary_dim // 2
    cos = cos_sin[:, :half].astype(jnp.float32)[:, None, :]
    sin = cos_sin[:, half:rotary_dim].astype(jnp.float32)[:, None, :]

    def rope(x):
        x1, x2 = x[..., :half], x[..., half:rotary_dim]
        rest = x[..., rotary_dim:]
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    q = rope(norm(q, q_weight)).astype(q_gate.dtype)
    k_heads = rope(norm(k_heads, k_weight)).astype(k.dtype)
    return (
        q.reshape(rows, num_heads * head_dim),
        k_heads.reshape(rows, num_kv_heads * head_dim),
        gate.reshape(rows, num_heads * head_dim),
    )
