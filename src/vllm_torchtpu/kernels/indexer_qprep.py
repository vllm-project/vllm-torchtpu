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
"""Fused DSA indexer query prep (half-split RoPE + FP8 quantize) in one Pallas pass."""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes  # pyrefly: ignore[missing-import]
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_TOKENS = None
MIN_BLOCK_TOKENS = 8


def _bf16_xla(x):
    return jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)


def _rot_half(x, rope_dim: int):
    half = rope_dim // 2
    lanes = x.shape[-1]
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    fwd = pltpu.roll(x, lanes - half, x.ndim - 1)
    bwd = pltpu.roll(x, half, x.ndim - 1)
    return jnp.where(lane < half, fwd, jnp.where(lane < rope_dim, bwd, x))


def _quant_rows_bf16(y, q_max: float, quant_dtype):
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    recip = jnp.bfloat16(float(ml_dtypes.bfloat16(1.0 / q_max)))
    scale = amax * recip
    scale_f = scale.astype(jnp.float32)
    inv = (jnp.float32(1.0) / scale_f).astype(jnp.bfloat16)
    inv = jnp.where(scale == 0, jnp.bfloat16(0), inv)
    v = jnp.clip(y * inv, jnp.bfloat16(-q_max), jnp.bfloat16(q_max))
    return v.astype(quant_dtype), scale_f


def _kernel(
    x_ref,
    c_ref,
    s_ref,
    q_ref,
    scale_ref,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    q_max: float,
):
    t = x_ref.shape[0]
    c = c_ref[...]
    s = s_ref[...]
    heads = [x_ref[:, h * head_dim : (h + 1) * head_dim] for h in range(n_head)]
    x = jnp.stack(heads, axis=0).transpose(1, 0, 2)
    c3, s3 = c.reshape(t, 1, head_dim), s.reshape(t, 1, head_dim)
    y = x * c3 + _rot_half(x, rope_dim) * s3
    q, scale = _quant_rows_bf16(y, q_max, q_ref.dtype)
    q_ref[...] = q
    scale_ref[...] = scale.reshape(scale_ref.shape)


def _block_tokens(num_tokens: int, block_tokens: int) -> int:
    bt = min(block_tokens, num_tokens)
    while num_tokens % bt:
        bt //= 2
    return bt


@functools.partial(
    jax.jit,
    static_argnames=("n_head", "rope_dim", "quant_dtype", "block_tokens", "interpret"),
)
def indexer_q_prep(
    q: jax.Array,
    c: jax.Array,
    s: jax.Array,
    *,
    n_head: int,
    rope_dim: int,
    quant_dtype=jnp.float8_e4m3fn,
    block_tokens: int | None = DEFAULT_BLOCK_TOKENS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(q8 [T, H, D] quant_dtype, scale f32 [T, H])``."""
    num_tokens, width = q.shape
    head_dim = c.shape[-1]
    assert width == n_head * head_dim, (q.shape, n_head, head_dim)
    assert head_dim % 128 == 0 and rope_dim % 2 == 0 and rope_dim <= head_dim
    pad = (-num_tokens) % MIN_BLOCK_TOKENS if num_tokens < MIN_BLOCK_TOKENS else 0
    if pad:
        q = jnp.pad(q, ((0, pad), (0, 0)))
        c = jnp.pad(c, ((0, pad), (0, 0)))
        s = jnp.pad(s, ((0, pad), (0, 0)))
    rows = num_tokens + pad
    if block_tokens is None:
        block_tokens = 512 if rows > 1024 else 256
    bt = _block_tokens(rows, block_tokens)
    if bt < MIN_BLOCK_TOKENS or bt % MIN_BLOCK_TOKENS != 0:
        q8, scale = reference(
            q, c, s, n_head=n_head, rope_dim=rope_dim, quant_dtype=quant_dtype
        )
        return (q8[:num_tokens], scale[:num_tokens]) if pad else (q8, scale)
    q_max = float(jnp.finfo(quant_dtype).max)
    vmem_limit = (
        pltpu.get_tpu_info().vmem_capacity_bytes
        if not interpret and jax.default_backend() == "tpu"
        else 64 * 1024 * 1024
    )
    q8, scale = pl.pallas_call(
        functools.partial(
            _kernel, n_head=n_head, head_dim=head_dim, rope_dim=rope_dim, q_max=q_max
        ),
        grid=(rows // bt,),
        in_specs=[
            pl.BlockSpec((bt, width), lambda i: (i, 0)),
            pl.BlockSpec((bt, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((bt, head_dim), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bt, n_head, head_dim), lambda i: (i, 0, 0)),
            pl.BlockSpec((bt, n_head), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, n_head, head_dim), quant_dtype),
            jax.ShapeDtypeStruct((rows, n_head), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        name=f"indexer_qprep_bf16_h{n_head}_d{head_dim}_r{rope_dim}_b{bt}",
        interpret=interpret,
    )(q, c, s)
    if pad:
        q8, scale = q8[:num_tokens], scale[:num_tokens]
    return q8, scale


def reference(q, c, s, *, n_head: int, rope_dim: int, quant_dtype=jnp.float8_e4m3fn):
    """Plain-XLA reference matching Torch's per-op bf16 rounding on ``[T, H, D]``."""
    num_tokens = q.shape[0]
    head_dim = c.shape[-1]
    half = rope_dim // 2
    q_max = float(jnp.finfo(quant_dtype).max)
    x = q.reshape(num_tokens, n_head, head_dim).astype(jnp.float32)
    rot = jnp.concatenate([x[..., half:rope_dim], x[..., :half], x[..., rope_dim:]], -1)
    c_f = c.astype(jnp.float32)[:, None]
    s_f = s.astype(jnp.float32)[:, None]
    y = _bf16_xla(_bf16_xla(x * c_f) + _bf16_xla(rot * s_f))
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    recip = float(ml_dtypes.bfloat16(1.0 / q_max))
    scale = _bf16_xla(_bf16_xla(amax) * jnp.float32(recip))
    inv = _bf16_xla(jnp.float32(1.0) / scale)
    inv = jnp.where(scale == 0, jnp.float32(0.0), inv)
    v = jnp.clip(_bf16_xla(y * inv), -q_max, q_max)
    return v.astype(quant_dtype), scale[..., 0]
