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
"""DSA indexer query prep (RoPE via ``p_mat`` + per-(token, head) FP8 quant) in Pallas.

``VllmTPUIndexer.forward`` turns ``wq_b(qr)`` bf16 ``[T, H*D]`` into the top-k
kernel's query operands via:

    q         = q.view(T, H, D)
    q         = q * cos_full[:, None, :] + (q @ p_mat) * sin_full[:, None, :]
    q8, scale = quantize_tensor(q.view(T * H, D), float8_e4m3fn, use_ue8m0=False)

which XLA lowers as four HBM passes (relayout, RoPE, amax, convert). This
kernel computes every ``(token, head)`` row in VMEM in one read of ``q`` and one
write of ``q8`` while reproducing ``VllmTPUIndexer.forward``'s MXU ``q @ p_mat``
and ``layers/core/quantization.py`` fp32 reciprocal rounding bit-for-bit.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MIN_BLOCK_TOKENS = 8


def _quant_rows_exact(
    y: jax.Array, q_max: float, quant_dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """Match ``layers.core.quantization.quantize_tensor(..., use_ue8m0=False)``."""
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    # PyTorch keeps ``abs_max``'s dtype (bf16) when multiplying by a 0-D fp32
    # tensor ``dtype_max_recip``, then casts ``scale`` to fp32 on return.
    recip = jnp.float32(1.0 / q_max).astype(y.dtype)
    scale = amax * recip
    one = jnp.asarray(1.0, dtype=scale.dtype)
    scale_inv = jnp.where(scale == 0, jnp.zeros_like(scale), one / scale)
    v = jnp.clip(y * scale_inv, -q_max, q_max)
    return v.astype(quant_dtype), scale.astype(jnp.float32)


def _kernel(
    x_ref,
    c_ref,
    s_ref,
    p_ref,
    q_ref,
    scale_ref,
    *,
    n_head: int,
    head_dim: int,
    q_max: float,
) -> None:
    t = x_ref.shape[0]
    p_mat = p_ref[...]
    c3 = c_ref[...].reshape(t, 1, head_dim)
    s3 = s_ref[...].reshape(t, 1, head_dim)
    heads = [x_ref[:, h * head_dim : (h + 1) * head_dim] for h in range(n_head)]
    x = jnp.stack(heads, axis=0).transpose(1, 0, 2)
    rot = jnp.dot(x, p_mat, preferred_element_type=jnp.float32).astype(x.dtype)
    y = x * c3 + rot * s3
    q8, scale = _quant_rows_exact(y, q_max, q_ref.dtype)
    q_ref[...] = q8
    scale_ref[...] = scale.reshape(scale_ref.shape)


def _block_tokens(num_tokens: int, block_tokens: int) -> int:
    bt = min(block_tokens, num_tokens)
    while num_tokens % bt:
        bt //= 2
    return bt


def reference(
    q: jax.Array,
    cos_full: jax.Array,
    sin_full: jax.Array,
    p_mat: jax.Array,
    *,
    n_head: int,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
    """Reference JAX lowering matching ``VllmTPUIndexer.forward``."""
    num_tokens = q.shape[0]
    head_dim = cos_full.shape[-1]
    x = q.reshape(num_tokens, n_head, head_dim)
    rot = jnp.dot(x, p_mat, preferred_element_type=jnp.float32).astype(x.dtype)
    y = x * cos_full[:, None, :] + rot * sin_full[:, None, :]
    q8, scale = _quant_rows_exact(y, float(jnp.finfo(quant_dtype).max), quant_dtype)
    return q8, scale[..., 0]


@functools.partial(
    jax.jit,
    static_argnames=("n_head", "quant_dtype", "block_tokens", "interpret"),
)
def indexer_q_prep(
    q: jax.Array,
    cos_full: jax.Array,
    sin_full: jax.Array,
    p_mat: jax.Array,
    *,
    n_head: int,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    block_tokens: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(q8 [T, H, D] quant_dtype, scale f32 [T, H])``."""
    num_tokens, width = q.shape
    head_dim = cos_full.shape[-1]
    if width != n_head * head_dim:
        raise ValueError(f"{q.shape=} incompatible with {n_head=}, {head_dim=}")
    if head_dim % 128 != 0 or p_mat.shape != (head_dim, head_dim):
        return reference(
            q, cos_full, sin_full, p_mat, n_head=n_head, quant_dtype=quant_dtype
        )
    pad = (-num_tokens) % MIN_BLOCK_TOKENS if num_tokens < MIN_BLOCK_TOKENS else 0
    if pad:
        q = jnp.pad(q, ((0, pad), (0, 0)))
        cos_full = jnp.pad(cos_full, ((0, pad), (0, 0)))
        sin_full = jnp.pad(sin_full, ((0, pad), (0, 0)))
    rows = num_tokens + pad
    if block_tokens is None:
        block_tokens = 256 if rows >= 256 else rows
    bt = _block_tokens(rows, block_tokens)
    if bt < MIN_BLOCK_TOKENS:
        return reference(
            q[:num_tokens],
            cos_full[:num_tokens],
            sin_full[:num_tokens],
            p_mat,
            n_head=n_head,
            quant_dtype=quant_dtype,
        )
    q_max = float(jnp.finfo(quant_dtype).max)
    q8, scale = pl.pallas_call(
        functools.partial(
            _kernel,
            n_head=n_head,
            head_dim=head_dim,
            q_max=q_max,
        ),
        grid=(rows // bt,),
        in_specs=[
            pl.BlockSpec((bt, width), lambda i: (i, 0)),
            pl.BlockSpec((bt, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((bt, head_dim), lambda i: (i, 0)),
            pl.BlockSpec(
                (head_dim, head_dim),
                lambda i: (0, 0),
                pipeline_mode=pl.Buffered(1),
            ),
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
            dimension_semantics=("parallel",),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=f"indexer_qprep_h{n_head}_d{head_dim}_b{bt}",
        interpret=interpret,
    )(q, cos_full, sin_full, p_mat)
    if pad:
        q8, scale = q8[:num_tokens], scale[:num_tokens]
    return q8, scale
