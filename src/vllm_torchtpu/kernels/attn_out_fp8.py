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
"""Pallas kernel fusing the attention-output relayout with o_proj FP8 quant.

The ragged paged attention kernels emit their output head-major,
``[num_kv_heads, M, q_per_kv // 2, 2, head_dim]``, and the wrapper hands the
model ``o.swapaxes(1, 0).reshape(M, K)``. On the dynamic FP8 path XLA then
fuses ``abs_max`` into that transpose but materialises the scaled activation
as an ``[M, K]`` float32 temporary plus a float32 relayout copy before the
``o_proj`` ``dot_general``.

This kernel reads the attention output in its native layout (the
``reshape``/``transpose`` pair below folds away against the wrapper's), computes
the per-row ``abs_max`` scale and writes ``[M, K]`` ``float8_e4m3fn`` plus a
lane-broadcast ``[M, 128]`` float32 scale in one VMEM pass.

Numerics match ``xla_quantized_matmul(..., quantize_activation=True)``
bit-for-bit: ``scale = abs_max / 448`` (1 when the row is all zero),
``x_q = clip(f32(x) / scale)``, ``out = dot(x_q, w_q) * scale * w_scale``.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.quantized_matmul.util import get_max_min

ATTN_OUT_FP8_MIN_TOKENS = 32
_SCALE_LANES = 128
# Double-buffered bf16 input + fp8 output + scale must fit comfortably in VMEM.
_VMEM_BUDGET_BYTES = 24 * 1024 * 1024


def select_block_m(m: int, k: int) -> int | None:
    """Largest row tile whose double-buffered working set fits the budget.

    Multi-tile grids need ``bm % 32 == 0`` for 1-byte FP8 sublane packing.
    """
    bytes_per_row = 2 * (2 * k + k + _SCALE_LANES * 4)
    for cand in (512, 256, 128, 64, 32):
        if m >= cand and m % cand == 0 and cand * bytes_per_row <= _VMEM_BUDGET_BYTES:
            return cand
    return None


def can_use_attn_out_fp8(
    m: int, num_kv_heads: int, q_per_kv: int, head_dim: int
) -> bool:
    """True when the concrete token bucket can lower through the kernel.

    Called inside JAX lowering, where ``m`` is a static per-bucket int, so no
    symbolic token-count guards reach TorchDynamo.
    """
    if m < ATTN_OUT_FP8_MIN_TOKENS:
        return False
    if q_per_kv % 2 != 0 or head_dim % 128 != 0:
        return False
    return select_block_m(m, num_kv_heads * q_per_kv * head_dim) is not None


def _attn_out_fp8_kernel(
    x_ref,
    q_ref,
    s_ref,
    *,
    num_kv_heads: int,
    num_pairs: int,
    head_dim: int,
    quant_dtype: jnp.dtype,
) -> None:
    """``x_ref``: ``[kv, bm, q_per_kv // 2, 2, d]``; ``q_ref``: ``[bm, K]``."""
    d = head_dim
    slots = [
        (h, p, s) for h in range(num_kv_heads) for p in range(num_pairs) for s in (0, 1)
    ]

    def head(h: int, p: int, s: int) -> jax.Array:
        # Load the head pair as one lane-dense `[bm, 2 * d]` row block and
        # slice it; on v7x this beats indexing the size-2 axis directly.
        pair = x_ref[h, :, p, :, :].astype(jnp.float32)
        return pair.reshape(pair.shape[0], 2 * d)[:, s * d : (s + 1) * d]

    abs_max = None
    for h, p, s in slots:
        a = jnp.max(jnp.abs(head(h, p, s)), axis=1, keepdims=True)
        abs_max = a if abs_max is None else jnp.maximum(abs_max, a)
    assert abs_max is not None
    dtype_max, dtype_min = get_max_min(quant_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0.0, 1.0, scale)
    for i, (h, p, s) in enumerate(slots):
        q_ref[:, i * d : (i + 1) * d] = jnp.clip(
            head(h, p, s) / scale, dtype_min, dtype_max
        ).astype(quant_dtype)
    s_ref[...] = jnp.broadcast_to(scale, (scale.shape[0], _SCALE_LANES))


def quantize_attn_out(
    x: jax.Array,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
    """Quantize the ``[M, K]`` attention output per row to FP8.

    Returns ``(x_q [M, K], scale [M, 128] float32)``; column 0 of ``scale``
    is the per-row scale.
    """
    m, k = x.shape
    if k != num_kv_heads * q_per_kv * head_dim:
        raise ValueError(f"{k=} must equal {num_kv_heads=} * {q_per_kv=} * {head_dim=}")
    bm = select_block_m(m, k)
    if bm is None or q_per_kv % 2 != 0:
        raise ValueError(f"Unsupported attention-output quant shape {x.shape}")
    num_pairs = q_per_kv // 2
    # Undo the wrapper's `o.swapaxes(1, 0).reshape(M, K)`. XLA folds the two
    # reshape/transpose pairs, so the kernel reads the attention kernel's
    # output buffer directly instead of a relaid-out copy.
    x5 = x.reshape(m, num_kv_heads, num_pairs, 2, head_dim).transpose(1, 0, 2, 3, 4)
    return pl.pallas_call(
        functools.partial(
            _attn_out_fp8_kernel,
            num_kv_heads=num_kv_heads,
            num_pairs=num_pairs,
            head_dim=head_dim,
            quant_dtype=quant_dtype,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), quant_dtype),
            jax.ShapeDtypeStruct((m, _SCALE_LANES), jnp.float32),
        ],
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec(
                (num_kv_heads, bm, num_pairs, 2, head_dim),
                lambda i: (0, i, 0, 0, 0),
            )
        ],
        out_specs=[
            pl.BlockSpec((bm, k), lambda i: (i, 0)),
            pl.BlockSpec((bm, _SCALE_LANES), lambda i: (i, 0)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        name="attn_out_fp8_quant",
    )(x5)


def attn_out_fp8_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    num_kv_heads: int,
    q_per_kv: int,
    head_dim: int,
) -> jax.Array:
    """``o_proj`` W8A8 matmul with the fused attention-output quantization.

    ``w_q`` is ``[K, N]`` FP8 and ``w_s`` the ``[N]`` per-channel scale.
    """
    x_q, s_wide = quantize_attn_out(x, num_kv_heads, q_per_kv, head_dim, w_q.dtype)
    out = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    out *= s_wide[:, :1]
    out *= jnp.expand_dims(w_s, 0)
    return out.astype(x.dtype)
