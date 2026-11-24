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
"""Skinny matmul against MXFP8 weights: e4m3 values with power-of-two scales
over blocks of 32 contraction rows.

The weight is a VMEM operand, so it moves from HBM as e4m3 (half the bytes of
bf16) in an async copy XLA schedules, which it can overlap with the ops before
the call. The kernel dequantizes it to bf16 one column chunk at a time. An
e4m3 value times a power of two is exact in bf16, so each product is the one a
bf16 matmul against the dequantized weight computes; only the f32 accumulation
order may differ.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MX_BLOCK = 32
# Weight elements dequantized at a time: keeps the bf16 temporaries (<= 6 MiB)
# within the default scoped VMEM limit.
_CHUNK_ELEMS = 6144 * 256
# Callers keep the operands XLA places in VMEM plus the temporaries within this
# share of a tpu7x TensorCore's 64 MiB, and the e4m3 weight within the largest
# size tested on TPU (6144 x 3072); 6144 x 6144 (36 MiB) ran out of scoped VMEM.
VMEM_BUDGET_BYTES = 42 << 20
MAX_WEIGHT_BYTES = 18 << 20


def _column_chunk(k: int, n: int) -> int | None:
    """The widest multiple of 128 dividing n with k * chunk <= _CHUNK_ELEMS
    (at least 128 columns)."""
    for tn in range(min(max(_CHUNK_ELEMS // k, 128), n) // 128 * 128, 0, -128):
        if n % tn == 0:
            return tn
    return None


def vmem_bytes(m: int, k: int, n: int) -> int:
    """The e4m3 weight and its scales, x and out (VMEM operands), plus two bf16
    temporaries of a column chunk and an f32 result. Deliberately high (x and
    out at 4 bytes): the budgets below were checked on TPU against it."""
    chunk = k * (_column_chunk(k, n) or n)
    return k * n + k // MX_BLOCK * n * 2 + 2 * chunk * 2 + (m * k + 3 * m * n) * 4


def fits(m: int, k: int, n: int) -> bool:
    """Whether `mxfp8_matmul` tiles this shape within its VMEM envelope."""
    return (
        _column_chunk(k, n) is not None
        and k * n <= MAX_WEIGHT_BYTES
        and vmem_bytes(m, k, n) <= VMEM_BUDGET_BYTES
    )


def _kernel(x_ref, w_ref, s_ref, o_ref, *, tn: int):
    k, n = w_ref.shape
    x = x_ref[...]
    for j in range(n // tn):
        cols = slice(j * tn, (j + 1) * tn)
        w = w_ref[:, cols].astype(jnp.bfloat16).reshape(k // MX_BLOCK, MX_BLOCK, tn)
        w = (w * s_ref[:, cols][:, None, :]).reshape(k, tn)
        o_ref[:, cols] = jnp.dot(x, w, preferred_element_type=jnp.float32).astype(
            o_ref.dtype
        )


@jax.jit
def mxfp8_matmul(
    x: jax.Array,  # [m, k] bf16
    w: jax.Array,  # [k, n] float8_e4m3fn
    scale: jax.Array,  # [k // 32, n] bf16 powers of two
) -> jax.Array:
    """``x @ (w * repeat(scale, 32, axis=0))`` in x's dtype, f32 accumulation.

    Callers check `fits` first.
    """
    m, k = x.shape
    n = w.shape[1]
    if w.shape[0] != k or k % MX_BLOCK or scale.shape != (k // MX_BLOCK, n):
        raise ValueError(
            f"Shapes do not match: x {x.shape}, w {w.shape}, scale {scale.shape}."
        )
    if not fits(m, k, n):
        raise ValueError(f"mxfp8_matmul does not fit m={m}, k={k}, n={n}.")
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        functools.partial(_kernel, tn=_column_chunk(k, n)),
        in_specs=[vmem, vmem, vmem],
        out_specs=vmem,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        name="mxfp8_matmul",
    )(x, w, scale)
