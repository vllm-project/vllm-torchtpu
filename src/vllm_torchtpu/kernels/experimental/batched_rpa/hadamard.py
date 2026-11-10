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
"""Hadamard rotation helpers for KV cache quantization.

The batched RPA kernel supports storing the KV cache rotated by a normalized
Sylvester-Hadamard matrix H along head_dim. H is symmetric and orthonormal
(H == H.T == inv(H)), so:

  * QK^T is preserved exactly when Q is rotated by the same H:
      (q H)(k H)^T = q H H^T k^T = q k^T
  * The PV product of rotated V is the rotated output, undone by one more
    application of H:
      p (v H) = (p v) H  =>  o = ((p v) H) H = p v

Rotating K/V before per-token quantization spreads per-channel outliers evenly
across head_dim, which reduces the quantization error of low-bit (e.g. FP4)
KV caches. Because the rotation of Q and the inverse rotation of the output
happen outside the Pallas kernel (but inside the same jit), the attention
kernel itself is unchanged.


The torch-side counterpart used at quantization time lives in
`vllm_torchtpu.layers.common.quantization` and must produce a bitwise
identical matrix; both build the same +-1 Sylvester pattern in float32 and
scale it by float32(1/sqrt(dim)).
"""

import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np


@functools.lru_cache(maxsize=None)
def hadamard_pm1_np(dim: int) -> np.ndarray:
  """Unnormalized +-1 Sylvester-Hadamard matrix as float32 numpy."""
  if dim <= 0 or dim & (dim - 1):
    raise ValueError(
        "Hadamard rotation requires head_dim to be a power of two, got"
        f" {dim}.")
  h = np.array([[1.0]], dtype=np.float32)
  while h.shape[0] < dim:
    h = np.block([[h, h], [h, -h]])
  return h


@functools.lru_cache(maxsize=None)
def hadamard_matrix_np(dim: int) -> np.ndarray:
  """Normalized Sylvester-Hadamard matrix as a float32 numpy array.

  H is built from the +-1 Sylvester recursion and scaled by 1/sqrt(dim), so
  H @ H.T == I and H == H.T (involutory).
  """
  return (hadamard_pm1_np(dim) *
          np.float32(1.0 / math.sqrt(dim))).astype(np.float32)


def hadamard_matrix(dim: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  """Normalized Sylvester-Hadamard matrix as a jax array."""
  return jnp.asarray(hadamard_matrix_np(dim), dtype=dtype)


def apply_hadamard(x: jax.Array,
                   out_dtype: jnp.dtype | None = None) -> jax.Array:
  """Rotates `x` by the normalized Hadamard matrix along the last dim.

  Computed as `(x @ H_pm1) * (1/sqrt(dim))`: the +-1 matrix is exact in any
  float dtype, so the matmul runs at native low-precision (e.g. bf16 MXU)
  throughput while accumulating in float32, and the normalization is a fused
  scalar multiply on the float32 result before casting back to `x.dtype`.
  This matches an all-float32 rotation up to one output rounding, but is ~8x
  cheaper than an fp32 matmul on TPU. Since H is involutory, this function is
  its own inverse (up to float rounding).

  `out_dtype` widens the matmul result before the normalization instead of
  keeping it in `x.dtype`. The default (None) is what the Q rotation and the
  output un-rotation want. Pass float32 to match the torch-side K/V rotation
  in `vllm_torchtpu.layers.common.quantization.apply_hadamard_transform`,
  which returns float32 so the following quantization sees full precision.
  """
  dim = x.shape[-1]
  h = jnp.asarray(hadamard_pm1_np(dim), x.dtype)
  # Performance notes (measured on v6e, [2048,32,128] bf16):
  #  * The dot must be 2D: a dot_general over the last dim of a >=3D operand
  #    lowers ~2.6x slower than reshape + plain 2D matmul.
  #  * Keep the output in x.dtype: the MXU accumulates in float32 internally
  #    either way, but a float32 result buffer forces an extra full-size
  #    convert pass over HBM.
  #  * This still costs a full extra HBM pass over `x`; when the rotation sits
  #    next to a relayout, prefer the fused rotate_*_relayout Pallas kernels
  #    below, which piggyback the rotation on the relayout's HBM pass.
  # The normalization is a scalar multiply fused into the dot's epilogue.
  flat = x.reshape(-1, dim)
  rotated = jax.lax.dot_general(
      flat,
      h,
      dimension_numbers=(((1,), (0,)), ((), ())),
      preferred_element_type=x.dtype,
  )
  if out_dtype is not None:
    rotated = rotated.astype(out_dtype)
  rotated = rotated * jnp.asarray(1.0 / math.sqrt(dim), rotated.dtype)
  return rotated.reshape(x.shape)


# ---------------------------------------------------------------------------
# Fused rotate + relayout Pallas kernels.
#
# The batched RPA wrapper has to physically relayout Q (token-major ->
# kv-head-major) and the attention output (kv-head-major -> token-major)
# around the attention kernel; Pallas inputs demand standard row-major
# layouts, so those transposes are real HBM passes even in the baseline.
# XLA will not fold such a materialized transpose into the rotation matmul,
# which would make the Hadamard rotation cost one extra full pass over
# Q / O (~10% of a short-prefill wrapper call). These kernels perform the
# already-required relayout and the 128x128 rotation in a single pass, so
# the rotation only adds MXU cycles to a memory-bound copy.
# ---------------------------------------------------------------------------

# Token-tile sizes for the fused kernels; T must be divisible by one of these
# (vLLM pads token counts to power-of-two-ish buckets, so this almost always
# holds; otherwise we fall back to the plain XLA path). Blocks span all kv
# heads (contiguous DMA on the token-major side); the tile is chosen as the
# largest divisor of T whose block stays within a modest VMEM budget.
_FUSED_T_BLOCKS = (512, 256, 128, 64, 32, 16, 8)
_FUSED_BLOCK_BYTE_BUDGET = 2 * 1024 * 1024


def pick_t_block(total_tokens: int, row_bytes: int) -> int | None:
  """Largest t block dividing total_tokens within the VMEM budget."""
  smallest_fit = None
  for t_blk in _FUSED_T_BLOCKS:
    if total_tokens % t_blk:
      continue
    smallest_fit = t_blk
    if t_blk * row_bytes <= _FUSED_BLOCK_BYTE_BUDGET:
      return t_blk
  return smallest_fit


def _rotate_relayout_kernel(x_ref, h_ref, o_ref, *, tokens_major_in: bool):
  """Rotates along d while swapping the leading (tokens, kv_heads) dims.

  One operand is [t_blk, kvh, gq, d] (token-major), the other
  [kvh, t_blk, gq, d] (head-major). Each kv head's slab is rotated with a
  single MXU matmul and written to its transposed destination, so the
  transpose costs no extra HBM traffic and no in-VMEM shuffle.
  """
  num_kv_heads = x_ref.shape[1] if tokens_major_in else x_ref.shape[0]
  d = x_ref.shape[-1]
  scale = np.float32(1.0 / math.sqrt(d))
  for head in range(num_kv_heads):
    x = x_ref[:, head] if tokens_major_in else x_ref[head]
    rot = jax.lax.dot_general(
        x.reshape(-1, d),
        h_ref[...],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    rot = (rot * scale).astype(o_ref.dtype)
    if tokens_major_in:
      o_ref[head] = rot.reshape(o_ref.shape[1:])
    else:
      o_ref[:, head] = rot.reshape(o_ref.shape[0], *o_ref.shape[2:])


def rotate_q_relayout(q: jax.Array, t_blk: int) -> jax.Array:
  """[T, kvh, gq, d] -> Hadamard-rotated [kvh, T, gq, d] in one pass."""
  total_tokens, num_kv_heads, gq, d = q.shape
  h = jnp.asarray(hadamard_pm1_np(d), q.dtype)
  return pl.pallas_call(
      functools.partial(_rotate_relayout_kernel, tokens_major_in=True),
      grid=(total_tokens // t_blk,),
      in_specs=[
          pl.BlockSpec((t_blk, num_kv_heads, gq, d), lambda t_i: (t_i, 0, 0, 0)),
          pl.BlockSpec((d, d), lambda t_i: (0, 0)),
      ],
      out_specs=pl.BlockSpec((num_kv_heads, t_blk, gq, d),
                             lambda t_i: (0, t_i, 0, 0)),
      out_shape=jax.ShapeDtypeStruct(
          (num_kv_heads, total_tokens, gq, d), q.dtype),
      name="hadamard_rotate_q_relayout",
  )(q, h)


def rotate_o_relayout(o: jax.Array, t_blk: int) -> jax.Array:
  """[kvh, T, gq, d] -> Hadamard-rotated [T, kvh, gq, d] in one pass."""
  num_kv_heads, total_tokens, gq, d = o.shape
  h = jnp.asarray(hadamard_pm1_np(d), o.dtype)
  return pl.pallas_call(
      functools.partial(_rotate_relayout_kernel, tokens_major_in=False),
      grid=(total_tokens // t_blk,),
      in_specs=[
          pl.BlockSpec((num_kv_heads, t_blk, gq, d), lambda t_i: (0, t_i, 0, 0)),
          pl.BlockSpec((d, d), lambda t_i: (0, 0)),
      ],
      out_specs=pl.BlockSpec((t_blk, num_kv_heads, gq, d),
                             lambda t_i: (t_i, 0, 0, 0)),
      out_shape=jax.ShapeDtypeStruct(
          (total_tokens, num_kv_heads, gq, d), o.dtype),
      name="hadamard_rotate_o_relayout",
  )(o, h)