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
"""Per-head Gemma RMSNorm + partial NeoX RoPE of MiniMax M3's fused qkv rows.

One Pallas pass over ``qkv [T, (nq + 3 * nkv + 1) * D]`` (heads ``[q | k | v
| index_q | index_k]``) writes the normed and rotated q, k, index_q and
index_k and passes v through, with the numerics of
`attention.norm_rope_heads` (the TPU ``rms_norm`` provider and the FP32
rotary embedding): RMSNorm in f32 rounded to the activation dtype, then the
rotation in f32 rounded once.

The kernel's outputs have fixed layouts, so XLA cannot re-lay out the norm
and rope for the consumers of q: as XLA fusions they emitted q tokens-minor
and every consumer paid a relayout copy (the DCP query all-gather, ~49 us
per layer at 8K tokens on the c4 twin, j-db4e365d). With ``q_head_major``
q comes out ``[nq, T, D]``, the layout the DCP attention reads. With
``rotate`` q and index_q come out rotated for a 4-bit cache
(`msa_fp4.rotate_queries`): as an XLA dot on the kernel's output the
rotation re-laid q out (c64 8K-token step: ~0.3 ms per layer of rotation
fusions and copies, twin j-6bf91132).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

DEFAULT_BLOCK_TOKENS = 1024
SUB_BLOCK_TOKENS = 256
# Double-buffered input and output blocks stay within this at any head count.
BLOCK_VMEM_BYTES = 16 * 1024 * 1024


def _rope_matrices(head_dim: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """0 / +-1 matrices applied on the MXU (exact: one nonzero term per
    output): ``[D, D]`` moves each rotated lane's NeoX partner (``x2`` for
    ``x1`` and back) into place; ``[2, rotary_dim, D]`` spread ``cos_sin``
    into the cos and signed-sin factors of each lane."""
    half = rotary_dim // 2
    partner = np.zeros((head_dim, head_dim), np.float32)
    factors = np.zeros((2, rotary_dim, head_dim), np.float32)
    for j in range(half):
        partner[j + half, j] = partner[j, j + half] = 1.0
        factors[0, j, j] = factors[0, j, j + half] = 1.0
        factors[1, half + j, j] = -1.0  # o1 = x1 cos - x2 sin
        factors[1, half + j, j + half] = 1.0  # o2 = x2 cos + x1 sin
    return partner, factors


def _kernel(
    qkv_ref,
    cos_sin_ref,
    weight_ref,
    partner_ref,
    factors_ref,
    q_ref,
    k_ref,
    v_ref,
    iq_ref,
    ik_ref,
    *,
    hadamard_ref=None,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    q_head_major,
):
    dtype = qkv_ref.dtype
    dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)
    nq, nkv, d = num_heads, num_kv_heads, head_dim

    def rows(r):
        cos_sin = cos_sin_ref[r, :].astype(jnp.bfloat16)
        cos = dot(cos_sin, factors_ref[0])
        sin = dot(cos_sin, factors_ref[1])
        # Past the rotary lanes n * 1 + partner * -0 is n, signed zeros
        # included (partner is +0 there): no select per head.
        lane = jax.lax.broadcasted_iota(jnp.int32, cos.shape, 1)
        cos = jnp.where(lane < rotary_dim, cos, 1.0)
        sin = jnp.where(lane < rotary_dim, sin, -0.0)

        def norm_rope(col, w_row):
            x = qkv_ref[r, col * d : (col + 1) * d].astype(jnp.float32)
            variance = jnp.mean(x * x, axis=1, keepdims=True)
            x = x * jax.lax.rsqrt(variance + eps) * weight_ref[pl.ds(w_row, 1), :]
            n = x.astype(dtype)
            partner = dot(n.astype(jnp.bfloat16), partner_ref[...])
            n = n.astype(jnp.float32)
            return (n * cos + partner * sin).astype(dtype)

        def rotated(x):
            # msa_fp4.rotate_queries: one bf16 pass, f32 accumulation, then 1 / D.
            if hadamard_ref is None:
                return x
            return (dot(x, hadamard_ref[...]) * (1.0 / d)).astype(dtype)

        for h in range(nq):
            if q_head_major:
                q_ref[h, r, :] = rotated(norm_rope(h, h))
            else:
                q_ref[r, h * d : (h + 1) * d] = rotated(norm_rope(h, h))
        for h in range(nkv):
            k_ref[r, h * d : (h + 1) * d] = norm_rope(nq + h, nq + h)
            v_ref[r, h * d : (h + 1) * d] = qkv_ref[
                r, (nq + nkv + h) * d : (nq + nkv + h + 1) * d
            ]
            iq_ref[r, h * d : (h + 1) * d] = rotated(
                norm_rope(nq + 2 * nkv + h, nq + nkv + h)
            )
        ik_ref[r, :] = norm_rope(nq + 3 * nkv, nq + 2 * nkv)

    # Straight-line sub-blocks: one body over a whole large block keeps too
    # many values live and spills; a loop over them serializes their latency.
    # A block that is not a multiple runs whole (at 384-416 rows that beat
    # a 256-row sub-block and a tail).
    block_t = qkv_ref.shape[0]
    sub = block_t if block_t % SUB_BLOCK_TOKENS else SUB_BLOCK_TOKENS
    for start in range(0, block_t, sub):
        rows(pl.ds(start, sub))


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "eps",
        "q_head_major",
        "rotate",
        "block_tokens",
        "interpret",
    ),
)
def qk_norm_rope(
    qkv: jax.Array,
    cos_sin: jax.Array,
    weight: jax.Array,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    q_head_major: bool = False,
    rotate: bool = False,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """``(q, k, v, index_q, index_k)`` of ``qkv [T, (nq + 3 * nkv + 1) * D]``.

    ``cos_sin [T, rotary_dim]`` holds each token's ``cos | sin`` (the rotary
    cache rows of its position, in the activation dtype); ``weight [nq + 2 *
    nkv + 1, D]`` f32 the ``1 + w`` rows of the q, k, index_q and index_k
    heads. q is ``[nq, T, D]`` with `q_head_major`, else ``[T, nq * D]``;
    k, v and index_q ``[T, nkv * D]``, index_k ``[T, D]``. `rotate`: q and
    index_q times ``H / D`` (`msa_fp4.rotate_queries`).
    """
    num_tokens = qkv.shape[0]
    nq, nkv, d = num_heads, num_kv_heads, head_dim
    if qkv.shape[1] != (nq + 3 * nkv + 1) * d or weight.shape != (nq + 2 * nkv + 1, d):
        raise ValueError(
            f"qkv {qkv.shape} / weight {weight.shape} do not hold {nq} query and "
            f"{nkv} KV heads of {d}."
        )
    if qkv.dtype != jnp.bfloat16:
        # The rope's partner lanes move on the MXU, exact for bf16 values.
        raise NotImplementedError(f"qk_norm_rope takes bf16 qkv, got {qkv.dtype}.")
    rotary_dim = cos_sin.shape[1]
    # Whole-array blocks for small batches (any row count); else multiples
    # of the 16-row bf16 tile, the last block partial.
    row_bytes = 4 * qkv.shape[1] * qkv.dtype.itemsize  # in + out, two buffers
    fit = max(1, BLOCK_VMEM_BYTES // (row_bytes * SUB_BLOCK_TOKENS)) * SUB_BLOCK_TOKENS
    block_t = max(1, min(num_tokens, block_tokens, fit))
    rows = lambda i: (i, 0)  # noqa: E731
    kv_shape = jax.ShapeDtypeStruct((num_tokens, nkv * d), qkv.dtype)
    if q_head_major:
        q_spec = pl.BlockSpec((nq, block_t, d), lambda i: (0, i, 0))
        q_shape = jax.ShapeDtypeStruct((nq, num_tokens, d), qkv.dtype)
    else:
        q_spec = pl.BlockSpec((block_t, nq * d), rows)
        q_shape = jax.ShapeDtypeStruct((num_tokens, nq * d), qkv.dtype)
    kv_spec = pl.BlockSpec((block_t, nkv * d), rows)
    mats = [jnp.asarray(m, jnp.bfloat16) for m in _rope_matrices(d, rotary_dim)]
    mat_specs = [
        pl.BlockSpec((d, d), lambda i: (0, 0)),
        pl.BlockSpec((2, rotary_dim, d), lambda i: (0, 0, 0)),
    ]
    kernel = _kernel
    if rotate:
        mats.append(jnp.asarray(msa_fp4.hadamard(d), jnp.bfloat16))
        mat_specs.append(pl.BlockSpec((d, d), lambda i: (0, 0)))

        def kernel(qkv, cs, w, partner, factors, hadamard, *outs, **kw):
            return _kernel(
                qkv, cs, w, partner, factors, *outs, hadamard_ref=hadamard, **kw
            )

    return pl.pallas_call(
        functools.partial(
            kernel,
            num_heads=nq,
            num_kv_heads=nkv,
            head_dim=d,
            rotary_dim=rotary_dim,
            eps=eps,
            q_head_major=q_head_major,
        ),
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((block_t, qkv.shape[1]), rows),
            pl.BlockSpec((block_t, rotary_dim), rows),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            *mat_specs,
        ],
        out_specs=[
            q_spec,
            kv_spec,
            kv_spec,
            kv_spec,
            pl.BlockSpec((block_t, d), rows),
        ],
        out_shape=[
            q_shape,
            kv_shape,
            kv_shape,
            kv_shape,
            jax.ShapeDtypeStruct((num_tokens, d), qkv.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_qk_norm_rope",
    )(qkv, cos_sin, weight.astype(jnp.float32), *mats)
