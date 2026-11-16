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
"""Fused per-head QK RMSNorm + partial half-split 3D RoPE Pallas kernel.

Used for MiniMax-H3 DiT self-attention (``head_dim=128``, ``rotary_dim=96``,
``interleaved=False``), where ``rope_table`` has shape ``[T, rotary_dim]``
storing ``[cos(theta) | sin(theta)]`` with ``theta`` of width
``rotary_dim // 2 = 48``.

The half-split rotation ``(d, d + 48)`` is expressed on the natural
``(block_rows, heads, 128)`` layout as::

    qn = bf16(q * rsqrt(mean(q ^ 2) + eps) * qw)
    out_q = qn * cos_pad + (qn @ P_rot) * sin_pad

where ``P_rot`` is the exact signed permutation matrix mapping
``[x_0..47, x_48..95, x_96..127] -> [-x_48..95, x_0..47, 0]`` on the MXU,
avoiding strided slice/cat relayouts on TPU while matching the intermediate
BF16 rounding of ``vllm_omni.diffusion.layers.fused_qk_norm_rope``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.bfloat16
F32 = jnp.float32
SUBLANES = 8


@functools.cache
def rot_perm_matrix(head_dim: int = 128, rot_dim: int = 96) -> np.ndarray:
    """Signed permutation ``P_rot [D, D]`` mapping ``[x_0..h-1, x_h..2h-1, rest]``
    to ``[-x_h..2h-1, x_0..h-1, 0]``."""
    half = rot_dim // 2
    p = np.zeros((head_dim, head_dim), dtype=np.float32)
    for j in range(half):
        p[half + j, j] = -1.0
        p[j, half + j] = 1.0
    return p


def _qknorm_rope_kernel(
    q_ref: jax.Ref,
    k_ref: jax.Ref,
    qw_ref: jax.Ref,
    kw_ref: jax.Ref,
    cos_ref: jax.Ref,
    sin_ref: jax.Ref,
    prot_ref: jax.Ref,
    qo_ref: jax.Ref,
    ko_ref: jax.Ref,
    *,
    eps: float,
) -> None:
    q = q_ref[...].astype(F32)
    k = k_ref[...].astype(F32)
    qw = qw_ref[...][:1, None, :]
    kw = kw_ref[...][:1, None, :]
    cos = cos_ref[...][:, None, :]
    sin = sin_ref[...][:, None, :]
    prot = prot_ref[...]

    # Match vllm-omni's fused_qk_norm_rope rounding: RMSNorm output rounds to
    # BF16 before FP32 rotary multiplication and final BF16 store.
    qn = (
        (q * jax.lax.rsqrt(jnp.mean(q * q, axis=-1, keepdims=True) + eps) * qw)
        .astype(BF16)
        .astype(F32)
    )
    kn = (
        (k * jax.lax.rsqrt(jnp.mean(k * k, axis=-1, keepdims=True) + eps) * kw)
        .astype(BF16)
        .astype(F32)
    )

    qr = jax.lax.dot_general(
        qn,
        prot,
        (((2,), (0,)), ((), ())),
        preferred_element_type=F32,
        precision=jax.lax.Precision.HIGHEST,
    )
    kr = jax.lax.dot_general(
        kn,
        prot,
        (((2,), (0,)), ((), ())),
        preferred_element_type=F32,
        precision=jax.lax.Precision.HIGHEST,
    )
    qo_ref[...] = (qn * cos + qr * sin).astype(qo_ref.dtype)
    ko_ref[...] = (kn * cos + kr * sin).astype(ko_ref.dtype)


@functools.partial(jax.jit, static_argnames=("rot_dim", "eps", "interpret"))
def fused_minimax_qk_norm_rope(
    q: jax.Array,
    k: jax.Array,
    qw: jax.Array,
    kw: jax.Array,
    rope_table: jax.Array,
    *,
    rot_dim: int = 96,
    eps: float = 1e-6,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused per-head FP32 RMSNorm + partial half-split RoPE on ``q, k [T, H, 128]``."""
    r0, hq0, d = q.shape
    _, hk0, _ = k.shape
    br = 32
    while r0 % br and br > SUBLANES:
        br //= 2
    r_pad = -(-r0 // br) * br
    hq_pad = -(-hq0 // SUBLANES) * SUBLANES
    hk_pad = -(-hk0 // SUBLANES) * SUBLANES

    qp = (
        jnp.pad(q, ((0, r_pad - r0), (0, hq_pad - hq0), (0, 0)))
        if (r_pad != r0 or hq_pad != hq0)
        else q
    )
    kp = (
        jnp.pad(k, ((0, r_pad - r0), (0, hk_pad - hk0), (0, 0)))
        if (r_pad != r0 or hk_pad != hk0)
        else k
    )

    half = rot_dim // 2
    rt = rope_table.astype(F32)
    cos_h = rt[:, :half]
    sin_h = rt[:, half:rot_dim]
    pad_d = d - rot_dim
    cos_pad = jnp.concatenate([cos_h, cos_h, jnp.ones((r0, pad_d), dtype=F32)], axis=-1)
    sin_pad = jnp.concatenate(
        [sin_h, sin_h, jnp.zeros((r0, pad_d), dtype=F32)], axis=-1
    )
    if r_pad != r0:
        cos_pad = jnp.pad(cos_pad, ((0, r_pad - r0), (0, 0)), constant_values=1.0)
        sin_pad = jnp.pad(sin_pad, ((0, r_pad - r0), (0, 0)), constant_values=0.0)

    qwb = jnp.broadcast_to(qw.astype(F32)[None, :], (SUBLANES, d))
    kwb = jnp.broadcast_to(kw.astype(F32)[None, :], (SUBLANES, d))
    prot = jnp.asarray(rot_perm_matrix(d, rot_dim))

    q_spec = pl.BlockSpec((br, hq_pad, d), lambda i: (i, 0, 0))
    k_spec = pl.BlockSpec((br, hk_pad, d), lambda i: (i, 0, 0))
    w_spec = pl.BlockSpec((SUBLANES, d), lambda i: (0, 0))
    cs_spec = pl.BlockSpec((br, d), lambda i: (i, 0))
    p_spec = pl.BlockSpec((d, d), lambda i: (0, 0))

    qo, ko = pl.pallas_call(
        functools.partial(_qknorm_rope_kernel, eps=eps),
        grid=(r_pad // br,),
        in_specs=[q_spec, k_spec, w_spec, w_spec, cs_spec, cs_spec, p_spec],
        out_specs=[q_spec, k_spec],
        out_shape=[
            jax.ShapeDtypeStruct((r_pad, hq_pad, d), q.dtype),
            jax.ShapeDtypeStruct((r_pad, hk_pad, d), k.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        interpret=interpret,
        name="minimax_h3_fused_qknorm_rope",
    )(qp, kp, qwb, kwb, cos_pad, sin_pad, prot)
    return qo[:r0, :hq0], ko[:r0, :hk0]
