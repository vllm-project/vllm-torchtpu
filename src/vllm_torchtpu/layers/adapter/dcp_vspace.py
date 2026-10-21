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
"""TPU_DCP_MERGE_VSPACE_FUSED: the V-space DCP merge operand of
#1935 trial 6 (TPU_DCP_MERGE_VSPACE), ported onto TPU_DCP_MLA_UV_BEFORE_MERGE.

`vspace_weighted_heads_major` builds the chunked merge's reduce-scatter
operand in V space in one XLA computation: a head-batched matmul of this
rank's latent partial with W_UV for every head of the DCP group, written
heads-major, times the merge weight, `where`-masked and rounded once. The
pre-merge `project_uv_heads` + weighted heads-major operand it replaces run
as separate ops with relayouts in between.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def vspace_weighted_heads_major_jax(
    o: jax.Array,
    w: jax.Array,
    w_uv: jax.Array,
    w_uv_scale: jax.Array | None = None,
    out_dtype=jnp.bfloat16,
) -> jax.Array:
    """`o` [T, H, C] (this rank's latent partial over every head of the DCP
    group), `w` f32[T, H] (its merge weights), `w_uv` [H, C, V] (W_UV over
    the same H heads, in the same order), `w_uv_scale` optional [H, 1, V] ->
    `[H, T, V]` = `where(w > 0, (o_h @ W_UV_h) * w, 0)`, heads-major, rounded
    once to `out_dtype`.

    The projection runs on the MXU with o's dtype operands and f32
    accumulation; the weight multiplies the f32 product, so the only rounding
    before the collective is the final cast. A NaN partial (zero weight)
    only reaches its own (token, head) row of the product, which the where
    zeroes."""
    out_dtype = jnp.dtype(out_dtype)
    t, h, c = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    if w_uv.shape[:2] != (h, c):
        raise ValueError(f"w_uv {w_uv.shape} must be [{h}, {c}, V]")
    v = jax.lax.dot_general(
        o,
        w_uv.astype(o.dtype),
        (((2,), (1,)), ((1,), (0,))),
        preferred_element_type=jnp.float32,
    )  # [H, T, V]
    if w_uv_scale is not None:
        v = v * w_uv_scale.astype(jnp.float32).reshape(h, 1, -1)
    wt = jnp.transpose(w.astype(jnp.float32), (1, 0))[:, :, None]  # [H, T, 1]
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    return jnp.where(wt > 0, v * wt, jnp.zeros_like(v)).astype(out_dtype)


# (out dtype name, scaled W_UV) -> torch op; registered at import, never in a
# trace.
_OPS: dict = {}


def _register(out_name: str, scaled: bool) -> None:
    out_dtype = jnp.dtype(out_name)
    t_dtype = getattr(torch, out_name)

    if scaled:

        def fn(
            o: jax.Array, w: jax.Array, w_uv: jax.Array, scale: jax.Array
        ) -> jax.Array:
            return vspace_weighted_heads_major_jax(
                o, w, w_uv, scale, out_dtype=out_dtype
            )

        def fake(o, w, w_uv, scale):
            t, h, _ = o.shape
            return torch.empty((h, t, w_uv.shape[-1]), dtype=t_dtype, device=o.device)

    else:

        def fn(o: jax.Array, w: jax.Array, w_uv: jax.Array) -> jax.Array:
            return vspace_weighted_heads_major_jax(
                o, w, w_uv, None, out_dtype=out_dtype
            )

        def fake(o, w, w_uv):
            t, h, _ = o.shape
            return torch.empty((h, t, w_uv.shape[-1]), dtype=t_dtype, device=o.device)

    name = f"pallas::dcp_vspace_weighted_hm_{out_name}" + ("_scaled" if scaled else "")
    op = pallas.jax_op(name, fn)
    op.register_fake(fake)
    _OPS[(out_name, scaled)] = op


for _name in ("bfloat16", "float32"):
    for _scaled in (False, True):
        _register(_name, _scaled)


def vspace_weighted_heads_major(
    output: torch.Tensor,
    weight: torch.Tensor,
    w_uv: torch.Tensor,
    w_uv_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """[T, H, C], f32[T, H], W_UV [H, C, V] (+ scale) -> [H, T, V] in
    `out_dtype` (bf16 or f32): `vspace_weighted_heads_major_jax` as a torch
    op."""
    name = str(out_dtype).removeprefix("torch.")
    op = _OPS[(name, w_uv_scale is not None)]
    args = [
        output.contiguous(),
        weight.to(torch.float32).contiguous(),
        w_uv.contiguous(),
    ]
    if w_uv_scale is not None:
        args.append(w_uv_scale.contiguous())
    return op(*args)
