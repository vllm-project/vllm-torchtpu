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
"""TensorCore MoE combine by routed-row pairs (TP-sliced experts, large batches).

out[t] = sum_j w[t, j] * x[pos[t, j]], with x the expert-sorted GMM output.

With every expert local (TP-sliced experts) a block of tokens touches every
expert's range for a few rows each, so the 16-row tiles tc_combine loads are
mostly other tokens' rows. This kernel instead DMAs each route's own row,
as the bf16 row pair that holds it (packed bf16 rows pair up in one sublane,
the smallest DMA unit), into a token-major buffer, and an MXU product per
32-token group with a [32, 64] weight matrix (one nonzero per token row per
route: the row's half of its pair) sums the routes. It moves 2x the routed
bytes, with no tile rounding.

A non-finite routed row reaches the token that shares its pair through a zero
weight, as tc_combine's tiles do; rows the GMM did not write must be zeroed by
the caller (gmm_v2 zero_initialize).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TOKEN_BLOCK = 64
# Tokens per MXU product: their pairs are a [64, hidden] slab.
_GROUP = 32
# DMA issues per loop iteration (whole tokens): overlaps their address arithmetic.
_UNROLL = 8
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


# Pair indices are scalar-prefetched (two int32 per route): 256 KiB of SMEM at
# the 8192-token top-4 chunk, the largest measured.
_MAX_ROUTES = 32768


def supported(
    num_tokens: int, num_rows: int, hidden: int, topk: int, dtype, weights_dtype
) -> bool:
    if dtype != jnp.bfloat16 or weights_dtype != jnp.bfloat16:
        return False
    if hidden % 128 or num_tokens % TOKEN_BLOCK or num_tokens * topk > _MAX_ROUTES:
        return False
    if num_rows % 16 or _UNROLL % topk:
        return False
    buf = 2 * topk * 2 * TOKEN_BLOCK * hidden * 2
    out = 2 * TOKEN_BLOCK * hidden * 2
    return buf + out <= _VMEM_LIMIT_BYTES - 8 * 1024 * 1024


def _kernel(
    tile_ref,  # SMEM [T * k]: each route's 16-row tile of x, token-major
    pair_ref,  # SMEM [T * k]: the route's row pair within that tile
    par_ref,  # VMEM [TB, k] int32: the route's row within its pair
    w_ref,  # VMEM [TB, k]
    x_hbm,  # ANY [R / 16, 16, H]
    out_ref,  # VMEM [TB, H]
    buf,  # VMEM [2, k, 2 * TB, H]
    sems,  # DMA [2]
    *,
    topk: int,
):
    b = pl.program_id(0)
    nb = pl.num_programs(0)
    tb = out_ref.shape[0]

    def copy(block, t, j, slot):
        # Route j of token t lands at buf[slot, j, 2t:2t+2].
        r = (block * tb + t) * topk + j
        return pltpu.make_async_copy(
            x_hbm.at[tile_ref[r], pl.ds(pl.multiple_of(2 * pair_ref[r], 2), 2)],
            buf.at[slot, j, pl.ds(pl.multiple_of(2 * t, 2), 2)],
            sems.at[slot],
        )

    def issue(block):
        slot = block % 2

        def body(i, c):
            for u in range(_UNROLL // topk):
                for j in range(topk):
                    copy(block, i * (_UNROLL // topk) + u, j, slot).start()
            return c

        lax.fori_loop(0, tb * topk // _UNROLL, body, 0)

    @pl.when(b == 0)
    def _():
        issue(0)

    @pl.when(b + 1 < nb)
    def _():
        issue(b + 1)

    slot = b % 2

    def wait(i, c):
        for _ in range(_UNROLL):
            copy(b, 0, 0, slot).wait()
        return c

    lax.fori_loop(0, tb * topk // _UNROLL, wait, 0)

    par = par_ref[...]
    w = w_ref[...].astype(jnp.float32)
    cols = lax.broadcasted_iota(jnp.int32, (_GROUP, 2 * _GROUP), 1)
    rows = lax.broadcasted_iota(jnp.int32, (_GROUP, 2 * _GROUP), 0)
    for g in range(tb // _GROUP):
        acc = jnp.zeros((_GROUP, out_ref.shape[1]), jnp.float32)
        for j in range(topk):
            sel = 2 * rows + par[g * _GROUP : (g + 1) * _GROUP, j : j + 1]
            p = jnp.where(cols == sel, w[g * _GROUP : (g + 1) * _GROUP, j : j + 1], 0)
            start = g * 2 * _GROUP
            acc += jnp.dot(
                p.astype(buf.dtype),
                buf[slot, j, start : start + 2 * _GROUP, :],
                preferred_element_type=jnp.float32,
            )
        out_ref[g * _GROUP : (g + 1) * _GROUP, :] = acc.astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("topk", "interpret"))
def pair_combine(
    x: jax.Array,
    pos: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
    *,
    topk: int,
    interpret: bool = False,
) -> jax.Array:
    """out[t] = sum_j where(valid, weights * x[pos], 0) over j < topk, f32
    accumulation, bf16 out. `pos`, `weights` (bf16) and `valid` are flat
    [T * topk], token-major; `x` rows past the GMM's valid prefix must be
    finite. Callers check `supported` first."""
    num_tokens = pos.shape[0] // topk
    hidden = x.shape[1]
    pos = jnp.where(valid, pos, 0).astype(jnp.int32)
    w = jnp.where(valid, weights, 0).reshape(num_tokens, topk)
    tb = TOKEN_BLOCK
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(num_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, topk), lambda b, *_: (b, 0)),
            pl.BlockSpec((tb, topk), lambda b, *_: (b, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((tb, hidden), lambda b, *_: (b, 0)),
        scratch_shapes=[
            pltpu.VMEM((2, topk, 2 * tb, hidden), x.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    return pl.pallas_call(
        functools.partial(_kernel, topk=topk),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
            # Every pair index is an expert-sorted row of x: in range.
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="pair_moe_combine",
    )(
        pos // 16,
        (pos % 16) // 2,
        (pos % 2).reshape(num_tokens, topk),
        w,
        # Whole bf16 tiles of 16 rows: a bitcast of x's tiled layout.
        x.reshape(x.shape[0] // 16, 16, hidden),
    )
