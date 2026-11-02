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
"""A shared expert over row-sharded tokens, tensor-parallel within small
groups of ranks.

Each rank holds its own rows and `1/g` of the expert's intermediate channels.
The group all-gathers its rows, every member runs its channel slice on all of
them, and a reduce-scatter hands each member the full sum for its own rows.
The weight cost per rank is `1/g` of the expert instead of all of it; the
traffic is `2 * (g - 1)` row blocks per rank, within the group.

The down projection arrives as `blocks` k-slices, each with its own
per-channel scale (a TP shard's quantization, kept as it was); each slice is
contracted on its own, quantizing its activation columns per row, which is
the arithmetic of the TP shard it came from. `blocked_down` writes every
slice's partial to HBM before summing them (one small matmul per slice).
`blocked_down_fused` instead dequantizes each weight slice with its scales
to the activation dtype in VMEM and contracts the unquantized activation
over the whole `k` at once: no per-slice partials, and no fp8 rounding of
the activation (so not the same values: closer to the unquantized product).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul


def silu_and_mul(x: jax.Array) -> jax.Array:
    d = x.shape[-1] // 2
    return jax.nn.silu(x[..., :d]) * x[..., d:]


def blocked_down(act: jax.Array, w2: jax.Array, s2: jax.Array) -> jax.Array:
    """`act [m, k] @ w2 [k, n]` with `s2 [blocks, n]` per k-block scales;
    each block's partial is summed in f32. Returns `act.dtype`."""
    blocks, n = s2.shape
    kb = w2.shape[0] // blocks
    out = jnp.zeros((act.shape[0], n), jnp.float32)
    for b in range(blocks):
        rows = slice(b * kb, (b + 1) * kb)
        out += xla_quantized_matmul(act[:, rows], w2[rows], s2[b]).astype(jnp.float32)
    return out.astype(act.dtype)


def _blocked_down_kernel(act_ref, w_ref, s_ref, out_ref, *, blocks, kb):
    # Each k-slice of the weight tile times its per-channel scales (in f32),
    # rounded to the activation dtype; then one f32-accumulated dot over k.
    w = jnp.concatenate(
        [
            (
                w_ref[b * kb : (b + 1) * kb, :].astype(jnp.float32)
                * s_ref[b : b + 1, :]
            ).astype(act_ref.dtype)
            for b in range(blocks)
        ],
        axis=0,
    )
    acc = jnp.dot(act_ref[...], w, preferred_element_type=jnp.float32)
    out_ref[...] = acc.astype(out_ref.dtype)


def blocked_down_fused(
    act: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    *,
    block_m: int = 512,
    block_n: int = 512,
    interpret: bool = False,
) -> jax.Array:
    """`act [m, k] @ (w2 [k, n] scaled per k-block by s2 [blocks, n])` as
    one kernel: each output tile (at most `block_m` rows by `block_n`
    columns) dequantizes its weight columns to `act.dtype` and runs a single
    dot over `k`, accumulated in f32. The activation is not quantized. Rows
    split into equal tiles of a multiple of 32, the last padded with zero rows
    (sliced off) when `m` does not divide."""
    blocks, n = s2.shape
    m, k = act.shape
    kb = k // blocks
    tn = min(n, block_n)
    if kb * blocks != k or w2.shape != (k, n) or n % tn:
        raise ValueError(
            f"blocked_down_fused: act {act.shape}, w2 {w2.shape}, s2 {s2.shape}, "
            f"block_n {block_n}"
        )
    if m == 0:
        return jnp.zeros((0, n), act.dtype)
    tiles = pl.cdiv(m, block_m)
    tm = m if tiles == 1 else pl.cdiv(pl.cdiv(m, tiles), 32) * 32
    pad = tiles * tm - m
    if pad:
        act = jnp.pad(act, ((0, pad), (0, 0)))
    out = pl.pallas_call(
        functools.partial(_blocked_down_kernel, blocks=blocks, kb=kb),
        grid=(tiles, n // tn),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, tn), lambda i, j: (0, j)),
            pl.BlockSpec((blocks, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((tiles * tm, n), act.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="shared_expert_blocked_down",
    )(act, w2, s2.astype(jnp.float32))
    return out[:m] if pad else out


def grouped_shared_expert(
    x: jax.Array,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    *,
    axis: str,
    groups: tuple[tuple[int, ...], ...],
    fused_down: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Per shard (inside `shard_map` over `axis`): `x [t, H]` own rows,
    `w13 [H, 2c]` this member's `[gate | up]` channels with per-channel `s13
    [2c]`, `w2 [c, H]` with `s2 [blocks, H]`. Returns `[t, H]`, the whole
    expert on the own rows. `groups` lists the members of each group in the
    order their rows are gathered. `fused_down` runs the down projection as
    `blocked_down_fused` (an unquantized activation, so not the same values;
    `interpret` runs it on the host)."""
    if len(groups[0]) > 1:
        rows = lax.all_gather(x, axis, axis=0, tiled=True, axis_index_groups=groups)
    else:
        rows = x
    act = silu_and_mul(xla_quantized_matmul(rows, w13, s13))
    if fused_down:
        out = blocked_down_fused(act, w2, s2, interpret=interpret)
    else:
        out = blocked_down(act, w2, s2)
    if len(groups[0]) > 1:
        out = lax.psum_scatter(
            out, axis, scatter_dimension=0, tiled=True, axis_index_groups=groups
        )
    return out


def grouped_shared_expert_sharded(
    mesh, axis: str, group: int, *, fused_down: bool = False, interpret: bool = False
):
    """`grouped_shared_expert` on global arrays sharded on `axis` (rows of `x`
    and each rank's weight slice), with groups of `group` consecutive mesh
    indices."""
    n = mesh.shape[axis]
    if n % group:
        raise ValueError(f"{n} ranks do not split into groups of {group}")
    groups = tuple(tuple(range(g, g + group)) for g in range(0, n, group))
    body = functools.partial(
        grouped_shared_expert,
        axis=axis,
        groups=groups,
        fused_down=fused_down,
        interpret=interpret,
    )
    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P(axis),) * 5,
        out_specs=P(axis),
        check_vma=False,
    )
