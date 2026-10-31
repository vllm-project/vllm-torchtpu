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
the arithmetic of the TP shard it came from.
"""

import jax
import jax.numpy as jnp
from jax import lax
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


def grouped_shared_expert(
    x: jax.Array,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    *,
    axis: str,
    groups: tuple[tuple[int, ...], ...],
) -> jax.Array:
    """Per shard (inside `shard_map` over `axis`): `x [t, H]` own rows,
    `w13 [H, 2c]` this member's `[gate | up]` channels with per-channel `s13
    [2c]`, `w2 [c, H]` with `s2 [blocks, H]`. Returns `[t, H]`, the whole
    expert on the own rows. `groups` lists the members of each group in the
    order their rows are gathered."""
    if len(groups[0]) > 1:
        rows = lax.all_gather(x, axis, axis=0, tiled=True, axis_index_groups=groups)
    else:
        rows = x
    act = silu_and_mul(xla_quantized_matmul(rows, w13, s13))
    out = blocked_down(act, w2, s2)
    if len(groups[0]) > 1:
        out = lax.psum_scatter(
            out, axis, scatter_dimension=0, tiled=True, axis_index_groups=groups
        )
    return out


def grouped_shared_expert_sharded(mesh, axis: str, group: int):
    """`grouped_shared_expert` on global arrays sharded on `axis` (rows of `x`
    and each rank's weight slice), with groups of `group` consecutive mesh
    indices."""
    n = mesh.shape[axis]
    if n % group:
        raise ValueError(f"{n} ranks do not split into groups of {group}")
    groups = tuple(tuple(range(g, g + group)) for g in range(0, n, group))
    return jax.shard_map(
        lambda *a: grouped_shared_expert(*a, axis=axis, groups=groups),
        mesh=mesh,
        in_specs=(P(axis),) * 5,
        out_specs=P(axis),
        check_vma=False,
    )
