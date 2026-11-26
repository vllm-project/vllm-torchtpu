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

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_block,
    xla_quantized_matmul,
)

_LANE_ALIGN = 128
_SUBLANE_ALIGN = 8
_BM_CANDIDATES = (1024, 512, 256, 128, 64)
_BN_CANDIDATES = (1536, 1024, 512, 256, 128)
_BPS_CANDIDATES = (16, 8)
_DOT_DIMS = (((1,), (0,)), ((), ()))


def silu_and_mul(x: jax.Array) -> jax.Array:
    d = x.shape[-1] // 2
    return jax.nn.silu(x[..., :d]) * x[..., d:]


def _select_pallas_tile(
    m: int, k: int, n: int, blocks: int, w2_dtype: jnp.dtype
) -> tuple[int, int, int] | None:
    if (
        w2_dtype != jnp.float8_e4m3fn
        or blocks < _SUBLANE_ALIGN
        or blocks % _SUBLANE_ALIGN != 0
        or k % blocks != 0
    ):
        return None
    kb = k // blocks
    if kb < _LANE_ALIGN or kb % _LANE_ALIGN != 0:
        return None
    bm = next((c for c in _BM_CANDIDATES if c <= m and m % c == 0), None)
    bn = next((c for c in _BN_CANDIDATES if c <= n and n % c == 0), None)
    bps = next((c for c in _BPS_CANDIDATES if blocks % c == 0), None)
    if bm is None or bn is None or bps is None:
        return None
    # Avoid CompileTimeScopedVmemOom and slower multi-step tiles on wide dense
    # k-chunks (e.g. k=12288, kb=768): XLA is already MXU-saturated past 4 MiB.
    if bps * kb * bn > 4 * 1024 * 1024:
        bps = _SUBLANE_ALIGN
    kc = bps * kb
    vmem_bytes = 2 * (bm * kc + kc * bn + 2 * bm * bn) + (
        4 * bm * bn if blocks > bps else 0
    )
    if kc * bn > 4 * 1024 * 1024 or vmem_bytes > 24 * 1024 * 1024:
        return None
    return bm, bn, bps


def _blocked_down_single_step_kernel(
    aq_ref,
    as_ref,
    w2_ref,
    s2_ref,
    out_ref,
    *,
    kb: int,
    bps: int,
):
    aq = aq_ref[...]
    a_scales = as_ref[...]
    w2 = w2_ref[...]
    s2 = s2_ref[...]
    acc = jnp.zeros(out_ref.shape, jnp.float32)
    for j in range(bps):
        partial = lax.dot_general(
            aq[:, j * kb : (j + 1) * kb],
            w2[j * kb : (j + 1) * kb, :],
            _DOT_DIMS,
            preferred_element_type=jnp.float32,
        )
        acc = acc + partial * a_scales[:, j : j + 1] * s2[j : j + 1, :]
    out_ref[...] = acc.astype(out_ref.dtype)


def _blocked_down_multi_step_kernel(
    aq_ref,
    as_ref,
    w2_ref,
    s2_ref,
    out_ref,
    acc_ref,
    *,
    kb: int,
    bps: int,
    n_steps: int,
):
    step = pl.program_id(2)
    aq = aq_ref[...]
    a_scales = as_ref[...]
    w2 = w2_ref[...]
    s2 = s2_ref[...]
    acc = jnp.where(step == 0, jnp.zeros(acc_ref.shape, jnp.float32), acc_ref[...])
    for j in range(bps):
        partial = lax.dot_general(
            aq[:, j * kb : (j + 1) * kb],
            w2[j * kb : (j + 1) * kb, :],
            _DOT_DIMS,
            preferred_element_type=jnp.float32,
        )
        acc = acc + partial * a_scales[:, j : j + 1] * s2[j : j + 1, :]
    acc_ref[...] = acc

    @pl.when(step == n_steps - 1)
    def _store():
        out_ref[...] = acc.astype(out_ref.dtype)


def _pallas_blocked_down(
    act: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    *,
    bm: int,
    bn: int,
    bps: int,
    interpret: bool = False,
) -> jax.Array:
    m, k = act.shape
    blocks, n = s2.shape
    kb = k // blocks
    n_steps = blocks // bps
    kc = bps * kb

    act_q, a_scale = quantize_block(
        act.reshape(m, blocks, kb), axis=-1, target_dtype=w2.dtype
    )
    a_q = act_q.reshape(m, k)
    s2_f32 = s2.astype(jnp.float32)
    # Avoid 1-col VMEM Ref slicing: pad per-step activation scales to 128 lanes.
    a_scale_steps = jnp.pad(
        a_scale[..., 0].astype(jnp.float32).reshape(m, n_steps, bps).transpose(1, 0, 2),
        ((0, 0), (0, 0), (0, _LANE_ALIGN - bps)),
    )

    if n_steps == 1:
        return pl.pallas_call(
            functools.partial(_blocked_down_single_step_kernel, kb=kb, bps=bps),
            out_shape=jax.ShapeDtypeStruct((m, n), act.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bm, kc), lambda mi, ni: (mi, 0)),
                    pl.BlockSpec((bm, _LANE_ALIGN), lambda mi, ni: (mi, 0)),
                    pl.BlockSpec((kc, bn), lambda mi, ni: (0, ni)),
                    pl.BlockSpec((bps, bn), lambda mi, ni: (0, ni)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda mi, ni: (mi, ni)),
                grid=(m // bm, n // bn),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name=f"shared_expert_blocked_down_bm{bm}_bn{bn}_b{blocks}",
        )(a_q, a_scale_steps[0], w2, s2_f32)

    return pl.pallas_call(
        functools.partial(
            _blocked_down_multi_step_kernel, kb=kb, bps=bps, n_steps=n_steps
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), act.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, kc), lambda mi, ni, s: (mi, s)),
                pl.BlockSpec((None, bm, _LANE_ALIGN), lambda mi, ni, s: (s, mi, 0)),
                pl.BlockSpec((kc, bn), lambda mi, ni, s: (s, ni)),
                pl.BlockSpec((bps, bn), lambda mi, ni, s: (s, ni)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda mi, ni, s: (mi, ni)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, n_steps),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"shared_expert_blocked_down_bm{bm}_bn{bn}_b{blocks}_s{n_steps}",
    )(a_q, a_scale_steps, w2, s2_f32)


def _xla_blocked_down(act: jax.Array, w2: jax.Array, s2: jax.Array) -> jax.Array:
    blocks, n = s2.shape
    kb = w2.shape[0] // blocks
    out = jnp.zeros((act.shape[0], n), jnp.float32)
    for b in range(blocks):
        rows = slice(b * kb, (b + 1) * kb)
        out += xla_quantized_matmul(act[:, rows], w2[rows], s2[b]).astype(jnp.float32)
    return out.astype(act.dtype)


def blocked_down(
    act: jax.Array, w2: jax.Array, s2: jax.Array, *, interpret: bool = False
) -> jax.Array:
    """Contract `act [m, k] @ w2 [k, n]` with per-k-block `s2 [blocks, n]` in f32."""
    if interpret or jax.default_backend() == "tpu":
        tile = _select_pallas_tile(
            act.shape[0], w2.shape[0], s2.shape[1], s2.shape[0], w2.dtype
        )
        if tile is not None:
            bm, bn, bps = tile
            return _pallas_blocked_down(
                act, w2, s2, bm=bm, bn=bn, bps=bps, interpret=interpret
            )
    return _xla_blocked_down(act, w2, s2)


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
    if group <= 0 or n % group:
        raise ValueError(f"{n} ranks do not split into groups of {group}")
    groups = tuple(tuple(range(g, g + group)) for g in range(0, n, group))
    return jax.shard_map(
        lambda *a: grouped_shared_expert(*a, axis=axis, groups=groups),
        mesh=mesh,
        in_specs=(P(axis),) * 5,
        out_specs=P(axis),
        check_vma=False,
    )
