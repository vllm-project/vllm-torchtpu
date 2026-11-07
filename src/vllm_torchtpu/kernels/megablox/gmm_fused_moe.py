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

"""Both routed expert matmuls of a MoE layer in one Pallas kernel with one
weight stream.

gmm_v2 runs gate·up (with the fused activation) and down as two calls. Each call fills
its weight pipeline from empty and drains it at its end, and the activation rows go to
HBM between them. This kernel walks the experts in order and, for each expert, streams
its gate·up weight tiles and then its down tile through one rotating set of VMEM slots,
so the next tile's copy is always in flight across the gate·up → down boundary and
across experts. The activation rows of an expert stay in VMEM between its two matmuls.

Every tile computes what gmm_v2 computes for its rows: the same per-row fp8 quantize of
the left side in 512-column blocks, the same dot per MXU column block, the same
accumulation, scales, activation, row mask and output dtype. A row's result does not
depend on how many rows share its tile, so the outputs equal gmm_v2's bit for bit on
every routed row.

An expert with more rows than fit in VMEM is split into segments of `rows_cap` rows;
each segment streams the expert's weights once.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .gmm_v2 import (
    FusedWeightsRef,
    GmmConfigs,
    TileSizes,
    WeightsRef,
    _get_lhs_sublane_size,
    align_to,
    apply_act_fn,
    make_gmm_configs,
)


@dataclasses.dataclass(frozen=True)
class FusedMoeConfigs:
    gate_up: GmmConfigs
    down: GmmConfigs
    size_m: int
    hidden: int
    inter: int
    num_groups: int
    sublane: int
    tile_n: int
    buckets: tuple[int, ...]
    rows_cap: int

    @property
    def chunk(self) -> int:
        return self.buckets[-1]

    @property
    def gate_up_tiles(self) -> int:
        return self.inter // self.tile_n

    @property
    def tiles_per_segment(self) -> int:
        return self.gate_up_tiles + 1

    @property
    def max_segments(self) -> int:
        return self.num_groups + pl.cdiv(self.size_m, self.rows_cap)


def quantized_tile_dot(tiled_lhs, rhs_ref, bucket_m: int, cfgs: GmmConfigs):
    """gmm_v2 inner_kernel's quantized matmul of one tile whose k grid has one step."""
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size
    tiled_rhs = rhs_ref.get_weight()
    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    acc_dtype = cfgs.acc_dtype
    lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
    q_block_size = cfgs.lhs_cfgs.quant_block_size
    assert lhs_q_dtype is not None and q_block_size is not None and rhs_qbs is not None

    if jnp.issubdtype(lhs_q_dtype, jnp.floating):
        dtype_max = float(jnp.finfo(lhs_q_dtype).max)
        preferred_element_type = jnp.float32
    else:
        dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
        preferred_element_type = jnp.int32

    acc_list = []
    for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_dtype)
        for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
            end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

            block_lhs = tiled_lhs[:, start_k:end_k]
            block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

            block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
            block_scale = block_abs_max / dtype_max
            block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
            block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

            if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
                block_rhs = block_rhs.astype(lhs_q_dtype)

            block_acc = jnp.matmul(
                block_lhs_q,
                block_rhs,
                preferred_element_type=preferred_element_type,
            ).astype(acc_dtype)

            block_acc *= block_scale.astype(acc_dtype)

            b_id = start_k // rhs_qbs
            rhs_scale_replicated = rhs_ref.get_scale(replicate_size=bucket_m)[
                b_id, :, start_n : start_n + col_size
            ]
            block_acc *= rhs_scale_replicated.astype(acc_dtype)

            acc_n += block_acc
        acc_list.append(acc_n)
    return jnp.concatenate(acc_list, axis=1)


def masked_rows(acc, first_row, lo, hi, out_dtype):
    rows = first_row + lax.broadcasted_iota(jnp.int32, acc.shape, 0)
    mask = jnp.logical_and(lo <= rows, rows < hi)
    return jnp.where(mask, acc, 0).astype(out_dtype)


def plan_segments(
    group_sizes_ref, group_offset_ref, seg_group, seg_base, seg_lo, seg_hi, cfg
):
    """Splits each routed group's rows into segments of at most rows_cap rows from
    its first sublane."""
    group_offset = group_offset_ref[0]
    sub = cfg.sublane

    def group_loop(lhs_group_id, carry):
        num_seg, start = carry
        group_id = lhs_group_id - group_offset
        size = group_sizes_ref[lhs_group_id]
        end = start + size
        first_base = start - start % sub
        count = pl.cdiv(end - first_base, cfg.rows_cap)
        count = jnp.where(jnp.logical_and(size > 0, group_id >= 0), count, 0)

        def segment_loop(i, n):
            base = first_base + i * cfg.rows_cap
            seg_group[n] = group_id
            seg_base[n] = base
            seg_lo[n] = jnp.maximum(start, base)
            seg_hi[n] = jnp.minimum(end, base + cfg.rows_cap)
            return n + 1

        num_seg = lax.fori_loop(0, count, segment_loop, num_seg)
        return num_seg, end

    num_seg, _ = lax.fori_loop(
        0, group_offset + cfg.num_groups, group_loop, (jnp.int32(0), jnp.int32(0))
    )
    return num_seg


def kernel(
    group_sizes_ref,
    group_offset_ref,
    x_hbm,
    w1_hbm,
    s1_hbm,
    w2_hbm,
    s2_hbm,
    y_hbm,
    gate_up_w,
    gate_up_s,
    down_w,
    down_s,
    x_buf,
    h_buf,
    y_buf,
    partial_ref,
    seg_group,
    seg_base,
    seg_lo,
    seg_hi,
    state,
    weight_sems,
    x_sems,
    y_sem,
    *,
    cfg: FusedMoeConfigs,
):
    sub = cfg.sublane
    chunk = cfg.chunk
    tn = cfg.tile_n
    n_gu = cfg.gate_up_tiles
    per_seg = cfg.tiles_per_segment
    x3 = x_hbm.reshape(-1, sub, cfg.hidden)
    y3 = y_hbm.reshape(-1, sub, cfg.hidden)

    num_seg = plan_segments(
        group_sizes_ref, group_offset_ref, seg_group, seg_base, seg_lo, seg_hi, cfg
    )
    num_tiles = num_seg * per_seg

    def weight_copies(kind: int, group):
        if kind < n_gu:
            gate_cols = pl.ds(kind * tn, tn)
            up_cols = pl.ds(cfg.inter + kind * tn, tn)
            sem = weight_sems.at[kind]
            return [
                pltpu.make_async_copy(
                    w1_hbm.at[group, :, gate_cols], gate_up_w.at[kind, 0], sem
                ),
                pltpu.make_async_copy(
                    w1_hbm.at[group, :, up_cols], gate_up_w.at[kind, 1], sem
                ),
                pltpu.make_async_copy(
                    s1_hbm.at[group, :, :, gate_cols], gate_up_s.at[kind, 0], sem
                ),
                pltpu.make_async_copy(
                    s1_hbm.at[group, :, :, up_cols], gate_up_s.at[kind, 1], sem
                ),
            ]
        sem = weight_sems.at[n_gu]
        return [
            pltpu.make_async_copy(w2_hbm.at[group], down_w, sem),
            pltpu.make_async_copy(s2_hbm.at[group], down_s, sem),
        ]

    def for_tile(t, action):
        group = seg_group[t // per_seg]
        for kind in range(per_seg):

            @pl.when(t % per_seg == kind)
            def _(*, kind=kind):
                for copy in weight_copies(kind, group):
                    action(copy)

    def start_tile(t):
        for_tile(t, lambda copy: copy.start())

    def wait_tile(t):
        for_tile(t, lambda copy: copy.wait())

    def x_copy(t, c, slot):
        s = t // per_seg
        first_row = seg_base[s] + c * chunk
        num_sub = pl.cdiv(jnp.minimum(seg_hi[s] - first_row, chunk), sub)
        return pltpu.make_async_copy(
            x3.at[pl.ds(first_row // sub, num_sub)],
            x_buf.at[slot, pl.ds(0, num_sub)],
            x_sems.at[slot],
        )

    def y_copy(slot, first_sub, num_sub):
        return pltpu.make_async_copy(
            y_buf.at[slot, pl.ds(0, num_sub)],
            y3.at[pl.ds(first_sub, num_sub)],
            y_sem.at[0],
        )

    def for_bucket(rows, body):
        """Runs body with the smallest bucket that holds the chunk's rows."""
        index = sum((rows > b).astype(jnp.int32) for b in cfg.buckets[:-1])
        for i, bucket_m in enumerate(cfg.buckets):

            @pl.when(index == i)
            def _(*, bucket_m=bucket_m):
                body(bucket_m)

    def gate_up_chunk(kind, slot, s, c):
        first_row = seg_base[s] + c * chunk
        rhs = FusedWeightsRef(
            gate=WeightsRef(
                weight=gate_up_w.at[kind, 0], scale=gate_up_s.at[kind, 0], bias=None
            ),
            up=WeightsRef(
                weight=gate_up_w.at[kind, 1], scale=gate_up_s.at[kind, 1], bias=None
            ),
        )

        def body(bucket_m):
            lhs = x_buf[slot, : bucket_m // sub].reshape(bucket_m, cfg.hidden)
            acc = quantized_tile_dot(lhs, rhs, bucket_m, cfg.gate_up)
            acc = apply_act_fn(acc, cfg.gate_up.fuse_act)
            h_buf[kind, pl.ds(pl.multiple_of(c * chunk, chunk), bucket_m)] = (
                masked_rows(acc, first_row, seg_lo[s], seg_hi[s], cfg.gate_up.out_dtype)
            )

        for_bucket(seg_hi[s] - first_row, body)

    def down_chunk(s, c, is_last_chunk):
        first_row = seg_base[s] + c * chunk
        rhs = WeightsRef(weight=down_w, scale=down_s, bias=None)
        slot = state[3] % 2
        state[3] += 1

        def body(bucket_m):
            rows = pl.ds(pl.multiple_of(c * chunk, chunk), bucket_m)
            lhs = jnp.concatenate([h_buf[j, rows] for j in range(n_gu)], axis=1)
            acc = quantized_tile_dot(lhs, rhs, bucket_m, cfg.down)
            out = masked_rows(acc, first_row, seg_lo[s], seg_hi[s], cfg.down.out_dtype)
            y_buf[slot, : bucket_m // sub] = out.reshape(
                bucket_m // sub, sub, cfg.hidden
            )

        for_bucket(seg_hi[s] - first_row, body)
        first_sub = first_row // sub

        @pl.when(jnp.logical_and(state[0] == 1, state[1] == first_sub))
        def _():
            y_buf[slot, 0] += partial_ref[...]

        hi = seg_hi[s]
        num_sub = pl.cdiv(jnp.minimum(hi - first_row, chunk), sub)

        @pl.when(is_last_chunk)
        def _():
            last_sub = (hi - first_row) // sub
            has_partial = hi % sub != 0
            state[0] = has_partial.astype(jnp.int32)
            state[1] = first_sub + last_sub

            @pl.when(has_partial)
            def _():
                partial_ref[...] = y_buf[slot, last_sub]

        @pl.when(state[2] == 1)
        def _():
            y_copy(state[4], state[5], state[6]).wait()

        y_copy(slot, first_sub, num_sub).start()
        state[2] = 1
        state[4] = slot
        state[5] = first_sub
        state[6] = num_sub

    for i in range(7):
        state[i] = 0

    @pl.when(num_tiles > 0)
    def _():
        start_tile(0)
        x_copy(0, 0, 0).start()

    @pl.when(num_tiles > 1)
    def _():
        start_tile(1)

    def tile_loop(t, x_slot):
        s = t // per_seg
        kind = t % per_seg
        is_gate_up = kind < n_gu
        wait_tile(t)

        @pl.when(t + 2 < num_tiles)
        def _():
            start_tile(t + 2)

        num_chunks = pl.cdiv(seg_hi[s] - seg_base[s], chunk)

        def chunk_loop(c, x_slot):
            is_last_chunk = c == num_chunks - 1
            next_t = jnp.where(is_last_chunk, t + 1, t)
            next_c = jnp.where(is_last_chunk, 0, c + 1)
            next_is_gate_up = jnp.logical_and(
                next_t < num_tiles, next_t % per_seg < n_gu
            )
            next_slot = jnp.where(is_gate_up, 1 - x_slot, x_slot)

            @pl.when(next_is_gate_up)
            def _():
                x_copy(next_t, next_c, next_slot).start()

            for k in range(n_gu):

                @pl.when(kind == k)
                def _(*, k=k):
                    x_copy(t, c, x_slot).wait()
                    gate_up_chunk(k, x_slot, s, c)

            @pl.when(jnp.logical_not(is_gate_up))
            def _():
                down_chunk(s, c, is_last_chunk)

            return next_slot

        return lax.fori_loop(0, num_chunks, chunk_loop, x_slot)

    lax.fori_loop(0, num_tiles, tile_loop, jnp.int32(0))

    @pl.when(state[2] == 1)
    def _():
        y_copy(state[4], state[5], state[6]).wait()


def supports(
    w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, fuse_act, rhs_quant_dtype
) -> bool:
    """The configurations this kernel computes exactly as gmm_v2: fp8 weights with
    one scale per column."""
    return (
        fuse_act is not None
        and w1_bias is None
        and w2_bias is None
        and w1_scale is not None
        and w2_scale is not None
        and w1_scale.shape[1] == 1
        and w2_scale.shape[1] == 1
        and (rhs_quant_dtype is None or rhs_quant_dtype == w1.dtype)
        and w1.dtype == jnp.float8_e4m3fn
        and w2.dtype == jnp.float8_e4m3fn
        and w1.shape[2] % 2 == 0
        and (w1.shape[2] // 2) % 256 == 0
        and w2.shape[1] == w1.shape[2] // 2
        and w2.shape[2] == w1.shape[1]
    )


def make_fused_moe_configs(
    x,
    w1,
    w1_scale,
    w2,
    w2_scale,
    group_sizes,
    group_offset,
    *,
    fuse_act,
    buckets,
    rows_cap,
    vmem_limit_bytes,
) -> FusedMoeConfigs:
    chunk = buckets[-1]
    size_m, hidden = x.shape
    num_groups = w1.shape[0]
    inter = w1.shape[2] // 2
    tile_n = inter // 2
    gate_up = make_gmm_configs(
        x,
        w1,
        w1_scale,
        None,
        group_sizes,
        group_offset,
        tile_info=TileSizes(
            tile_m=chunk, tile_k=hidden, tile_n=tile_n, bucket_base=chunk
        ),
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=False,
        fuse_act=fuse_act,
    )
    h = jax.ShapeDtypeStruct((size_m, inter), gate_up.out_dtype)
    down = make_gmm_configs(
        h,  # pyrefly: ignore[bad-argument-type]
        w2,
        w2_scale,
        None,
        group_sizes,
        group_offset,
        tile_info=TileSizes(
            tile_m=chunk, tile_k=inter, tile_n=hidden, bucket_base=chunk
        ),
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=x.dtype,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=False,
        fuse_act=None,
    )
    if gate_up.lhs_cfgs.quant_dtype is None or down.lhs_cfgs.quant_dtype is None:
        raise ValueError("gmm_fused_moe computes the quantized matmul path only")
    if gate_up.dims.size_lhs_sublane != down.dims.size_lhs_sublane:
        raise ValueError("gate·up and down rows must share one sublane tiling")
    return FusedMoeConfigs(
        gate_up=gate_up,
        down=down,
        size_m=size_m,
        hidden=hidden,
        inter=inter,
        num_groups=num_groups,
        sublane=gate_up.dims.size_lhs_sublane,
        tile_n=tile_n,
        buckets=buckets,
        rows_cap=rows_cap,
    )


@functools.partial(
    jax.jit, static_argnames=["fuse_act", "buckets", "rows_cap", "vmem_limit_bytes"]
)
def gmm_fused_moe(
    x: jax.Array,  # [size_m, hidden]
    w1: jax.Array,  # [groups, hidden, 2 * inter]: gate columns, then up columns
    w1_scale: jax.Array,  # [groups, 1, 1, 2 * inter]
    w2: jax.Array,  # [groups, inter, hidden]
    w2_scale: jax.Array,  # [groups, 1, 1, hidden]
    group_sizes: jax.Array,  # int32[lhs groups]
    group_offset: jax.Array,  # int32 scalar or [1]
    *,
    fuse_act: str,
    buckets: tuple[int, ...] = (128, 256),
    rows_cap: int = 1024,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """gmm_v2(gmm_v2(x, w1, fuse_act), w2) on the routed rows, as one kernel."""
    size_m = x.shape[0]
    sublane = _get_lhs_sublane_size(x.dtype, size_m)
    padded_m = align_to(size_m, sublane)
    if padded_m != size_m:
        x = jnp.pad(x, ((0, padded_m - size_m), (0, 0)))
    if jnp.ndim(group_offset) == 0:
        group_offset = group_offset[None]
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    buckets = tuple(sorted(buckets))
    chunk = buckets[-1]
    if any(b % sublane for b in buckets) or rows_cap % chunk:
        raise ValueError(
            f"{buckets=} must be multiples of {sublane=} and the largest must "
            f"divide {rows_cap=}"
        )

    cfg = make_fused_moe_configs(
        x,
        w1,
        w1_scale,
        w2,
        w2_scale,
        group_sizes,
        group_offset,
        fuse_act=fuse_act,
        buckets=buckets,
        rows_cap=rows_cap,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    hidden, inter, tn = cfg.hidden, cfg.inter, cfg.tile_n
    n_gu = cfg.gate_up_tiles
    weight_dtype = w1.dtype
    act_dtype = cfg.gate_up.out_dtype
    out_dtype = cfg.down.out_dtype
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scratch_shapes = [
        pltpu.VMEM((n_gu, 2, hidden, tn), weight_dtype),
        pltpu.VMEM((n_gu, 2, 1, 1, tn), jnp.float32),
        pltpu.VMEM((inter, hidden), weight_dtype),
        pltpu.VMEM((1, 1, hidden), jnp.float32),
        pltpu.VMEM((2, chunk // sublane, sublane, hidden), x.dtype),
        pltpu.VMEM((n_gu, rows_cap, tn), act_dtype),
        pltpu.VMEM((2, chunk // sublane, sublane, hidden), out_dtype),
        pltpu.VMEM((sublane, hidden), out_dtype),
        pltpu.SMEM((cfg.max_segments,), jnp.int32),
        pltpu.SMEM((cfg.max_segments,), jnp.int32),
        pltpu.SMEM((cfg.max_segments,), jnp.int32),
        pltpu.SMEM((cfg.max_segments,), jnp.int32),
        pltpu.SMEM((8,), jnp.int32),
        pltpu.SemaphoreType.DMA((n_gu + 1,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((1,)),
    ]
    weight_bytes = (w1.size + w2.size) * jnp.dtype(weight_dtype).itemsize
    cost = pl.CostEstimate(
        flops=2 * padded_m * hidden * 3 * inter,
        transcendentals=padded_m * inter,
        bytes_accessed=weight_bytes
        + 2 * padded_m * hidden * jnp.dtype(x.dtype).itemsize,
    )
    out = pl.pallas_call(
        functools.partial(kernel, cfg=cfg),
        out_shape=jax.ShapeDtypeStruct((padded_m, hidden), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm, hbm, hbm, hbm, hbm],
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=(
            f"gmm_fused_moe-g_{cfg.num_groups}-m_{padded_m}-k_{hidden}-i_{inter}"
            f"-b_{'_'.join(map(str, buckets))}-cap_{rows_cap}"
        ),
        cost_estimate=cost,
    )(
        group_sizes,
        group_offset,
        x,
        w1,
        w1_scale.astype(jnp.float32),
        w2,
        w2_scale.astype(jnp.float32),
    )
    return out[:size_m]
