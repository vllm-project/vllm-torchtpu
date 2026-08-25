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
"""Single-call fused MoE grouped GEMM: gmm1 -> activation -> gmm2.

The two grouped matmuls of an MoE layer are normally two ``gmm_v2`` calls with
a ``[size_m, intermediate_size]`` activation written to HBM in between. This
module runs both from a single ``pallas_call`` and keeps that activation in
VMEM, so it never leaves the core.

The kernel body is ``gmm_v2.inner_kernel`` invoked twice per gm tile, so all
numerics, quantization, activation and masking behaviour are inherited from
``gmm_v2`` unchanged and the result is expected to be *bit-identical* to the
two-call path (see ``tests/kernels/test_gmm_v2_fused.py``).

Holding a whole expert's weights resident is what makes the k/n grid collapse,
and it is also what bounds applicability: ``can_fuse`` returns False when the
slabs do not fit in VMEM, and the caller keeps the two-call path.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.megablox.gmm_v2 import (GmmConfigs, MetadataRef,
                                                   WeightsRef, inner_kernel)

# gmm_v2 triple-buffers rhs because one of its blocks is a (tile_k, tile_n)
# slice. Here a block is a whole expert, so a third buffer would cost another
# full slab (8 MB of w1 on the 397B shapes) and buy nothing: with a single grid
# axis the pipeline can only ever be one expert ahead.
_WEIGHT_BUFFER_COUNT = 2

# Slack for what the estimator below deliberately does not model. The biggest
# term is the lane-interleaved gate|up weight that FusedWeightsRef.get_weight
# builds per gm tile -- 8.4 MB at the 397B gmm1 slab. Counting it outright
# would be double counting (gmm_v2 runs the same interleave at the same
# tile_k/tile_n today and fits), but leaving no room for it lets the tile_m
# search pick a size that only fits on paper and then fails to allocate.
# Raise this first if Mosaic reports a VMEM overflow.
_VMEM_HEADROOM_BYTES = 12 * 1024 * 1024

# Matches gmm_v2.calculate_tiling: the m-bucket switch is unrolled, so the
# branch count is bounded to keep instruction memory in check.
_MAX_NUM_BUCKETS = 4


class _FusedIndexMaps(gmm_v2.IndexMaps):
    """``gmm_v2.IndexMaps`` bound to the fused kernel's ``(num_gm,)`` grid.

    gmm_v2 runs a ``(num_n, num_gm, num_k)`` grid. The fused kernel keeps a
    whole expert resident, so ``num_n == num_k == 1`` and gm is the only axis;
    every map below pins ``n_id`` and ``k_id`` to zero and reuses the base
    implementation.
    """

    def lhs(self, gm_id: jax.Array):
        return self.lhs_index_map(0, gm_id, 0)

    def weight(self, gm_id: jax.Array):
        return self.rhs_weight_index_map(0, gm_id, 0)

    def scale(self, gm_id: jax.Array):
        return self.rhs_scale_index_map(0, gm_id, 0)

    def bias(self, gm_id: jax.Array):
        return self.rhs_bias_index_map(0, gm_id, 0)

    def out(self, gm_id: jax.Array):
        # Cannot delegate: the base version asks pl.num_programs(1) for the
        # last gm tile, and gm is axis 0 on this grid. Everything else follows
        # out_index_map -- only the last gm tile writes the trailing partial
        # sublane row, the rest leave it to partial_out.
        size_lhs_sublane = self.cfgs.dims.size_lhs_sublane
        is_last_gm = gm_id == (pl.num_programs(0) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // size_lhs_sublane
        capped_row_end = m_end // size_lhs_sublane
        last_row_end = pl.cdiv(m_end, size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)

        return (pl.ds(row_start, row_end - row_start), 0, 0)


def _acc_cols(cfgs: GmmConfigs) -> int:
    """Accumulator width: fused activations accumulate gate and up together."""
    if cfgs.fuse_act is None:
        return cfgs.tiles.tile_n
    return 2 * cfgs.tiles.tile_n


def _acc_view(acc_ref: jax.Array, cols: int):
    """The leading ``cols`` lanes of the shared accumulator."""
    if cols == acc_ref.shape[-1]:
        return acc_ref
    return acc_ref.at[:, pl.ds(0, cols)]


def _weights_spec(cfgs: GmmConfigs, index_maps: _FusedIndexMaps) -> WeightsRef:
    """Block specs for one gmm's rhs, one block per expert.

    Mirrors ``gmm_v2.generate_block_specs``; only the index maps and the
    buffer count differ. ``tile_k`` counts unpacked elements, matching the
    ``bitcast`` the kernel applies to sub-byte weights before the pipeline.
    """
    tiles = cfgs.tiles
    weight_spec = pl.BlockSpec(
        (None, tiles.tile_k, tiles.tile_n),
        index_maps.weight,
        pipeline_mode=pl.Buffered(buffer_count=_WEIGHT_BUFFER_COUNT),
    )
    scale_spec = bias_spec = None
    if cfgs.rhs_cfgs.has_scale:
        scale_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, 1, tiles.tile_n),
            index_maps.scale,
        )
    if cfgs.rhs_cfgs.has_bias:
        bias_spec = pl.BlockSpec((None, 1, tiles.tile_n), index_maps.bias)

    return WeightsRef(weight=weight_spec, scale=scale_spec, bias=bias_spec)


def fused_inner_kernel(
    # In
    lhs_ref: jax.Array,  # [tile_m // size_lhs_sublane, sublane, size_k1]
    w1_ref: WeightsRef,  # [size_k1, size_n1]
    w2_ref: WeightsRef,  # [size_k2, size_n2]
    # Out
    out_ref: jax.Array,  # [tile_m // size_lhs_sublane, sublane, size_n2]
    # Scratch
    gmm1_out_ref: jax.Array,  # [tile_m // sublane + 1, sublane, size_k2]
    partial_out1_ref: jax.Array,  # [size_lhs_sublane, out_size_n1]
    partial_out2_ref: jax.Array,  # [size_lhs_sublane, size_n2]
    acc_ref: jax.Array,  # [tile_m, max(acc_cols1, acc_cols2)]
    metadata_ref: MetadataRef,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """One gm tile: gmm1 -> fused activation -> gmm2.

    ``gmm1_out_ref`` is gmm1's output ref and gmm2's lhs ref with no reshape,
    cast or copy in between: gmm1 already writes the post-activation value in
    gmm2's lhs dtype and in the same 3D sublane layout.

    The straddle handling of both matmuls is left exactly as gmm_v2 wrote it,
    with a ``partial_out`` ref each. gmm1's straddle accumulates the previous
    expert's last rows into the intermediate, which is harmless: those rows
    sit outside ``[m_start_local, m_end_local)`` and gmm2 masks them away, and
    the two-call path feeds gmm2 the same rows from HBM.
    """
    gm_id = pl.program_id(0)

    inner_kernel(
        lhs_ref,
        w1_ref,
        gmm1_out_ref,
        partial_out1_ref,
        _acc_view(acc_ref, _acc_cols(cfgs1)),
        metadata_ref,
        cfgs=cfgs1,
        _gm_id=gm_id,
        _k_id=0,
        _num_k=1,
    )

    inner_kernel(
        gmm1_out_ref,
        w2_ref,
        out_ref,
        partial_out2_ref,
        _acc_view(acc_ref, _acc_cols(cfgs2)),
        metadata_ref,
        cfgs=cfgs2,
        _gm_id=gm_id,
        _k_id=0,
        _num_k=1,
    )


def fused_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k1]
    w1_ref: WeightsRef,  # [size_group, size_k1, size_n1]
    w2_ref: WeightsRef,  # [size_group, size_k2, size_n2]
    # Out
    out_ref: jax.Array,  # [size_m, size_n2]
    # Scratch memory
    gmm1_out_ref: jax.Array,
    partial_out1_ref: jax.Array,
    partial_out2_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """Entry point for the fused GMM kernel.

    Same structure as ``gmm_v2.kernel_main``: compute the gm tiling metadata,
    then run ``emit_pipeline`` over it. The grid is ``(num_gm,)`` because both
    matmuls hold a whole expert, so there is nothing left to tile over k or n.
    """
    # One metadata scan serves both matmuls, which is only sound because
    # fill_metadata reads nothing that differs between the two configs.
    assert cfgs1.dims.size_group == cfgs2.dims.size_group
    assert cfgs1.dims.size_lhs_group == cfgs2.dims.size_lhs_group
    assert cfgs1.dims.size_lhs_sublane == cfgs2.dims.size_lhs_sublane
    assert cfgs1.tiles.tile_m == cfgs2.tiles.tile_m

    if cfgs1.rhs_cfgs.should_unpack:
        w1_ref = dataclasses.replace(w1_ref,
                                     weight=w1_ref.weight.bitcast(
                                         cfgs1.rhs_cfgs.quant_dtype))
    if cfgs2.rhs_cfgs.should_unpack:
        w2_ref = dataclasses.replace(w2_ref,
                                     weight=w2_ref.weight.bitcast(
                                         cfgs2.rhs_cfgs.quant_dtype))

    num_gm = gmm_v2.fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs1,
    )

    index_maps1 = _FusedIndexMaps(metadata_ref, cfgs1)
    index_maps2 = _FusedIndexMaps(metadata_ref, cfgs2)
    size_lhs_sublane = cfgs1.dims.size_lhs_sublane
    bounded_slice_gm = pl.BoundedSlice(cfgs1.tiles.tile_m // size_lhs_sublane)

    lhs_spec = pl.BlockSpec(
        (bounded_slice_gm, size_lhs_sublane, cfgs1.tiles.tile_k),
        index_maps1.lhs,
    )
    w1_spec = _weights_spec(cfgs1, index_maps1)
    w2_spec = _weights_spec(cfgs2, index_maps2)
    out_spec = pl.BlockSpec(
        (bounded_slice_gm, size_lhs_sublane, cfgs2.tiles.tile_n),
        index_maps2.out,
    )

    if cfgs1.fuse_act is not None:
        # The same gate/up split kernel_main does, on the HBM refs before the
        # pipeline: the gate half is the unsliced ref (tile_n stops at
        # out_size_n) and the up half is pre-sliced, so the two arrive as
        # separate VMEM blocks and FusedWeightsRef lane-interleaves them.
        w1_up_ref = jax.tree.map(lambda x: x.at[..., cfgs1.out_size_n:],
                                 w1_ref)
        w1_ref = gmm_v2.FusedWeightsRef(gate=w1_ref, up=w1_up_ref)
        w1_spec = gmm_v2.FusedWeightsRef(gate=w1_spec, up=w1_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(fused_inner_kernel, cfgs1=cfgs1, cfgs2=cfgs2),
        grid=(num_gm, ),
        in_specs=(lhs_spec, w1_spec, w2_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane
    # size. w1/w2 use static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, size_lhs_sublane, out_ref.shape[-1])
    scratches = [
        gmm1_out_ref, partial_out1_ref, partial_out2_ref, acc_ref, metadata_ref
    ]
    pipeline_fn(lhs_in, w1_ref, w2_ref, out_in, scratches=scratches)


def _itemsize(dtype: jnp.dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _rhs_vmem_estimate(cfgs: GmmConfigs) -> int:
    """VMEM held by one gmm's rhs: every buffer of weight, scale and bias."""
    dims = cfgs.dims
    rhs_cfgs = cfgs.rhs_cfgs
    rhs_bits = jax.dtypes.itemsize_bits(rhs_cfgs.quant_dtype or rhs_cfgs.dtype)

    # tile_n covers one of gate/up, so size_n already counts both halves.
    total = _WEIGHT_BUFFER_COUNT * dims.size_k * dims.size_n * rhs_bits // 8
    if rhs_cfgs.has_scale:
        # Only the weight spec overrides the buffer count; emit_pipeline
        # double-buffers scale and bias.
        total += 2 * cfgs.num_quant_blocks_per_tile_k * dims.size_n * 4
    if rhs_cfgs.has_bias:
        total += 2 * dims.size_n * 4
    if rhs_cfgs.should_dequantize_before_matmul:
        # The whole slab is dequantized into a live lhs-dtype value per gm
        # tile. gmm_v2's estimator ignores this because its tile_k keeps the
        # temporary small; here tile_k is the entire contraction, so a
        # sub-MXU quant block can easily be the term that stops us fusing.
        total += dims.size_k * dims.size_n * _itemsize(cfgs.lhs_cfgs.dtype)
    return total


def fused_vmem_estimate(cfgs1: GmmConfigs, cfgs2: GmmConfigs) -> int:
    """VMEM bytes the fused kernel needs at ``cfgs1.tiles.tile_m``."""
    dims1, dims2 = cfgs1.dims, cfgs2.dims
    size_lhs_sublane = dims1.size_lhs_sublane
    tile_m = cfgs1.tiles.tile_m
    mid_bytes = _itemsize(cfgs1.out_dtype)
    out_bytes = _itemsize(cfgs2.out_dtype)

    total = _rhs_vmem_estimate(cfgs1) + _rhs_vmem_estimate(cfgs2)
    # Pipelined lhs and out, double-buffered.
    total += 2 * tile_m * dims1.size_k * _itemsize(cfgs1.lhs_cfgs.dtype)
    total += 2 * tile_m * dims2.size_n * out_bytes
    # The VMEM intermediate, plus its guard sublane row.
    total += (tile_m + size_lhs_sublane) * dims2.size_k * mid_bytes
    # The shared accumulator and the two straddle buffers.
    total += tile_m * max(_acc_cols(cfgs1), _acc_cols(cfgs2)) * _itemsize(
        cfgs1.acc_dtype)
    total += size_lhs_sublane * (cfgs1.out_size_n * mid_bytes +
                                 dims2.size_n * out_bytes)
    # On-the-fly lhs quantization needs a single-buffered cast buffer each.
    for cfgs in (cfgs1, cfgs2):
        if cfgs.lhs_cfgs.should_quantize:
            total += tile_m * cfgs.dims.size_k * _itemsize(
                cfgs.lhs_cfgs.quant_dtype)

    return total + _VMEM_HEADROOM_BYTES


def _configs_are_fusable(cfgs1: GmmConfigs, cfgs2: GmmConfigs,
                         vmem_limit_bytes: int) -> bool:
    """Whether this pair of configs can run as one kernel."""
    # gmm2 consumes gmm1's output as its lhs with no reshape or cast.
    if cfgs1.out_size_n != cfgs2.dims.size_k:
        return False
    if cfgs1.out_dtype != cfgs2.lhs_cfgs.dtype:
        return False
    # A second fused activation would halve gmm2's output width, which the
    # single output ref cannot express.
    if cfgs2.fuse_act is not None:
        return False
    # Skipped groups must stay uninitialized; the fused kernel has no
    # zero_out path and the MoE combine masks those rows anyway.
    if cfgs1.zero_init or cfgs2.zero_init:
        return False
    # One accumulator serves both matmuls.
    if cfgs1.acc_dtype != cfgs2.acc_dtype:
        return False
    # One metadata scan and one gm grid serve both matmuls.
    if cfgs1.dims.size_m != cfgs2.dims.size_m:
        return False
    if cfgs1.dims.size_group != cfgs2.dims.size_group:
        return False
    if cfgs1.dims.size_lhs_group != cfgs2.dims.size_lhs_group:
        return False
    if cfgs1.dims.size_lhs_sublane != cfgs2.dims.size_lhs_sublane:
        return False
    if cfgs1.tiles.tile_m != cfgs2.tiles.tile_m:
        return False
    if cfgs1.tiles.bucket_base != cfgs2.tiles.bucket_base:
        return False

    num_lanes = pltpu.get_tpu_info().num_lanes
    for cfgs in (cfgs1, cfgs2):
        # num_k == 1: the kernel drives inner_kernel with a single k step.
        if cfgs.tiles.tile_k != cfgs.dims.size_k:
            return False
        # num_n == 1: inner_kernel reshapes the whole output ref by tile_n, so
        # an n-tile of a wider buffer is not expressible.
        if cfgs.tiles.tile_n != cfgs.out_size_n:
            return False
        # A padded output row would make the out block a partial slice of a
        # wider HBM buffer, which Blocked() cannot index.
        if gmm_v2.align_to(cfgs.out_size_n, num_lanes) != cfgs.out_size_n:
            return False
        # tile_m counts whole sublane rows (it bounds the lhs/out
        # BoundedSlice) and whole m buckets (inner_kernel dispatches on
        # m_end_local // bucket_base, and lax.switch clamps that index, so a
        # remainder would silently drop the tail rows of a full tile).
        if cfgs.tiles.tile_m % cfgs.dims.size_lhs_sublane != 0:
            return False
        if cfgs.tiles.tile_m % cfgs.tiles.bucket_base != 0:
            return False

    return fused_vmem_estimate(cfgs1, cfgs2) <= vmem_limit_bytes


def _slab_tiles(tile_m: int, bucket_base: int):
    """A ``TileFn`` that pins tile_k and tile_n to the whole expert."""

    def tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act):
        del lhs_cfgs, rhs_cfgs, vmem_limit_bytes
        out_size_n = dims.size_n if fuse_act is None else dims.size_n // 2
        return gmm_v2.TileSizes(tile_m=tile_m,
                                tile_k=dims.size_k,
                                tile_n=out_size_n,
                                bucket_base=bucket_base)

    return tile_fn


def _retile(cfgs: GmmConfigs, tile_m: int, bucket_base: int) -> GmmConfigs:
    return dataclasses.replace(cfgs,
                               tiles=gmm_v2.TileSizes(tile_m=tile_m,
                                                      tile_k=cfgs.dims.size_k,
                                                      tile_n=cfgs.out_size_n,
                                                      bucket_base=bucket_base))


def _base_tile_m(cfgs: GmmConfigs) -> int:
    """gmm_v2.calculate_tiling's tile_m heuristic, without the vmem search."""
    rhs_bits = jax.dtypes.itemsize_bits(cfgs.rhs_cfgs.quant_dtype
                                        or cfgs.rhs_cfgs.dtype)
    bf16_bf16_tile_m = 128
    tile_m = bf16_bf16_tile_m // min(pl.cdiv(16, rhs_bits), 2)
    if cfgs.lhs_cfgs.should_quantize:
        tile_m *= 2
    return tile_m


def _normalize_group_offset(group_offset: jax.Array | None) -> jax.Array:
    """Same normalization gmm_v2 applies to its ``group_offset`` argument."""
    if group_offset is None:
        return jnp.array([0], dtype=jnp.int32)
    if jnp.isscalar(group_offset):
        return group_offset[None]
    return group_offset


def plan_fusion(
    lhs: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    vmem_limit_bytes: int | None = None,
    fuse_act: str | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    preferred_element_type: jnp.dtype | None = None,
    tile_m: int | None = None,
    bucket_base: int | None = None,
) -> tuple[GmmConfigs, GmmConfigs] | None:
    """Builds the fused kernel's two configs, or None if the shape can't fuse.

    Pure Python over shapes and dtypes -- it never traces the kernel, so a
    caller can run it on every trace to pick between this path and two
    ``gmm_v2`` calls. ``lhs`` may be a ``jax.ShapeDtypeStruct``.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    group_offset = _normalize_group_offset(group_offset)

    # Plan against the row-padded lhs the kernel actually sees, so callers get
    # the same answer whether or not they have padded yet.
    size_lhs_sublane = gmm_v2._get_lhs_sublane_size(lhs.dtype, lhs.shape[0])
    lhs = jax.ShapeDtypeStruct(
        (gmm_v2.align_to(lhs.shape[0], size_lhs_sublane), lhs.shape[1]),
        lhs.dtype)

    def build(probe_tile_m: int, probe_bucket_base: int):
        cfgs1 = gmm_v2.make_gmm_configs(
            lhs,
            w1,
            w1_scale,
            w1_bias,
            group_sizes,
            group_offset,
            tile_info=_slab_tiles(probe_tile_m, probe_bucket_base),
            vmem_limit_bytes=vmem_limit_bytes,
            # None means "lhs dtype", matching what moe_gmm asks of gmm1.
            out_dtype=None,
            acc_dtype=None,
            maybe_quantize_lhs=maybe_quantize_lhs,
            zero_initialize=False,
            fuse_act=fuse_act,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        size_k2 = w2.shape[1] * gmm_v2.get_packing_factor(
            w2.dtype, rhs_quant_dtype)
        intermediate = jax.ShapeDtypeStruct((cfgs1.dims.size_m, size_k2),
                                            cfgs1.out_dtype)
        cfgs2 = gmm_v2.make_gmm_configs(
            intermediate,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            group_offset,
            tile_info=_slab_tiles(probe_tile_m, probe_bucket_base),
            vmem_limit_bytes=vmem_limit_bytes,
            out_dtype=preferred_element_type,
            acc_dtype=None,
            maybe_quantize_lhs=maybe_quantize_lhs,
            zero_initialize=False,
            fuse_act=None,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        return cfgs1, cfgs2

    # Probe pass: only make_gmm_configs knows the derived dims and dtypes the
    # tile search needs, and it wants the tiles up front. The probe tiles are
    # discarded below.
    probe1, probe2 = build(1, 1)

    if tile_m is None:
        size_m = probe1.dims.size_m
        tile_m = min(_base_tile_m(probe1), _base_tile_m(probe2), size_m)
        # tile_m must be a whole number of sublane rows: it is the bound of
        # the lhs/out BoundedSlice, which counts sublane rows.
        tile_m = max(size_lhs_sublane, tile_m - tile_m % size_lhs_sublane)
        bucket_base = tile_m
        for _ in range(1, _MAX_NUM_BUCKETS):
            grown = tile_m + bucket_base
            if grown > size_m:
                break
            if fused_vmem_estimate(_retile(probe1, grown, bucket_base),
                                   _retile(probe2, grown,
                                           bucket_base)) > vmem_limit_bytes:
                break
            tile_m = grown
    elif bucket_base is None:
        bucket_base = tile_m

    cfgs1 = _retile(probe1, tile_m, bucket_base)
    cfgs2 = _retile(probe2, tile_m, bucket_base)
    if not _configs_are_fusable(cfgs1, cfgs2, vmem_limit_bytes):
        return None
    return cfgs1, cfgs2


def can_fuse(*args, **kwargs) -> bool:
    """Whether ``gmm_v2_fused`` can run this shape. See ``plan_fusion``."""
    return plan_fusion(*args, **kwargs) is not None


def get_scope_name(cfgs1: GmmConfigs, cfgs2: GmmConfigs) -> str:
    dims1, dims2 = cfgs1.dims, cfgs2.dims
    tiles = cfgs1.tiles
    return (f"gmm_v2_fused-g_{dims1.size_group}-m_{dims1.size_m}"
            f"-k1_{dims1.size_k}-n1_{dims1.size_n}-act_{cfgs1.fuse_act}"
            f"-k2_{dims2.size_k}-n2_{dims2.size_n}"
            f"-tm_{tiles.tile_m}-bb_{tiles.bucket_base}")


def get_fused_cost_estimate(cfgs1: GmmConfigs,
                            cfgs2: GmmConfigs) -> pl.CostEstimate:
    """Both matmuls, minus the HBM round trip the fusion removes."""
    cost1 = gmm_v2.get_cost_estimate(cfgs1)
    cost2 = gmm_v2.get_cost_estimate(cfgs2)
    # cost1 counts writing the intermediate and cost2 counts reading it back.
    intermediate_bytes = (cfgs1.dims.size_m * cfgs1.out_size_n *
                          _itemsize(cfgs1.out_dtype))
    return pl.CostEstimate(
        flops=cost1.flops + cost2.flops,
        bytes_accessed=(cost1.bytes_accessed + cost2.bytes_accessed -
                        2 * intermediate_bytes),
        transcendentals=0,
    )


def get_fused_metadata(cfgs1: GmmConfigs,
                       cfgs2: GmmConfigs) -> dict[str, str | int | float]:
    metadata = {f"gmm1.{k}": v for k, v in gmm_v2.get_metadata(cfgs1).items()}
    metadata.update({
        f"gmm2.{k}": v
        for k, v in gmm_v2.get_metadata(cfgs2).items()
    })
    return metadata


@jax.jit(static_argnames=[
    "vmem_limit_bytes",
    "fuse_act",
    "rhs_quant_dtype",
    "maybe_quantize_lhs",
    "preferred_element_type",
    "tile_m",
    "bucket_base",
])
def gmm_v2_fused(
    lhs: jax.Array,  # [size_m, size_k1]
    w1: jax.Array,  # [size_group, size_k1, size_n1]
    w2: jax.Array,  # [size_group, size_k2, size_n2]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    w1_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, size_n1]
    w2_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, size_n2]
    w1_bias: jax.Array | None = None,  # [size_group, 1, size_n1]
    w2_bias: jax.Array | None = None,  # [size_group, 1, size_n2]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    vmem_limit_bytes: int | None = None,
    fuse_act: str | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    preferred_element_type: jnp.dtype | None = None,
    tile_m: int | None = None,
    bucket_base: int | None = None,
) -> jax.Array:
    """Fused MoE grouped GEMM: ``gmm2(act(gmm1(lhs, w1)), w2)``.

    Equivalent to, and expected to be bit-identical with::

        mid = gmm_v2(lhs, w1, group_sizes, w1_scale, w1_bias, group_offset,
                     zero_initialize=False, fuse_act=fuse_act)
        out = gmm_v2(mid[:, :size_k2], w2, group_sizes, w2_scale, w2_bias,
                     group_offset, zero_initialize=False)

    only without ``mid`` ever reaching HBM. ``can_fuse`` must return True for
    the same arguments; callers that cannot guarantee that should ask it
    first and fall back to the two calls above.

    Args:
        lhs: lhs with shape [size_m, size_k1].
        w1: First rhs, gate and up concatenated along n when ``fuse_act`` is
            set: native -> [size_group, size_k1, size_n1], packed ->
            [size_group, size_k1/packing, size_n1].
        w2: Second rhs, laid out like ``w1`` but never activation-fused.
        group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
        w1_scale: Scale of ``w1``, [size_group, num_blocks, 1, size_n1].
        w2_scale: Scale of ``w2``, [size_group, num_blocks, 1, size_n2].
        w1_bias: Bias of ``w1``, [size_group, 1, size_n1].
        w2_bias: Bias of ``w2``, [size_group, 1, size_n2].
        group_offset: Optional. The group offset of shape [1,].
        vmem_limit_bytes: Optional vmem limit in bytes.
        fuse_act: Activation fused into gmm1, None if no fusion.
        rhs_quant_dtype: Quantized dtype encoded by a pre-packed rhs carrier.
        maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
        preferred_element_type: Optional jnp.dtype for the output matrix.
        tile_m: Overrides the m tile both matmuls use. Mostly for tests, which
            need the two paths to tile identically to compare bit-exactly.
        bucket_base: Overrides the m-bucket granularity. Defaults to tile_m.

    Returns:
        Output of shape [size_m, size_n2].
    """
    # The kernel reshapes lhs by its sublane block, so pad the rows up to it.
    # The extra rows sit past every group, so they add no work, and they are
    # sliced back off the output below.
    size_m = lhs.shape[0]
    size_lhs_sublane = gmm_v2._get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = gmm_v2.align_to(size_m, size_lhs_sublane)
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))

    group_offset = _normalize_group_offset(group_offset)

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    plan = plan_fusion(
        lhs,
        w1,
        w2,
        group_sizes,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        group_offset,
        vmem_limit_bytes=vmem_limit_bytes,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        preferred_element_type=preferred_element_type,
        tile_m=tile_m,
        bucket_base=bucket_base,
    )
    assert plan is not None, (
        "gmm_v2_fused called on a shape that cannot fuse; call can_fuse "
        "first and fall back to two gmm_v2 calls.")
    cfgs1, cfgs2 = plan
    dims1, dims2 = cfgs1.dims, cfgs2.dims
    tiles = cfgs1.tiles

    # Prepare block specs.
    w1_scale_spec = w1_bias_spec = None
    w2_scale_spec = w2_bias_spec = None
    if w1_scale is not None:
        w1_scale = w1_scale.astype(jnp.float32)
        w1_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if w2_scale is not None:
        w2_scale = w2_scale.astype(jnp.float32)
        w2_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if w1_bias is not None:
        w1_bias = w1_bias.astype(jnp.float32)
        w1_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if w2_bias is not None:
        w2_bias = w2_bias.astype(jnp.float32)
        w2_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Initialize scratch shapes.
    max_num_gm = dims1.size_group + pl.cdiv(dims1.size_m, tiles.tile_m) - 1
    acc_cols = max(_acc_cols(cfgs1), _acc_cols(cfgs2))
    scratch_shapes = [
        # gmm1_out_ref: gmm1's output tile and gmm2's lhs tile in one buffer.
        # inner_kernel stages the sublane straddle by reading row
        # m_end_local // size_lhs_sublane, which is one past the end when the
        # tile is full. gmm_v2 gets away with that because the pipeline's
        # output buffer has a neighbouring slot; a bare scratch does not, so
        # keep a guard row. The value read there is always discarded by the
        # jnp.where next to it.
        pltpu.VMEM((tiles.tile_m // dims1.size_lhs_sublane + 1,
                    dims1.size_lhs_sublane, dims2.size_k), cfgs1.out_dtype),
        # partial_out1_ref
        pltpu.VMEM((dims1.size_lhs_sublane, cfgs1.out_size_n),
                   cfgs1.out_dtype),
        # partial_out2_ref
        pltpu.VMEM((dims2.size_lhs_sublane, cfgs2.tiles.tile_n),
                   cfgs2.out_dtype),
        # acc_ref, shared by both matmuls.
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs1.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
    ]

    out_init = jax.ShapeDtypeStruct((dims2.size_m, cfgs2.out_size_n),
                                    cfgs2.out_dtype)
    w1_weights = WeightsRef(weight=w1, scale=w1_scale, bias=w1_bias)
    w2_weights = WeightsRef(weight=w2, scale=w2_scale, bias=w2_bias)

    return pl.pallas_call(
        functools.partial(fused_kernel_main, cfgs1=cfgs1, cfgs2=cfgs2),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=w1_scale_spec,
                    bias=w1_bias_spec,
                ),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=w2_scale_spec,
                    bias=w2_bias_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs1, cfgs2),
        cost_estimate=get_fused_cost_estimate(cfgs1, cfgs2),
        metadata=get_fused_metadata(cfgs1, cfgs2),
    )(group_sizes, group_offset, lhs, w1_weights,
      w2_weights)[:size_m, :cfgs2.out_size_n]
