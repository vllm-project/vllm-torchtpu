# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Experimental decode GMM2 + owner combine with VMEM-resident routes.

Uses the existing GMM arithmetic and BF16 output rounding. No changes to the
standard GMM entry point or MoE dispatch are needed to benchmark this kernel.
"""

import functools
import operator

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2 as gmm
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import \
    _accumulate_route_block


def _vmem_bytes(rows: int, size_k: int, num_tokens: int, tile_m: int,
                tile_n: int, sublane: int) -> int:
    """Conservative estimate, including pipeline buffers and compiler workspace.

    Route storage uses capacity, not the data-dependent number of valid routes.
    This is an admission estimate; the compiler still enforces its VMEM limit.
    """
    physical_rows = gmm.align_to(rows + tile_m, 128)
    routes = physical_rows * tile_n * 2
    owner_and_store = num_tokens * tile_n * (4 + 2)
    partial_and_acc = (sublane * 2 + tile_m * 4) * tile_n
    ids_and_weights = physical_rows * (4 + 4)
    # Double-buffered BF16 LHS and a quantized LHS temporary; triple-buffered
    # FP8 RHS and double-buffered FP32 scales, including sublane padding.
    lhs_pipeline = tile_m * size_k * (2 * 2 + 1)
    rhs_pipeline = 3 * size_k * tile_n + 2 * sublane * tile_n * 4
    selector = num_tokens * 128 * (2 + 4)
    workspace = 4 * 1024 * 1024
    return (routes + owner_and_store + partial_and_acc + ids_and_weights +
            lhs_pipeline + rhs_pipeline + selector + workspace)


def can_use_gmm_v2_owner(lhs,
                         rhs,
                         group_sizes,
                         rhs_scale,
                         sorted_token_ids,
                         sorted_weights,
                         *,
                         num_tokens,
                         tile_n=None,
                         tile_m=128) -> bool:
    """Check static input contracts and VMEM admission without reading tensors.

    Independent of model name, expert count, top-k, EP and routing backend.
    A supported shape is not a promise of a speedup; service use is opt-in.
    Runtime group sizes and the valid sorted prefix must obey the kernel's
    documented contract. Unsupported shapes should use the existing GMM path.
    """
    try:
        num_tokens = operator.index(num_tokens)
        tile_m = operator.index(tile_m)
        tile_n = None if tile_n is None else operator.index(tile_n)
    except TypeError:
        return False
    if lhs.ndim != 2 or rhs.ndim != 3:
        return False
    rows, size_k = lhs.shape
    experts, rhs_k, size_n = rhs.shape
    if (min(rows, size_k, experts, size_n, num_tokens, tile_m) <= 0
            or lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.float8_e4m3fn
            or rhs_k != size_k or rhs_scale is None
            or rhs_scale.shape != (experts, 1, 1, size_n)
            or rhs_scale.dtype not in (jnp.float32, jnp.bfloat16)
            or group_sizes.shape != (experts, )
            or group_sizes.dtype != jnp.int32
            or sorted_token_ids.shape != (rows, )
            or sorted_token_ids.dtype != jnp.int32
            or sorted_weights.shape != (rows, )
            or sorted_weights.dtype != jnp.bfloat16):
        return False
    info = pltpu.get_tpu_info()
    sublane = info.get_sublane_tiling(jnp.bfloat16)
    if tile_n is None:
        # Full-width by default. Splitting N repeats the pipeline and LHS
        # reads; do not shrink tiles just to admit larger shapes into VMEM.
        tile_n = size_n
    if (info.fp8_ops_per_second <= 0 or rows < sublane or tile_m % 128
            or tile_n <= 0 or tile_n % info.mxu_column_size or size_n % tile_n
            or size_k % info.num_lanes or num_tokens % sublane):
        return False
    required = _vmem_bytes(rows, size_k, num_tokens, tile_m, tile_n, sublane)
    # Scalar-prefetched group sizes plus the two expert/tile metadata arrays.
    max_gm = experts + pl.cdiv(gmm.align_to(rows, sublane), tile_m) - 1
    smem_required = (experts + 1 + 2 * max_gm + 1) * 4 + 4096
    return (required <= int(info.vmem_capacity_bytes * 0.9)
            and smem_required <= int(info.smem_capacity_bytes * 0.9))


def _gmm_body(lhs_ref, rhs_ref, partial_ref, acc_ref, metadata_ref, routes_ref,
              *, cfgs):
    # The normal output pipeline only DMA-writes completed sublanes. Here
    # full tiles alias VMEM; the normal partial-sublane carry still preserves
    # the previous expert's rows at the start of the next tile.
    start = metadata_ref.gm_id_to_m_offset[pl.program_id(1)]
    start = start // cfgs.dims.size_lhs_sublane
    out_tile = routes_ref.at[
        pl.ds(start, cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane), :, :]
    gmm.inner_kernel(lhs_ref,
                     rhs_ref,
                     out_tile,
                     partial_ref,
                     acc_ref,
                     metadata_ref,
                     cfgs=cfgs)


def _kernel(group_sizes_ref, group_offset_ref, ids_ref, weights_ref, lhs_ref,
            rhs_ref, out_ref, partial_ref, acc_ref, metadata_ref, routes_ref,
            owner_ref, store_ref, store_sem, *, cfgs, num_tokens):
    n_start = pl.program_id(0) * cfgs.tiles.tile_n
    rhs_tile = gmm.WeightsRef(
        weight=rhs_ref.weight.at[:, :, pl.ds(n_start, cfgs.tiles.tile_n)],
        scale=rhs_ref.scale.at[:, :, :,
                               pl.ds(n_start, cfgs.tiles.tile_n)],
        bias=None)
    num_gm = gmm.fill_metadata(group_sizes_ref,
                               group_offset_ref,
                               metadata_ref,
                               cfgs=cfgs)
    (lhs_spec, rhs_spec), _ = gmm.generate_block_specs(metadata_ref, cfgs)
    pipeline = pltpu.emit_pipeline(functools.partial(_gmm_body, cfgs=cfgs),
                                   grid=(1, num_gm,
                                         pl.cdiv(cfgs.dims.size_k,
                                                 cfgs.tiles.tile_k)),
                                   in_specs=(lhs_spec, rhs_spec),
                                   out_specs=())
    pipeline(lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, cfgs.dims.size_k),
             rhs_tile,
             scratches=(partial_ref, acc_ref, metadata_ref, routes_ref))

    valid_count = metadata_ref.gm_id_to_m_offset[num_gm]
    owner_ref[...] = jnp.zeros_like(owner_ref)
    route_rows_ref = routes_ref.reshape(-1, cfgs.tiles.tile_n)

    def combine_block(block, _):
        start = block * 128
        offsets = start + jnp.arange(128, dtype=jnp.int32)
        ids = ids_ref[pl.ds(start, 128)]
        weights = weights_ref[pl.ds(start, 128)].astype(jnp.bfloat16)
        route_rows = route_rows_ref[pl.ds(start, 128)]
        row_valid = start + lax.broadcasted_iota(jnp.int32, route_rows.shape,
                                                 0) < valid_count
        _accumulate_route_block(route_rows,
                                owner_ref,
                                ids,
                                weights,
                                offsets < valid_count,
                                row_valid,
                                num_tokens=num_tokens)

    lax.fori_loop(0, pl.cdiv(valid_count, 128), combine_block, None)
    store_ref[...] = owner_ref[...].astype(jnp.bfloat16)
    copy = pltpu.make_async_copy(
        store_ref, out_ref.at[:, pl.ds(n_start, cfgs.tiles.tile_n)], store_sem)
    copy.start()
    copy.wait()


@jax.jit(static_argnames=("num_tokens", "tile_n", "tile_m"))
def gmm_v2_owner(lhs,
                 rhs,
                 group_sizes,
                 rhs_scale,
                 sorted_token_ids,
                 sorted_weights,
                 *,
                 num_tokens,
                 tile_n=None,
                 tile_m=128):
    """BF16 LHS / per-channel FP8 GMM2 followed by weighted owner reduction.

Runtime contract: nonnegative group_sizes sum to at most lhs.shape[0]; the
valid expert-sorted prefix has token IDs in [0, num_tokens). Tail IDs and
weights are ignored. Shapes/configuration are checked here once.
    """
    if not can_use_gmm_v2_owner(lhs,
                                rhs,
                                group_sizes,
                                rhs_scale,
                                sorted_token_ids,
                                sorted_weights,
                                num_tokens=num_tokens,
                                tile_n=tile_n,
                                tile_m=tile_m):
        raise ValueError("owner GMM unsupported dtype, shape, alignment or "
                         "estimated VMEM requirement")
    rows, size_k = lhs.shape
    experts, _, size_n = rhs.shape
    # Mosaic's replicated scale load requires 32-bit elements. Widening BF16
    # scales is exact; GMM arithmetic still applies its original rounding.
    rhs_scale = rhs_scale.astype(jnp.float32)
    info = pltpu.get_tpu_info()
    if tile_n is None:
        tile_n = size_n
    sublane = gmm._get_lhs_sublane_size(lhs.dtype, rows)
    physical_rows = gmm.align_to(rows + tile_m, 128)
    lhs = jnp.pad(lhs, ((0, gmm.align_to(rows, sublane) - rows), (0, 0)))
    ids = jnp.pad(sorted_token_ids, (0, physical_rows - rows),
                  constant_values=-1)
    weights = jnp.pad(sorted_weights.astype(jnp.float32),
                      (0, physical_rows - rows))
    group_offset = jnp.zeros((1, ), dtype=jnp.int32)
    vmem_limit = int(info.vmem_capacity_bytes * 0.9)
    cfgs = gmm.make_gmm_configs(lhs,
                                rhs,
                                rhs_scale,
                                None,
                                group_sizes,
                                group_offset,
                                tile_info=gmm.TileSizes(tile_m=tile_m,
                                                        tile_k=size_k,
                                                        tile_n=tile_n,
                                                        bucket_base=128),
                                vmem_limit_bytes=vmem_limit,
                                out_dtype=jnp.bfloat16,
                                acc_dtype=None,
                                maybe_quantize_lhs=True,
                                zero_initialize=False,
                                fuse_act=None)
    max_gm = experts + pl.cdiv(lhs.shape[0], tile_m) - 1
    scratch = (pltpu.VMEM((sublane, tile_n), jnp.bfloat16),
               pltpu.VMEM((tile_m, tile_n), cfgs.acc_dtype),
               gmm.MetadataRef(pltpu.SMEM((max_gm, ), jnp.int32),
                               pltpu.SMEM((max_gm + 1, ), jnp.int32)),
               pltpu.VMEM((physical_rows // sublane, sublane, tile_n),
                          jnp.bfloat16),
               pltpu.VMEM((num_tokens, tile_n), jnp.float32),
               pltpu.VMEM((num_tokens, tile_n),
                          jnp.bfloat16), pltpu.SemaphoreType.DMA)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_kernel, cfgs=cfgs, num_tokens=num_tokens),
        out_shape=jax.ShapeDtypeStruct((num_tokens, size_n), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(size_n // tile_n, ),
            in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),
                      pl.BlockSpec(memory_space=pltpu.VMEM), hbm,
                      gmm.WeightsRef(weight=hbm, scale=hbm, bias=None)),
            out_specs=hbm,
            scratch_shapes=scratch),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
        name=f"gmm2-owner-vmem-b{num_tokens}-tm{tile_m}-tn{tile_n}",
    )(group_sizes, group_offset, ids, weights, lhs,
      gmm.WeightsRef(rhs, rhs_scale, None))
