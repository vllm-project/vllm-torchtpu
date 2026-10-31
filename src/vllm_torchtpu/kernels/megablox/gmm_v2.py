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

import dataclasses
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.quantized_rows import (
    ROWS_PER_WORD,
    SCALE_COLS,
    SOURCE_DTYPE,
    decode_scales,
    quantize_block,
    quantized_row_width,
)

# Util.


def swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""

    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def silu_and_mul_with_clamp(
    gate: jax.Array, up: jax.Array, limit: float = 10.0
) -> jax.Array:
    """Activation used in some models DeepSeek V4."""
    # The limit value is from DSV4's config.
    # TODO: pass limit from model config, instead of hardcoding here.
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    return jax.nn.silu(gate) * up


def situ_and_mul(
    gate: jax.Array, up: jax.Array, beta: float, linear_beta: float | None
) -> jax.Array:
    gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * jnp.tanh(up / linear_beta)
    return gate * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Interleaves two arrays along lane dim at zero-cost."""
    assert lhs.shape == rhs.shape
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = lhs.shape[-1] // chunk_size
    lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
    rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
    interleaved = []
    for i in range(num_chunks):
        interleaved += [lhs_chunks[i], rhs_chunks[i]]
    return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Deinterleaves an array along lane dim at zero-cost."""
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = val.shape[-1] // chunk_size
    chunks = jnp.split(val, num_chunks, axis=-1)
    lhs = jnp.concat(chunks[0::2], axis=-1)
    rhs = jnp.concat(chunks[1::2], axis=-1)
    return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Applies a fused activation function to the accumulator.

    This function is used when an activation function is fused with the matrix
    multiplication. The input accumulator `acc` is expected to contain
    concatenated results for both the 'gate' and 'up' projections.

    Args:
        acc: The accumulator array, with the last dimension being 2 * tile_n.
        fuse_act: The name of the activation function to apply.

    Returns:
        The result of applying the activation function.

    Raises:
        NotImplementedError: If an unsupported `fuse_act` is provided.
    """

    if fuse_act is None:
        return acc

    acc_gate, acc_up = deinterleave_lane(acc)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "gelu_tanh":
            return jax.nn.gelu(acc_gate, approximate=True) * acc_up
        case "swigluoai":
            return swigluoai(acc_gate, acc_up)
        case "silu_and_mul_with_clamp":
            return silu_and_mul_with_clamp(acc_gate, acc_up)
        case str() if fuse_act.startswith("situ:"):
            _, beta, linear_beta = fuse_act.split(":")
            linear_beta = None if linear_beta == "none" else float(linear_beta)
            return situ_and_mul(acc_gate, acc_up, float(beta), linear_beta)
        case _:
            raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    return pl.cdiv(x, a) * a


def _get_lhs_sublane_size(dtype: jnp.dtype, size_m: int) -> int:
    """Sublane block size the kernel processes lhs rows in."""
    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(dtype)
    size_lhs_sublane = min(size_lhs_sublane, size_m)
    return size_lhs_sublane


def get_packing_factor(storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None) -> int:
    if quant_dtype is None or quant_dtype == storage_dtype:
        return 1
    storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
    quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
    packing_factor, remainder = divmod(storage_bits, quant_bits)
    if remainder != 0:
        raise ValueError(
            f"Storage dtype {storage_dtype} is not divisible by "
            f"quant dtype {quant_dtype}"
        )
    return packing_factor


# Define data classes.


class RhsRef(ABC):
    """Abstract class that defines interfaces for rhs values."""

    @abstractmethod
    def get_weight(self) -> jax.Array: ...

    @abstractmethod
    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        """Returns scale array, optionally replicated across sublanes."""
        ...

    @abstractmethod
    def get_bias(self) -> jax.Array: ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Dataclass for a single weights."""

    weight: Any
    scale: Any | None
    bias: Any | None

    def get_weight(self) -> jax.Array:
        return self.weight[...]

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        assert self.scale is not None
        if replicate_size is not None:
            # Perform zero-stride load for efficient broadcasting across sublanes.
            return self.scale[:, pl.ds(0, replicate_size, 0), :]
        return self.scale[...]

    def get_bias(self) -> jax.Array:
        assert self.bias is not None
        return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """Dataclass for gate and up weights used in fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self) -> jax.Array:
        w_gate = self.gate.get_weight()
        w_up = self.up.get_weight()
        return interleave_lane(w_gate, w_up)

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        s_gate = self.gate.get_scale(replicate_size)
        s_up = self.up.get_scale(replicate_size)
        return interleave_lane(s_gate, s_up)

    def get_bias(self) -> jax.Array:
        b_gate = self.gate.get_bias()
        b_up = self.up.get_bias()
        return interleave_lane(b_gate, b_up)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
    tile_m: int
    tile_k: int
    tile_n: int
    bucket_base: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    quant_dtype: jnp.dtype | None
    quant_block_size: int
    dtype: jnp.dtype
    has_bias: bool = False
    has_scale: bool = False
    # The kernel reads lhs already quantized, as quantized_rows (lhs only).
    quantized_ahead: bool = False

    @property
    def should_unpack(self) -> bool:
        """True if rhs weights are sub-byte (e.g. INT4) packed in
        INT32/UINT32 carriers."""
        return get_packing_factor(self.dtype, self.quant_dtype) > 1

    @property
    def should_dequantize_before_matmul(self) -> bool:
        """Dequantize rhs before matmul if block size limits MXU utilization."""
        if not self.has_scale:
            return False
        assert self.quant_block_size is not None
        mxu_size = pltpu.get_tpu_info().mxu_column_size
        return self.quant_block_size < mxu_size

    @property
    def should_dequantize_after_matmul(self) -> bool:
        return self.has_scale and not self.should_dequantize_before_matmul

    @property
    def should_quantize(self) -> bool:
        if self.quant_dtype is None:
            return False
        return self.quant_dtype != self.dtype


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool
    fuse_act: str | None

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        if self.fuse_act is None:
            return self.dims.size_n
        else:
            return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
        # since a single quant block could be shared along multiple k tile.
        k_row = k_id * self.cfgs.tiles.tile_k
        b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> tuple[tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (None, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


# Define kernels.


def tile_matmul(
    tiled_lhs: jax.Array,  # [bucket_m, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    *,
    cfgs: GmmConfigs,
    is_last_k_step: bool,
    lhs_scales: list[jax.Array] | None = None,
) -> jax.Array:
    """Returns the [bucket_m, tile_n] product of one k tile in cfgs.acc_dtype.

    With ``lhs_scales`` (one [bucket_m, 1] scale per quantization block of the
    tile), tiled_lhs is already quantized (quantized_rows.quantize_rows).
    """
    bucket_m = tiled_lhs.shape[0]
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size

    # Step 1: Input pre-processing.
    tiled_rhs = tiled_rhs_ref.get_weight()

    # This should only be taken in the case where we don't requantize
    # the scales and thus we need to dequantize inside VMEM to avoid small
    # contracting dimmensions
    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
        tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(
            cfgs.lhs_cfgs.dtype
        )
        num_blocks = cfgs.num_quant_blocks_per_tile_k
        tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
            num_blocks, rhs_qbs, rhs_tile_n
        )
        tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
        tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
        rhs_qbs = cfgs.tiles.tile_k

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
        mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
        tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    # Step 2: Matmul.
    acc_list = []
    if not cfgs.lhs_cfgs.should_quantize:
        # Unquantized matmul path.
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n

            acc_n = jnp.zeros((bucket_m, col_size), dtype=cfgs.acc_dtype)
            for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)

                block_acc = jnp.matmul(
                    tiled_lhs[:, start_k:end_k],
                    tiled_rhs[start_k:end_k, start_n:end_n],
                    preferred_element_type=jnp.float32,
                ).astype(cfgs.acc_dtype)

                if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    rhs_scale_replicated = tiled_rhs_ref.get_scale(
                        replicate_size=bucket_m
                    )[b_id, :, start_n : start_n + col_size]
                    block_acc *= rhs_scale_replicated.astype(cfgs.acc_dtype)

                acc_n += block_acc
            acc_list.append(acc_n)
    else:
        # Quantized matmul path.
        lhs_q_dtype = cast(jnp.dtype, cfgs.lhs_cfgs.quant_dtype)
        q_block_size = cfgs.lhs_cfgs.quant_block_size

        if jnp.issubdtype(lhs_q_dtype, jnp.floating):
            preferred_element_type = jnp.float32
        else:
            preferred_element_type = jnp.int32

        # Without n outer loop, result of quantized matmul becomes available only
        # at the last iteration of the loop. This means [tile_m, tile_n] value
        # needs to be stored until the last iteration. By adding n outer loop,
        # result of [tile_m, mxu_size] becomes available at the end of every k
        # inner loop which can be used to pipeline subsequent VPU or VST ops with
        # MXU ops for the next [tile_m, mxu_size].
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n

            acc_n = jnp.zeros((bucket_m, col_size), dtype=cfgs.acc_dtype)
            for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                block_lhs = tiled_lhs[:, start_k:end_k]
                block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                if lhs_scales is None:
                    # Perform lhs quantization. Note that for every block_lhs,
                    # same computation will be performed tiles_n//mxu_size times.
                    # But we can let compiler perform CSE and avoid recomputation.
                    block_lhs_q, block_scale = quantize_block(block_lhs, lhs_q_dtype)
                else:
                    block_lhs_q = block_lhs
                    block_scale = lhs_scales[start_k // q_block_size]

                # Unlike unquantized path, compiler may not perform implicit type
                # conversion due to numeric concerns. As this can cause unsupported
                # matmul error, explicit type conversion is performed.
                if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
                    block_rhs = block_rhs.astype(lhs_q_dtype)

                block_acc = jnp.matmul(
                    block_lhs_q,
                    block_rhs,
                    preferred_element_type=preferred_element_type,
                ).astype(cfgs.acc_dtype)

                block_acc *= block_scale.astype(cfgs.acc_dtype)

                # Apply rhs subchannel scale per quant block.
                if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    rhs_scale_replicated = tiled_rhs_ref.get_scale(
                        replicate_size=bucket_m
                    )[b_id, :, start_n : start_n + col_size]
                    block_acc *= rhs_scale_replicated.astype(cfgs.acc_dtype)

                acc_n += block_acc
            acc_list.append(acc_n)
    return jnp.concatenate(acc_list, axis=1)


def finalize_tile(
    acc: jax.Array,  # [bucket_m, tile_n] or [bucket_m, 2 * tile_n]
    tiled_rhs_ref: RhsRef,
    m_start_local: jax.Array,
    m_end_local: jax.Array,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Adds the bias, applies the fused activation and zeroes rows outside
    [m_start_local, m_end_local)."""
    if cfgs.rhs_cfgs.has_bias:
        tiled_rhs_bias = tiled_rhs_ref.get_bias()
        acc += tiled_rhs_bias.astype(acc.dtype)

    acc = apply_act_fn(acc, cfgs.fuse_act)

    # Mask out rows that does not belong to the current group.
    iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
    mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
    acc_masked = jnp.where(mask, acc, 0)
    return acc_masked


def store_tile(
    acc_masked: jax.Array,  # [bucket_m, tile_n]
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    gm_id: jax.Array,
    m_end_local: jax.Array,
    *,
    cfgs: GmmConfigs,
) -> None:
    """Writes a finished tile, merging the sublane rows it shares with the
    previous gm tile through partial_out_ref."""
    acc_m = acc_masked.shape[0]

    # Write the final output to the output ref.
    tiled_out_2d_ref = tiled_out_ref.reshape(-1, cfgs.tiles.tile_n)
    tiled_out_2d_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype)

    # If this is the first tile for grid[n_id, :, :], we initialize the
    # partial out to zeros. Otherwise, partial out from last tile of
    # grid[n_id-1, :, :] can be used and cause numeric issues.
    partial_out_zeros = jnp.zeros_like(partial_out_ref)

    # Accumulate the partial output from the previous step.
    tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])

    # Consider following case where size_lhs_sublane = 4, number denotes group
    # id and | denotes boundaries between sublanes:
    # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
    #
    # Assuming group id of current step is 1, current step will not completely
    # fill size_lhs_sublane rows and will be revisited at the next step. By
    # storing the partial rows into the partial_out_ref, the next step can
    # read them and accumulate to them.  Additionally, for group id of 2,
    # since it completely fills the size_lhs_sublane rows, we need to zero out
    # partial_out_ref to avoid numeric error for group 3.
    last_row = m_end_local // cfgs.dims.size_lhs_sublane
    partial_out_ref[...] = jnp.where(
        m_end_local % cfgs.dims.size_lhs_sublane == 0,
        partial_out_zeros,
        tiled_out_ref[last_row],
    )


def inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

    tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
    Additionally, m_start and m_end does not have to align with tile boundaries
    [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
    [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
    invalid data and needs to be masked out.

    Args:
        tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
        tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
            g_id is the group associated with lhs[m_start:m_end, :]
        tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
        partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
            Will be initialized to zero if this is first tile for grid[n_id, :, :].
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """

    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int):
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
        acc = tile_matmul(
            tiled_lhs, tiled_rhs_ref, cfgs=cfgs, is_last_k_step=is_last_k_step
        )

        # Step 3: Output post-processing.
        if not is_first_k_step:
            acc = jnp.pad(acc, ((cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
            acc += acc_ref[...]
        acc_m = acc.shape[0]

        if is_last_k_step:
            acc_masked = finalize_tile(
                acc, tiled_rhs_ref, m_start_local, m_end_local, cfgs=cfgs
            )
            store_tile(
                acc_masked,
                tiled_out_ref,
                partial_out_ref,
                gm_id,
                m_end_local,
                cfgs=cfgs,
            )
        else:
            acc_ref[:acc_m] = acc

    def run_matmul_step(bucket_idx: int):
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)

        @jax.named_scope(f"bm{bucket_m}_first_last")
        def matmul_first_last():
            _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_first")
        def matmul_first():
            _matmul(is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_mid")
        def matmul_mid():
            _matmul(is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m)

        @jax.named_scope(f"bm{bucket_m}_last")
        def matmul_last():
            _matmul(is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m)

        num_k = pl.num_programs(2)
        k_id = pl.program_id(2)
        is_first_k_step = k_id == 0
        is_last_k_step = k_id == (num_k - 1)

        if bucket_m == cfgs.tiles.tile_m:
            lax.cond(
                is_first_k_step,
                lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
                lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
            )
        else:
            # partial m is only invoked at last matmul.
            lax.cond(is_first_k_step, matmul_first_last, matmul_last)

    branches = []
    for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base):
        branches.append(functools.partial(run_matmul_step, bucket_idx=bucket_idx))
    bucket_idx = m_end_local // cfgs.tiles.bucket_base
    lax.switch(bucket_idx, branches)


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m offsets
            for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(
            cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset
        )

        metadata_ref.gm_id_to_group_id[tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
    """Zero out output rows that are not used in the computation."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    assert num_lanes == zero_ref.shape[-1]
    zero_ref[...] = jnp.zeros_like(zero_ref)

    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    row_size = zero_dma.shape[0]

    compute_start = metadata_ref.gm_id_to_m_offset[0]
    compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

    left_zero_start = 0
    left_zero_end = compute_start // dims.size_lhs_sublane
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = pl.cdiv(left_zero_size, row_size)

    right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    right_zero_size = right_zero_end - right_zero_start
    right_num_loops = pl.cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        # Static loop. Will be unrolled during compile time.
        for n_start in range(0, out_dma.shape[-1], num_lanes):
            n_end = n_start + num_lanes
            pltpu.make_async_copy(
                src_ref=zero_dma.at[pl.ds(0, dma_size)],
                dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
                sem=semaphore_ref.at[0],
            ).start(priority=1)

        return zero_size + dma_size

    @jax.named_scope("left_fill_zero")
    def left_fill_zero(i, zero_size):
        return fill_zero(i, zero_size, start=left_zero_start, end=left_zero_end)

    @jax.named_scope("right_fill_zero")
    def right_fill_zero(i, zero_size):
        return fill_zero(i, zero_size, start=right_zero_start, end=right_zero_end)

    zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
    zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
    return zero_size


def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    pltpu.make_async_copy(
        src_ref=out_dma.at[pl.ds(0, zero_size)],
        dst_ref=out_dma.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


def prepare_rhs(
    rhs_ref: WeightsRef, rhs_spec: WeightsRef, *, cfgs: GmmConfigs
) -> tuple[RhsRef, RhsRef]:
    """Unpacks sub-byte rhs weights and, with a fused activation, splits the
    rhs into its gate and up halves along n."""
    if cfgs.rhs_cfgs.should_unpack:
        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    if cfgs.fuse_act is None:
        return rhs_ref, rhs_spec

    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
    return (
        FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref),
        FusedWeightsRef(gate=rhs_spec, up=rhs_spec),
    )


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
):
    """Entry point for GMM kernel.

    Computes metadata to determine which rows of lhs needs processing and how
    they will be tiled. And then, invoke inner kernel using metadata.

    Uses the following notation:
    - g: rhs group dimension
    - m: Batch dimension
    - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
        of tile_m. Skips over empty groups and accounts for revisited tiles.
    - k: in dimension
    - n: out dimension

    Args:
        lhs_group_sizes_ref: Reference to the group sizes of lhs.
        group_offset_ref: Reference to the group offset.
        lhs_ref: Reference to the lhs.
        rhs_ref: Reference to the rhs.
        out_ref: Reference to the out.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        zero_ref: Scratch memory for storing zero values used in initialization.
        semaphore_ref: Semaphore for zero initialization DMAs.
        cfgs: GmmConfigs.
    """

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    # Fill metadata buffer and return number of group & m interations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            cast(jax.Array, zero_ref),
            cast(jax.Array, semaphore_ref),
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )

    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    rhs_in, rhs_spec_in = prepare_rhs(rhs_ref, rhs_spec, cfgs=cfgs)

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_spec_in),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_in, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, cast(jax.Array, semaphore_ref), zero_size, dims=cfgs.dims)


def fused_mlp_inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, size_k]
    tiled_rhs1_ref: RhsRef,  # gate/up: [size_k, 2 * size_inter]
    tiled_rhs2_ref: RhsRef,  # down: [size_inter, size_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, size_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, size_n]
    inter_ref: jax.Array,  # [tile_m, size_inter]
    metadata_ref: MetadataRef,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """Runs both legs of one gm tile: gate/up with its fused activation into
    inter_ref, then down from inter_ref into the output tile.

    Each leg is a single k and n tile and does what inner_kernel does on its
    one-step path, so inter_ref holds the rows gmm_v2's gate/up call would
    write to HBM and its down call would read back.
    """

    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs1.dims.size_lhs_sublane

    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def run_tile(bucket_m: int):
        lhs_scales = None
        if cfgs1.lhs_cfgs.quantized_ahead:
            size_k = cfgs1.tiles.tile_k
            rows_ref = tiled_lhs_ref.reshape(-1, quantized_row_width(size_k))
            tiled_lhs = rows_ref[:bucket_m, :size_k]
            lhs_scales = decode_scales(
                rows_ref[:bucket_m, size_k:],
                size_k // cfgs1.lhs_cfgs.quant_block_size,
            )
        else:
            tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs1.tiles.tile_k)[:bucket_m]
        acc = tile_matmul(
            tiled_lhs,
            tiled_rhs1_ref,
            cfgs=cfgs1,
            is_last_k_step=True,
            lhs_scales=lhs_scales,
        )
        inter = finalize_tile(
            acc, tiled_rhs1_ref, m_start_local, m_end_local, cfgs=cfgs1
        )
        inter_ref[:bucket_m] = inter.astype(inter_ref.dtype)

        acc = tile_matmul(
            inter_ref[:bucket_m], tiled_rhs2_ref, cfgs=cfgs2, is_last_k_step=True
        )
        acc_masked = finalize_tile(
            acc, tiled_rhs2_ref, m_start_local, m_end_local, cfgs=cfgs2
        )
        store_tile(
            acc_masked,
            tiled_out_ref,
            partial_out_ref,
            gm_id,
            m_end_local,
            cfgs=cfgs2,
        )

    branches = []
    for bucket_idx in range(cfgs1.tiles.tile_m // cfgs1.tiles.bucket_base):
        bucket_m = cfgs1.tiles.bucket_base * (bucket_idx + 1)
        branches.append(
            jax.named_scope(f"bm{bucket_m}")(functools.partial(run_tile, bucket_m))
        )
    lax.switch(m_end_local // cfgs1.tiles.bucket_base, branches)


# Token rows one DMA wait covers before the last few, a multiple of 8.
TOKEN_ROW_WAIT = 8
# Most SMEM token_rows' word ids (one per row) and part masks (token_row_parts)
# may take, leaving headroom for the kernel's other scalars; a larger batch
# keeps its rows gathered.
TOKEN_ROWS_SMEM_BUDGET_BYTES = 512 * 1024


def token_row_parts(row_ids: jax.Array, sublane: int, packing: int) -> jax.Array:
    """TokenRows.parts_ref: which part of its word row each row is.

    Row j of sublane block b is part ``row_ids[b * sublane + j] % packing``,
    held in log2(packing) bits from bit ``j * log2(packing)`` of the block's
    ``token_row_part_words`` int32 words.
    """
    part_bits = packing.bit_length() - 1
    rows_per_word = min(32 // part_bits, sublane)
    shifts = jnp.arange(rows_per_word, dtype=jnp.int32) * part_bits
    parts = (row_ids & (packing - 1)).reshape(-1, rows_per_word) << shifts
    return jnp.sum(parts, axis=1)


def token_row_part_words(sublane: int, packing: int) -> int:
    return -(-sublane * (packing.bit_length() - 1) // 32)


def token_rows_fit_smem(size_m: int, sublane: int, packing: int) -> bool:
    words_per_block = token_row_part_words(sublane, packing)
    words = size_m + size_m // sublane * words_per_block
    return 4 * words <= TOKEN_ROWS_SMEM_BUDGET_BYTES


@dataclasses.dataclass(frozen=True)
class TokenRows:
    """Where gmm_v2_fused_mlp reads lhs rows past its head buffer.

    Row r >= head_rows of the batch is row ids[r] of the token rows. Rows of
    a narrower dtype sit in 32-bit words in groups of ``packing`` (row
    packing * w + j in part j of word row w: bf16 pairs, fp8 quads), so a
    token row arrives as the word row holding it, in ``stage_ref``, and its
    part is picked out into the tile's lhs buffer.
    """

    word_ids_ref: jax.Array  # int32[size_m], SMEM: ids[r] // packing
    parts_ref: jax.Array  # int32, SMEM: token_row_parts
    head_ref: jax.Array  # [head_rows // sublane, sublane, width], HBM
    words_ref: jax.Array  # uint32[num_tokens // packing, width], HBM
    lhs_bufs: jax.Array  # [2, tile_m // sublane, sublane, width], VMEM
    # uint32[packing, tile_m // sublane, sublane // packing, width]: part j
    # holds tile rows packing * w + j
    stage_ref: jax.Array
    sems: jax.Array  # DMA[2]: head rows, token rows

    @property
    def head_rows(self) -> int:
        return self.head_ref.shape[0] * self.head_ref.shape[1]

    @property
    def sublane(self) -> int:
        return self.head_ref.shape[1]

    @property
    def packing(self) -> int:
        return self.stage_ref.shape[0]

    def tile(self, metadata_ref: MetadataRef, gm_id: jax.Array):
        """First sublane-block row, end, head blocks and first token row of
        gm tile gm_id.

        Token rows start on a sublane block: rows of a block before the
        tile's own are fetched too and masked out like any row outside its
        group, so each block's DMAs land at fixed offsets in the stage.
        """
        m_start = metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
        m_offset = m_start - m_start % self.sublane
        head_end = jnp.minimum(m_end, self.head_rows)
        head_blocks = pl.cdiv(jnp.maximum(head_end - m_offset, 0), self.sublane)
        return m_offset, m_end, head_blocks, jnp.maximum(m_offset, self.head_rows)

    def start(self, metadata_ref: MetadataRef, gm_id: jax.Array, slot: jax.Array):
        """Starts the DMAs of gm tile gm_id's lhs rows into buffer slot."""
        m_offset, m_end, head_blocks, first = self.tile(metadata_ref, gm_id)

        @pl.when(head_blocks > 0)
        def _():
            pltpu.make_async_copy(
                self.head_ref.at[pl.ds(m_offset // self.sublane, head_blocks)],
                self.lhs_bufs.at[slot, pl.ds(0, head_blocks)],
                self.sems.at[0],
            ).start()

        # Unrolled over the tile's sublane blocks so each DMA's stage offset
        # is a constant; a row past the tile only predicates its DMA off.
        first_block = (first - m_offset) // self.sublane
        end_block = pl.cdiv(m_end - m_offset, self.sublane)
        for b in range(self.stage_ref.shape[1]):

            @pl.when((first_block <= b) & (b < end_block))
            def _(b=b):
                row0 = m_offset + b * self.sublane
                rows_left = m_end - row0
                for u in range(self.sublane):
                    word = self.word_ids_ref[row0 + u]

                    @pl.when(u < rows_left)
                    def _(b=b, u=u, word=word):
                        pltpu.make_async_copy(
                            self.words_ref.at[word],
                            self.stage_ref.at[u % self.packing, b, u // self.packing],
                            self.sems.at[1],
                        ).start()

    def wait(self, metadata_ref: MetadataRef, gm_id: jax.Array, slot: jax.Array):
        _, m_end, head_blocks, first = self.tile(metadata_ref, gm_id)

        @pl.when(head_blocks > 0)
        def _():
            pltpu.make_async_copy(
                self.head_ref.at[pl.ds(0, head_blocks)],
                self.lhs_bufs.at[slot, pl.ds(0, head_blocks)],
                self.sems.at[0],
            ).wait()

        # DMA semaphores count bytes, so a wait for n word rows covers any n.
        def wait_rows(rows):
            ref = self.stage_ref.at[0, 0, pl.ds(0, rows)]

            def body(i, carry):
                pltpu.make_async_copy(ref, ref, self.sems.at[1]).wait()
                return carry

            return body

        count = jnp.maximum(m_end - first, 0)
        lax.fori_loop(0, count // TOKEN_ROW_WAIT, wait_rows(TOKEN_ROW_WAIT), 0)
        lax.fori_loop(0, count % TOKEN_ROW_WAIT, wait_rows(1), 0)

    def unpack(self, metadata_ref: MetadataRef, gm_id: jax.Array, slot: jax.Array):
        """Writes gm tile gm_id's token rows into buffer slot, as packed words.

        Works on whole sublane blocks: rows of a block outside the tile's
        token rows get stale words, and the tile masks them out like any row
        outside its group. The head rows end on a block boundary.
        """
        m_offset, m_end, _, first = self.tile(metadata_ref, gm_id)
        lhs_words = self.lhs_bufs.at[slot].bitcast(jnp.uint32)
        packing = self.packing
        groups = self.sublane // packing
        part_bits = packing.bit_length() - 1
        item_bits = 32 // packing
        item_mask = (1 << item_bits) - 1
        words_per_block = token_row_part_words(self.sublane, packing)
        num_lanes = pltpu.get_tpu_info().num_lanes
        width = lhs_words.shape[-1]
        # Sublane w of a block's part j stage words holds row packing * w + j.
        row0 = packing * lax.broadcasted_iota(jnp.uint32, (groups, num_lanes), 0)
        first_tile_block = m_offset // self.sublane

        def block(b, carry):
            words = []
            for h in range(words_per_block):
                block_word = first_tile_block + b
                if words_per_block > 1:
                    block_word = block_word * words_per_block + h
                word = self.parts_ref[block_word].astype(jnp.uint32)
                words.append(jnp.full((groups, num_lanes), word, jnp.uint32))
            shifts = []
            for j in range(packing):
                bit = row0 + j if j else row0
                if part_bits > 1:
                    bit = bit * part_bits
                word = words[0]
                for h in range(1, words_per_block):
                    word = jnp.where(bit >= 32 * h, words[h], word)
                if words_per_block > 1:
                    bit = bit & 31
                part = (word >> bit) & (packing - 1)
                shifts.append(part << (item_bits.bit_length() - 1))
            for c in range(0, width, num_lanes):
                cols = pl.ds(c, num_lanes)
                items = [
                    self.stage_ref[j, b, :, cols] >> shifts[j] for j in range(packing)
                ]
                out = None
                for j, item in enumerate(items):
                    if j < packing - 1:
                        item = item & item_mask
                    if j:
                        item = item << (item_bits * j)
                    out = item if out is None else out | item
                lhs_words[b, :, cols] = out
            return carry

        first_block = (first - m_offset) // self.sublane
        end_block = jnp.where(
            m_end > first, pl.cdiv(m_end - m_offset, self.sublane), first_block
        )
        lax.fori_loop(first_block, end_block, block, 0)


def fused_mlp_token_rows_kernel(
    tiled_rhs1_ref: RhsRef,
    tiled_rhs2_ref: RhsRef,
    tiled_out_ref: jax.Array,
    partial_out_ref: jax.Array,
    inter_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    rows: TokenRows,
    num_gm: jax.Array,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """fused_mlp_inner_kernel with its lhs tile fetched by TokenRows.

    Tile gm_id's rows were started into buffer gm_id % 2 one step ahead; the
    next tile's start once this one's token rows have left the stage.
    """
    gm_id = pl.program_id(1)
    slot = gm_id % 2
    rows.wait(metadata_ref, gm_id, slot)
    rows.unpack(metadata_ref, gm_id, slot)

    @pl.when(gm_id + 1 < num_gm)
    def _():
        rows.start(metadata_ref, gm_id + 1, 1 - slot)

    fused_mlp_inner_kernel(
        rows.lhs_bufs.at[slot],
        tiled_rhs1_ref,
        tiled_rhs2_ref,
        tiled_out_ref,
        partial_out_ref,
        inter_ref,
        metadata_ref,
        cfgs1=cfgs1,
        cfgs2=cfgs2,
    )


def fused_mlp_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    *refs,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    token_rows: bool,
):
    """Entry point for the fused gate/up -> down kernel.

    One pipeline over the gm tiles streams each group's gate/up and down
    weights together. Rows outside every group are left unwritten, as gmm_v2
    leaves them with zero_initialize=False.

    ``refs`` are, in order: TokenRows' word ids and parts (scalar prefetch,
    with token_rows); lhs [size_m, size_k], or its first rows with
    token_rows; the token rows [num_tokens, size_k] (with token_rows; both
    as int32 words of quantized_rows rows when gate/up's lhs is quantized
    ahead);
    gate/up WeightsRef [size_group, size_k, 2 * size_inter] and down
    WeightsRef [size_group, size_inter, size_n]; out_ref [size_m, size_n];
    scratch partial_out_ref [size_lhs_sublane, size_n], inter_ref
    [tile_m, size_inter] and metadata_ref, then with token_rows the lhs
    buffers, the stage and two DMA semaphores of TokenRows.
    """
    if token_rows:
        word_ids_ref, parts_ref, lhs_ref, words_ref, *refs = refs
        if cfgs1.lhs_cfgs.quantized_ahead:
            # Quantized rows arrive as the int32 words holding them.
            lhs_ref = lhs_ref.bitcast(cfgs1.lhs_cfgs.quant_dtype)
    else:
        lhs_ref, *refs = refs
    rhs1_ref, rhs2_ref, out_ref, partial_out_ref, inter_ref, metadata_ref, *refs = refs

    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs1,
    )

    (lhs_spec, rhs1_spec), _ = generate_block_specs(metadata_ref, cfgs1)
    (_, rhs2_spec), out_spec = generate_block_specs(metadata_ref, cfgs2)
    rhs1_in, rhs1_spec_in = prepare_rhs(rhs1_ref, rhs1_spec, cfgs=cfgs1)
    rhs2_in, rhs2_spec_in = prepare_rhs(rhs2_ref, rhs2_spec, cfgs=cfgs2)

    sublane = cfgs1.dims.size_lhs_sublane
    lhs_in = lhs_ref.reshape(-1, sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, inter_ref, metadata_ref]
    if not token_rows:
        pipeline_fn = pltpu.emit_pipeline(
            functools.partial(fused_mlp_inner_kernel, cfgs1=cfgs1, cfgs2=cfgs2),
            grid=(1, num_gm, 1),
            in_specs=(lhs_spec, rhs1_spec_in, rhs2_spec_in),
            out_specs=out_spec,
        )
        pipeline_fn(lhs_in, rhs1_in, rhs2_in, out_in, scratches=scratches)
        return

    lhs_bufs, stage_ref, sems = refs
    rows = TokenRows(
        word_ids_ref=word_ids_ref,
        parts_ref=parts_ref,
        head_ref=lhs_in,
        words_ref=words_ref.bitcast(jnp.uint32),
        lhs_bufs=lhs_bufs,
        stage_ref=stage_ref,
        sems=sems,
    )

    @pl.when(num_gm > 0)
    def _():
        rows.start(metadata_ref, 0, 0)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            fused_mlp_token_rows_kernel,
            rows=rows,
            num_gm=num_gm,
            cfgs1=cfgs1,
            cfgs2=cfgs2,
        ),
        grid=(1, num_gm, 1),
        in_specs=(rhs1_spec_in, rhs2_spec_in),
        out_specs=out_spec,
    )
    pipeline_fn(rhs1_in, rhs2_in, out_in, scratches=scratches)


def gmm_vmem_estimate(
    tm: int,
    tn: int,
    tk: int,
    *,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    fuse_act_factor: int,
) -> int:
    """Estimates the VMEM bytes of one gmm_v2 pipeline at the given tiles."""
    lhs_bits = jax.dtypes.itemsize_bits(lhs_cfgs.dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_cfgs.quant_dtype or rhs_cfgs.dtype)

    # 1. LHS tile (double-buffered HBM load)
    lhs_tile_bytes = lhs_bits // 8
    lhs_vmem = 2 * tm * tk * lhs_tile_bytes
    # If LHS is quantized on-the-fly, we need an extra single-buffered cast
    # buffer in VMEM.
    if lhs_cfgs.should_quantize:
        lhs_quant_bits = jax.dtypes.itemsize_bits(cast(jnp.dtype, lhs_cfgs.quant_dtype))
        lhs_vmem += tm * tk * (lhs_quant_bits // 8)

    # 2. RHS tile (triple-buffered, includes scale and bias if present)
    # If fuse_act is enabled, we have both gate and up weights,
    # so RHS memory is doubled.
    rhs_weight_vmem = tk * tn * rhs_bits // 8
    rhs_scale_vmem = 0
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
        num_quant_blocks_per_tile_k = pl.cdiv(tk, rhs_cfgs.quant_block_size)
        rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
    rhs_bias_vmem = 0
    if rhs_cfgs.has_bias:
        rhs_bias_vmem = tn * 4
    rhs_vmem = fuse_act_factor * (
        3 * rhs_weight_vmem + 2 * rhs_scale_vmem + 2 * rhs_bias_vmem
    )

    # 3. Accumulator
    acc_cols = fuse_act_factor * tn
    acc_dtype_bytes = 2 if lhs_cfgs.quant_dtype is not None else 4
    acc_vmem = tm * acc_cols * acc_dtype_bytes

    # 4. Output tile (double-buffered)
    out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    out_vmem = 2 * tm * tn * out_dtype_bytes

    return lhs_vmem + rhs_vmem + acc_vmem + out_vmem


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if lower bitwidth is used, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m // rhs_mod
    if lhs_cfgs.should_quantize:
        tile_m *= 2
    tile_m = min(tile_m, dims.size_m)

    # To avoid stalling MXU, we add some buffer room where tile_n cannot go
    # smaller than 2x of mxu_column_size.
    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    size_n_per_rhs = dims.size_n
    fuse_act_factor = 1
    if fuse_act is not None:
        # When computing activation function, rhs is concatenated along dim n.
        fuse_act_factor = 2
        size_n_per_rhs //= fuse_act_factor
        tile_n_limit //= fuse_act_factor

    def _is_tile_k_quant_block_compatible(tk: int) -> bool:
        return (
            tk % rhs_cfgs.quant_block_size == 0 or rhs_cfgs.quant_block_size % tk == 0
        )

    # Initialize tile_k and tile_n to their maximum valid values.
    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    _gmm_vmem_estimate = functools.partial(
        gmm_vmem_estimate,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        fuse_act_factor=fuse_act_factor,
    )

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # to fit the tensors into vmem by only adjusting tile_n.

    # Decrease tile_n until total memory fits in vmem limit.
    while (
        _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
        and tile_n > tile_n_limit
    ):
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    # If decreasing tile_n is no longer possible, we decrease tile_k instead.
    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

        # Decrease tile_k until total memory fits in vmem limit and tile_k is valid.
        while (
            _gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
            or not _is_tile_k_quant_block_compatible(tile_k)
        ) and tile_k > num_lanes:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        final_estimate = _gmm_vmem_estimate(tile_m, tile_n, tile_k)
        raise ValueError(
            f"Could not find valid tile sizes for {dims=} and"
            f" {final_estimate=} (limit: {vmem_limit_bytes})."
        )

    # TODO(alynie, kyuyeunk): max number of bucket was choosen empiirically.
    # Revisit the value to be based on number of instruction memory size.
    max_num_buckets = 4
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
        new_tile_m = tile_m + bucket_base
        if new_tile_m > dims.size_m:
            break
        if _gmm_vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
            break
        tile_m = new_tile_m

    return TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base
    )


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    packing_factor: int = 1,
) -> Dimensions:
    """Validates the inputs for the GMM kernel."""

    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_k *= packing_factor
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (size_m, size_k)
    assert rhs.shape == (size_group, size_k // packing_factor, size_n)
    if rhs_bias is not None:
        assert rhs_bias.shape == (size_group, 1, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
        assert size_k % num_quant_blocks == 0

    assert group_offset.shape == (1,)

    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            raise ValueError(
                f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
                "enabled since we need to split n dimension for gate and up."
            )

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs):
    """Returns the cost estimate for the GMM kernel."""

    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype

    # We use bits for rhs since it could sub-byte dtype like int4.
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    # TODO(kyuyeunk): Add compute flops for quant, dequant, and bias.
    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_size = dims.size_group * dims.size_k * dims.size_n
    rhs_bytes = rhs_size * rhs_bits // 8
    if cfgs.rhs_cfgs.has_scale:
        num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
    if cfgs.rhs_cfgs.has_bias:
        rhs_bytes += dims.size_group * dims.size_n * fp32_bytes

    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(cfgs: GmmConfigs) -> str:
    dims = cfgs.dims
    tiles = cfgs.tiles
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


# Columns per block that gmm_v2 quantizes lhs rows in (quantize_block).
LHS_QUANT_BLOCK_SIZE = 512


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
):
    """Fills the GMM config for the GMM kernel."""

    packing_factor = get_packing_factor(rhs.dtype, rhs_quant_dtype)

    dims = validate_inputs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        fuse_act,
        packing_factor,
    )

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = rhs_quant_dtype or rhs.dtype
        num_blocks = rhs_scale.shape[1]
        block_size = dims.size_k // num_blocks
    else:
        has_scale = False
        block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
    )

    lhs_q_dtype = None
    if maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
        # Choose lhs quantization dtype based on TPU hardware support.
        is_rhs_float = jnp.issubdtype(cast(jnp.dtype, rhs_quant_dtype), jnp.floating)
        tpu_info = pltpu.get_tpu_info()
        # Check if there is hardware compute support for rhs dtype group.
        if tpu_info.fp8_ops_per_second > 0:
            # Special handling for 4-bit integer rhs as it can be converted to fp8
            # without a numeric issues. Note that this is not the case for 4-bit
            # floating rhs as conversion to int8 will cause numeric issues.
            is_rhs_4bits = (
                jax.dtypes.itemsize_bits(cast(jnp.dtype, rhs_quant_dtype)) == 4
            )
            if is_rhs_float or is_rhs_4bits:
                lhs_q_dtype = jnp.float8_e4m3fn.dtype
        if tpu_info.int8_ops_per_second > 0 and not is_rhs_float:
            lhs_q_dtype = jnp.int8.dtype

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        # Input quantization involves reading all elements in a block to compute
        # scale value. Since this operation is very memory intensive, we use a
        # block size that is small enough to minimize memory overhead but large
        # enough to minimize compute overhead of quantization.
        quant_block_size=LHS_QUANT_BLOCK_SIZE,
        dtype=lhs.dtype,
    )

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        if lhs_cfgs.quant_dtype is None:
            acc_dtype = jnp.float32.dtype
        else:
            # Input quantization requires elementwise ops which can put pressure on
            # VPUs. Using faster bf16 hardware during accumulation can help offset the
            # pressure.
            acc_dtype = jnp.bfloat16.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.dtype(out_dtype),
        acc_dtype=jnp.dtype(acc_dtype),
        zero_init=zero_initialize,
        fuse_act=fuse_act,
    )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


def _padded_size_m(lhs: jax.Array) -> int:
    """Rows of lhs padded up to the sublane block the kernel reshapes it by."""
    size_m = lhs.shape[0]
    return align_to(size_m, _get_lhs_sublane_size(lhs.dtype, size_m))


def _prepare_inputs(
    lhs: jax.Array, group_offset: jax.Array | None
) -> tuple[jax.Array, jax.Array]:
    """Pads lhs rows to their sublane block and makes group_offset int32[1].

    The extra rows sit past every group, so they add no work, and the caller
    slices them back off the output.
    """
    size_m = lhs.shape[0]
    padded_size_m = _padded_size_m(lhs)
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]
    return lhs, group_offset


def _default_vmem_limit_bytes() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)


@jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "maybe_quantize_lhs",
        "zero_initialize",
        "fuse_act",
        "rhs_quant_dtype",
    ]
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """GMM kernel implemented with emit_pipeline.

    Dynamically calculate offset lhs/out tiles to reduce redundant computations.
    Additionally, it adjusts dma size based on number of valid rows and utilize
    triple buffering on weights to better utilize memory.

    Args:
        lhs: lhs with shape [size_m, size_k].
        rhs: rhs with shape  native -> [size_group, size_k, size_n],
            packed -> [size_group, size_k/8, size_n]
        group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
        rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size].
        rhs_bias: The rhs bias of shape [size_group, 1, out_size].
        group_offset: Optional. The group offset of shape [1,].
        tile_info: The tile sizes or tile function to use.
        vmem_limit_bytes: Optional vmem limit in bytes.
        precision: Unused. Exists for compatibility reasons.
        preferred_element_type: Optional jnp.dtype for the output matrix.
        acc_dtype: Optional jnp.dtype for the accumulator.
        maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
        zero_initialize: Whether to initialize unvisited output elements to zero.
        fuse_act: Activation function to fuse with GMM, None if no fusion.
        rhs_quant_dtype: Quantized dtype encoded by a pre-packed RHS carrier.
            Packed along k-axis

    Returns:
        Output of shape [size_m, size_n].
    """

    del precision

    size_m = lhs.shape[0]
    lhs, group_offset = _prepare_inputs(lhs, group_offset)

    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit_bytes()

    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    # Prepare block specs.
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Initialize scratch shapes.
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    if cfgs.zero_init:
        # TODO(kyuyeunk): Create better heuristics for determining this value.
        target_zero_ref_bytes = 2 * 1024 * 1024

        # Zero initialization is done by tiling size_m dim where each tile invokes
        # zero initializing DMA for up-to tile_zero_m rows. This means larger
        # tile_zero_m will result in fewer number of tiles and lead to smaller
        # overhead. However, in order to invoke DMA call up-to tile_zero_m rows, we
        # need to store equivalent sized memory in VMEM buffer for the duration of
        # DMA. Storing [tile_zero_m, size_n] in buffer will trigger OOM if
        # tile_zero_m is too large. Instead, if we set column size as num_lanes
        # (which is smallest allowed column size for DMA) and reuse the buffer by
        # size_n//num_lanes times in a single tile, we can significantly increase
        # tile_zero_m without triggering OOM.
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims.size_m)

        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = align_to(cfgs.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    return pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                    bias=rhs_bias_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, lhs, rhs_weights)[:size_m, : cfgs.out_size_n]


# Most row buckets per tile in the fused gate/up -> down kernel, as in
# calculate_tiling; the VMEM estimate can pick fewer.
FUSED_MLP_MAX_BUCKETS = 4


def fused_mlp_vmem_estimate(tm: int, cfgs1: GmmConfigs, cfgs2: GmmConfigs) -> int:
    """Estimates the VMEM bytes of gmm_v2_fused_mlp at row tile tm."""
    leg1 = gmm_vmem_estimate(
        tm,
        cfgs1.tiles.tile_n,
        cfgs1.tiles.tile_k,
        lhs_cfgs=cfgs1.lhs_cfgs,
        rhs_cfgs=cfgs1.rhs_cfgs,
        fuse_act_factor=2,
    )
    leg2 = gmm_vmem_estimate(
        tm,
        cfgs2.tiles.tile_n,
        cfgs2.tiles.tile_k,
        lhs_cfgs=cfgs2.lhs_cfgs,
        rhs_cfgs=cfgs2.rhs_cfgs,
        fuse_act_factor=1,
    )
    # The intermediate is one VMEM tile, not gate/up's double-buffered output
    # plus down's double-buffered input.
    inter_tile = tm * cfgs2.tiles.tile_k * cfgs1.out_dtype.itemsize
    # Down's accumulator lives after gate/up's is dead.
    acc_tile = tm * min(
        2 * cfgs1.tiles.tile_n * cfgs1.acc_dtype.itemsize,
        cfgs2.tiles.tile_n * cfgs2.acc_dtype.itemsize,
    )
    return leg1 + leg2 - 3 * inter_tile - acc_tile


def _fused_mlp_widths_match(
    rhs1: jax.Array, rhs2: jax.Array, rhs_quant_dtype: jnp.dtype | None
) -> bool:
    """Whether down's input width is gate/up's output width (half its n)."""
    size_inter = rhs2.shape[1] * get_packing_factor(rhs2.dtype, rhs_quant_dtype)
    return rhs1.shape[-1] == 2 * size_inter


def make_fused_mlp_configs(
    lhs: jax.Array,  # [size_m, size_k], size_m aligned to the lhs sublane
    rhs1: jax.Array,
    rhs1_scale: jax.Array | None,
    rhs1_bias: jax.Array | None,
    rhs2: jax.Array,
    rhs2_scale: jax.Array | None,
    rhs2_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    vmem_limit_bytes: int,
    out_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    fuse_act: str,
    rhs_quant_dtype: jnp.dtype | None,
    token_rows: bool = False,
    quantized_lhs_dtype: jnp.dtype | None = None,
) -> tuple[GmmConfigs, GmmConfigs] | None:
    """Configs of the gate/up and down legs of gmm_v2_fused_mlp.

    With ``token_rows`` the row tile also leaves room for TokenRows' stage.
    With ``quantized_lhs_dtype``, gate/up reads its lhs rows quantized ahead
    to that dtype (quantized_rows); lhs still describes them unquantized. Both
    legs then step through rows in the quantized dtype's sublane blocks.

    Returns None when the pair cannot run fused with the arithmetic of two
    gmm_v2 calls: the down input width must equal the gate/up output width,
    gmm_v2 must run each leg as one k tile (a k split sums a row's partial
    products in another order), and both legs' full-width tiles must fit VMEM.
    """
    if not _fused_mlp_widths_match(rhs1, rhs2, rhs_quant_dtype):
        return None
    common: dict[str, Any] = dict(
        tile_info=calculate_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        acc_dtype=None,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=False,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    cfgs1 = make_gmm_configs(
        lhs,
        rhs1,
        rhs1_scale,
        rhs1_bias,
        group_sizes,
        group_offset,
        out_dtype=None,
        fuse_act=fuse_act,
        **common,
    )
    inter = jax.ShapeDtypeStruct((cfgs1.dims.size_m, cfgs1.out_size_n), cfgs1.out_dtype)
    cfgs2 = make_gmm_configs(
        inter,
        rhs2,
        rhs2_scale,
        rhs2_bias,
        group_sizes,
        group_offset,
        out_dtype=out_dtype,
        fuse_act=None,
        **common,
    )
    if any(c.tiles.tile_k < c.dims.size_k for c in (cfgs1, cfgs2)):
        return None

    num_lanes = pltpu.get_tpu_info().num_lanes
    bucket_base = cfgs1.tiles.bucket_base
    tiles1 = TileSizes(
        tile_m=bucket_base,
        tile_k=cfgs1.tiles.tile_k,
        tile_n=align_to(cfgs1.out_size_n, num_lanes),
        bucket_base=bucket_base,
    )
    tiles2 = TileSizes(
        tile_m=bucket_base,
        tile_k=cfgs2.tiles.tile_k,
        tile_n=align_to(cfgs2.out_size_n, num_lanes),
        bucket_base=bucket_base,
    )
    cfgs1 = dataclasses.replace(cfgs1, tiles=tiles1)
    cfgs2 = dataclasses.replace(cfgs2, tiles=tiles2)

    row_width = cfgs1.dims.size_k
    if quantized_lhs_dtype is not None:
        row_width = quantized_row_width(row_width)
        lhs_cfgs = cfgs1.lhs_cfgs
        if (
            lhs_cfgs.quant_dtype != quantized_lhs_dtype
            or lhs_cfgs.dtype != SOURCE_DTYPE
            or cfgs1.dims.size_k % lhs_cfgs.quant_block_size
        ):
            return None
        sublane = _get_lhs_sublane_size(quantized_lhs_dtype, cfgs1.dims.size_m)
        cfgs1 = dataclasses.replace(
            cfgs1,
            lhs_cfgs=dataclasses.replace(lhs_cfgs, quantized_ahead=True),
            dims=dataclasses.replace(cfgs1.dims, size_lhs_sublane=sublane),
        )
        cfgs2 = dataclasses.replace(
            cfgs2, dims=dataclasses.replace(cfgs2.dims, size_lhs_sublane=sublane)
        )

    max_tile_m = min(FUSED_MLP_MAX_BUCKETS * bucket_base, cfgs1.dims.size_m)
    tile_m = 0
    for tm in range(bucket_base, max_tile_m + 1, bucket_base):
        # TokenRows' stage holds a 32-bit word row per tile row.
        stage_bytes = tm * row_width * 4 if token_rows else 0
        if fused_mlp_vmem_estimate(tm, cfgs1, cfgs2) + stage_bytes > vmem_limit_bytes:
            break
        tile_m = tm
    if tile_m == 0:
        return None
    return (
        dataclasses.replace(cfgs1, tiles=dataclasses.replace(tiles1, tile_m=tile_m)),
        dataclasses.replace(cfgs2, tiles=dataclasses.replace(tiles2, tile_m=tile_m)),
    )


def can_fuse_mlp(
    lhs: jax.Array,  # [size_m, size_k]
    rhs1: jax.Array,  # [size_group, size_k, 2 * size_inter]
    rhs2: jax.Array,  # [size_group, size_inter, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs1_scale: jax.Array | None = None,
    rhs1_bias: jax.Array | None = None,
    rhs2_scale: jax.Array | None = None,
    rhs2_bias: jax.Array | None = None,
    *,
    fuse_act: str,
    preferred_element_type: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    rhs_quant_dtype: jnp.dtype | None = None,
    token_rows: bool = False,
    quantized_lhs_dtype: jnp.dtype | None = None,
) -> bool:
    """Whether gmm_v2_fused_mlp takes these operands. Shape arithmetic only.

    With ``token_rows``, whether it takes them with lhs rows read from token
    rows (gmm_v2_fused_mlp's token_rows); lhs then stands for all the rows,
    of quantized_lhs_dtype if given (quantized_rows rows, not their words).
    """
    if token_rows:
        if (
            lhs.dtype.itemsize != _token_row_itemsize(quantized_lhs_dtype)
            or _padded_size_m(lhs) != lhs.shape[0]
        ):
            return False
        size_m = lhs.shape[0]
        sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
        if not token_rows_fit_smem(size_m, sublane, 4 // lhs.dtype.itemsize):
            return False
    elif quantized_lhs_dtype is not None:
        return False
    if not _fused_mlp_widths_match(rhs1, rhs2, rhs_quant_dtype):
        return False
    cfgs = make_fused_mlp_configs(
        _unquantized_lhs(
            jax.ShapeDtypeStruct((_padded_size_m(lhs), lhs.shape[1]), lhs.dtype),
            quantized_lhs_dtype,
        ),
        rhs1,
        rhs1_scale,
        rhs1_bias,
        rhs2,
        rhs2_scale,
        rhs2_bias,
        group_sizes,
        jax.ShapeDtypeStruct((1,), jnp.int32),
        vmem_limit_bytes=_default_vmem_limit_bytes(),
        out_dtype=preferred_element_type,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
        token_rows=token_rows,
        quantized_lhs_dtype=quantized_lhs_dtype,
    )
    return cfgs is not None


def _token_row_itemsize(quantized_lhs_dtype: jnp.dtype | None) -> int:
    """Bytes of a token row's element: bf16, or 8 bits quantized ahead."""
    return 2 if quantized_lhs_dtype is None else 4 // ROWS_PER_WORD


def _unquantized_lhs(lhs: jax.ShapeDtypeStruct, quantized_lhs_dtype: jnp.dtype | None):
    """lhs as the configs describe it: before any quantization ahead."""
    if quantized_lhs_dtype is None:
        return lhs
    return jax.ShapeDtypeStruct((lhs.shape[0], lhs.shape[1] - SCALE_COLS), SOURCE_DTYPE)


@jax.jit(
    static_argnames=[
        "fuse_act",
        "preferred_element_type",
        "maybe_quantize_lhs",
        "rhs_quant_dtype",
        "quantized_lhs_dtype",
    ]
)
def gmm_v2_fused_mlp(
    lhs: jax.Array,  # [size_m, size_k]
    rhs1: jax.Array,  # [size_group, size_k, 2 * size_inter]
    rhs2: jax.Array,  # [size_group, size_inter, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs1_scale: jax.Array | None = None,
    rhs1_bias: jax.Array | None = None,
    rhs2_scale: jax.Array | None = None,
    rhs2_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    fuse_act: str,
    preferred_element_type: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    rhs_quant_dtype: jnp.dtype | None = None,
    token_rows: jax.Array | None = None,  # [num_tokens, size_k]
    token_row_ids: jax.Array | None = None,  # int32[size_m]
    quantized_lhs_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """gmm_v2 down(act(gate/up(lhs))) in one kernel.

    Computes what
        inter = gmm_v2(lhs, rhs1, ..., fuse_act=fuse_act, zero_initialize=False)
        out = gmm_v2(inter, rhs2, ..., zero_initialize=False)
    computes, with the intermediate kept in VMEM: per gm tile, gate/up and its
    activation write a bf16 intermediate tile that down quantizes and consumes,
    so it never round-trips HBM, and both legs' weights stream through one
    pipeline. Only operands for which can_fuse_mlp holds are accepted.

    With ``token_rows``, lhs holds only the first rows of the batch, a whole
    number of sublane blocks, and row r past them is row
    ``token_row_ids[r]`` of the bf16 ``token_rows``: the kernel fetches it
    itself, so those rows need not be gathered first. A row computes the
    same bits either way. Every id past the head must index ``token_rows``,
    rows outside the groups included: a tile also fetches the rows of its
    first sublane block that come before it.

    With ``quantized_lhs_dtype`` (token_rows only), lhs and token_rows are
    the int32 words of rows quantized_rows.quantize_rows quantized to that
    dtype, as the gate/up leg would have, so the kernel skips that step and
    computes the same bits.

    Returns:
        Output of shape [size_m, size_n].
    """

    if (token_rows is None) != (token_row_ids is None):
        raise ValueError("token_rows and token_row_ids go together")
    full_lhs = lhs
    # The dtype of lhs rows, and how many share an element of lhs.
    row_dtype, rows_per_word = lhs.dtype, 1
    if quantized_lhs_dtype is not None:
        if token_rows is None or lhs.dtype != jnp.int32:
            raise ValueError(
                "gmm_v2_fused_mlp reads lhs quantized ahead as int32 words, "
                "with token rows"
            )
        row_dtype, rows_per_word = jnp.dtype(quantized_lhs_dtype), ROWS_PER_WORD
    if token_rows is not None:
        full_lhs = jax.ShapeDtypeStruct(
            (token_row_ids.shape[0], lhs.shape[1]), row_dtype
        )
        head_rows = lhs.shape[0] * rows_per_word
        sublane = _get_lhs_sublane_size(row_dtype, full_lhs.shape[0])
        if (
            row_dtype.itemsize != _token_row_itemsize(quantized_lhs_dtype)
            or token_rows.dtype != lhs.dtype
            or token_rows.shape[1] != lhs.shape[1]
            or token_rows.shape[0] * rows_per_word % (4 // row_dtype.itemsize)
            or head_rows % sublane
            or _padded_size_m(full_lhs) != full_lhs.shape[0]
            or head_rows > full_lhs.shape[0]
        ):
            raise ValueError(
                f"token rows {token_rows.shape} {token_rows.dtype} with head "
                f"{lhs.shape} {lhs.dtype} and {full_lhs.shape[0]} row ids"
            )
    size_m = full_lhs.shape[0]
    lhs, group_offset = _prepare_inputs(lhs, group_offset)
    vmem_limit_bytes = _default_vmem_limit_bytes()

    cfgs = make_fused_mlp_configs(
        _unquantized_lhs(
            jax.ShapeDtypeStruct((_padded_size_m(full_lhs), lhs.shape[1]), row_dtype),
            quantized_lhs_dtype,
        ),
        rhs1,
        rhs1_scale,
        rhs1_bias,
        rhs2,
        rhs2_scale,
        rhs2_bias,
        group_sizes,
        group_offset,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
        token_rows=token_rows is not None,
        quantized_lhs_dtype=quantized_lhs_dtype,
    )
    if cfgs is None:
        raise ValueError(
            "gmm_v2_fused_mlp cannot run these operands as one kernel; call "
            "can_fuse_mlp first and use two gmm_v2 calls when it is False."
        )
    cfgs1, cfgs2 = cfgs
    dims, tiles = cfgs1.dims, cfgs1.tiles

    def rhs_weights(rhs, rhs_scale, rhs_bias):
        if rhs_scale is not None:
            rhs_scale = rhs_scale.astype(jnp.float32)
        if rhs_bias is not None:
            rhs_bias = rhs_bias.astype(jnp.float32)
        weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)
        specs = jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), weights)
        return weights, specs

    rhs1_weights, rhs1_specs = rhs_weights(rhs1, rhs1_scale, rhs1_bias)
    rhs2_weights, rhs2_specs = rhs_weights(rhs2, rhs2_scale, rhs2_bias)

    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    scratch_shapes = [
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, cfgs2.tiles.tile_n), cfgs2.out_dtype),
        # inter_ref
        pltpu.VMEM((tiles.tile_m, cfgs2.tiles.tile_k), cfgs1.out_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]
    scalars = [group_sizes, group_offset]
    inputs = [lhs, rhs1_weights, rhs2_weights]
    in_specs = [pl.BlockSpec(memory_space=pltpu.HBM), rhs1_specs, rhs2_specs]
    if token_rows is not None:
        sublane = dims.size_lhs_sublane
        packing = 4 // row_dtype.itemsize
        width = lhs.shape[1]
        row_ids = token_row_ids.astype(jnp.int32)
        # TokenRows.word_ids_ref and TokenRows.parts_ref
        scalars += [
            lax.shift_right_logical(row_ids, packing.bit_length() - 1),
            token_row_parts(row_ids, sublane, packing),
        ]
        inputs.insert(1, token_rows)
        in_specs.insert(1, pl.BlockSpec(memory_space=pltpu.HBM))
        scratch_shapes += [
            # TokenRows.lhs_bufs
            pltpu.VMEM((2, tiles.tile_m // sublane, sublane, width), row_dtype),
            # TokenRows.stage_ref
            pltpu.VMEM(
                (packing, tiles.tile_m // sublane, sublane // packing, width),
                jnp.uint32,
            ),
            pltpu.SemaphoreType.DMA((2,)),
        ]

    # The cost estimates count the intermediate as gate/up's output and as
    # down's lhs; the fused kernel moves neither.
    cost1, cost2 = get_cost_estimate(cfgs1), get_cost_estimate(cfgs2)
    lhs2_dtype = cfgs2.lhs_cfgs.quant_dtype or cfgs2.lhs_cfgs.dtype
    inter_itemsize = cfgs1.out_dtype.itemsize + lhs2_dtype.itemsize
    inter_bytes = dims.size_m * cfgs1.out_size_n * inter_itemsize
    out_init = jax.ShapeDtypeStruct((dims.size_m, cfgs2.tiles.tile_n), cfgs2.out_dtype)
    return pl.pallas_call(
        functools.partial(
            fused_mlp_kernel_main,
            cfgs1=cfgs1,
            cfgs2=cfgs2,
            token_rows=token_rows is not None,
        ),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=(
            f"gmm_v2_fused_mlp-g_{dims.size_group}-m_{dims.size_m}"
            f"-k_{dims.size_k}-act_{fuse_act}-i_{cfgs1.out_size_n}"
            f"-n_{cfgs2.out_size_n}-tm_{tiles.tile_m}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=cost1.flops + cost2.flops,
            bytes_accessed=cost1.bytes_accessed + cost2.bytes_accessed - inter_bytes,
            transcendentals=0,
        ),
        metadata={
            **{f"gate_up.{k}": v for k, v in get_metadata(cfgs1).items()},
            **{f"down.{k}": v for k, v in get_metadata(cfgs2).items()},
        },
    )(*scalars, *inputs)[:size_m, : cfgs2.out_size_n]
