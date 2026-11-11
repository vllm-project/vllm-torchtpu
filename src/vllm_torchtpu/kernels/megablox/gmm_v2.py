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
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

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


# Largest lhs bucket fed to the MXU as fp8 (see fp8_rhs_feed_dtype). Its
# matmuls per 128-deep quant block re-stream the lhs rows, which binds from
# 64 rows on (a tie with bf16 there). Crossover measured on tpu7x MXU timings.
FP8_RHS_MAX_BUCKET_M = 48
# Decode tile_m on the fp8 rhs feed (replaces tile_m 64): on tpu7x, rows up to
# it cost no more than one row.
FP8_RHS_TILE_M = 32

# Skinny-m path of calls with fp4 rhs and an unquantized lhs, both scaling
# after the matmul: "fp8_feed" (default) feeds the MXU the rhs as fp8
# (`fp8_rhs_feed_dtype`, buckets up to FP8_RHS_MAX_BUCKET_M rows, and
# gmm_v2_mlp's fused kernel); "bf16_after" keeps the bf16 per-block path
# (`SKINNY_DEQUANT_AFTER_MATMUL_M`, `HYBRID_DEQUANT_AFTER_MATMUL_MAX_ROWS`).
# "fp8_feed_wide" is "fp8_feed" that also keeps the fp8 tiles (tile_m
# FP8_RHS_TILE_M) up to FP8_RHS_WIDE_MAX_M rows when VMEM stopped tile_m's
# growth: from 128 decode rows (T=32) "fp8_feed" grows tile_m past
# FP8_RHS_MAX_BUCKET_M and drops every bucket to bf16 dequantize-before.
# Set with --additional-config '{"moe_skinny_gmm": "bf16_after"}'
# (`set_skinny_gmm_path`), before the model's first trace.
SKINNY_GMM_PATHS = ("fp8_feed", "bf16_after", "fp8_feed_wide")
FP8_RHS_WIDE_MAX_M = 2048
_skinny_gmm_path = "fp8_feed"


# r6 (opt-in, `set_fp8_decode_tiles`): decode-sized buffers past
# FP8_RHS_TILE_M rows per tile take tile_m FP8_DECODE_TILE_M in the buckets
# FP8_DECODE_BUCKETS (fp8 feed up to FP8_RHS_MAX_BUCKET_M rows, one bf16
# bucket for the rest), or fp8-sized tiles up to FP8_SMALL_DECODE_MAX_M rows,
# ahead of the skinny paths above; kernel_mlp fuses such tiles. Set with
# key moe_gmm_fp8_decode_tiles before the model's first trace.
FP8_DECODE_TILE_M = 128
FP8_DECODE_BUCKETS = (16, 32, 48, FP8_DECODE_TILE_M)
FP8_DECODE_MAX_M = 2048
FP8_SMALL_DECODE_MAX_M = 512
_fp8_decode_tiles = False


def set_fp8_decode_tiles(enabled: bool) -> None:
    global _fp8_decode_tiles
    _fp8_decode_tiles = bool(enabled)


def set_skinny_gmm_path(path: str) -> None:
    global _skinny_gmm_path
    if path not in SKINNY_GMM_PATHS:
        raise ValueError(f"moe_skinny_gmm must be one of {SKINNY_GMM_PATHS}, got {path!r}")
    _skinny_gmm_path = path


def skinny_gmm_path() -> str:
    return _skinny_gmm_path


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
    # Row counts a tile can end at, ascending, the last one tile_m. None means
    # every multiple of bucket_base.
    buckets: tuple[int, ...] | None = None
    # Weight blocks the rhs pipeline keeps in flight.
    rhs_buffer_count: int = 3

    @property
    def bucket_sizes(self) -> tuple[int, ...]:
        if self.buckets is not None:
            return self.buckets
        return tuple(
            range(self.bucket_base, self.tile_m + 1, self.bucket_base)
        )


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


# Unquantized-lhs calls of at most this many rows (decode) scale each
# `quant_block_size`-deep matmul partial ([m, tn]) instead of dequantizing the
# weight tile ([tk, tn]) before one deep matmul. Below the MXU column size
# the before-matmul path keeps the MXU deep, which matters at large m; at
# skinny m it spends the step on the VPU converting and scaling every weight
# element (NVFP4 block 128 on tpu7x, gate_up + down per MoE layer: 124 -> 86
# us at 32 rows / 4 active experts, 210 -> 171 at 64 / 8, 410 -> 343 at
# 256 / 16).
SKINNY_DEQUANT_AFTER_MATMUL_M = 256
# Blocks shallower than this make the per-block matmuls too small.
MIN_DEQUANT_AFTER_MATMUL_BLOCK = 128
# Larger calls keep the before-matmul path for full tiles but scale the
# partials of tiles ending at or below this many rows. An EP decode call is
# sized for every route (c64: 1024-2048 rows) while each local expert gets a
# few (~8-16), so its tiles are skinny although size_m is not. Converting
# each weight element once per tile, the after path is cheaper while the
# per-block [rows, tn] epilogue stays under the weight-scale multiply.
HYBRID_DEQUANT_AFTER_MATMUL_MAX_ROWS = 64


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    quant_dtype: jnp.dtype | None
    quant_block_size: int
    dtype: jnp.dtype
    has_bias: bool = False
    has_scale: bool = False
    # Scale each block's matmul partial instead of the dequantized weights
    # (see `SKINNY_DEQUANT_AFTER_MATMUL_M`).
    dequantize_after_matmul: bool = False
    # Dequantize before the matmul, except in tiles ending at or below this
    # many rows, which scale the partials (see
    # `HYBRID_DEQUANT_AFTER_MATMUL_MAX_ROWS`). 0 disables.
    hybrid_after_max_rows: int = 0

    @property
    def should_unpack(self) -> bool:
        """True if rhs weights are sub-byte (e.g. INT4) packed in
        INT32/UINT32 carriers."""
        return get_packing_factor(self.dtype, self.quant_dtype) > 1

    @property
    def should_dequantize_before_matmul(self) -> bool:
        """Dequantize rhs before matmul if block size limits MXU utilization."""
        if not self.has_scale or self.dequantize_after_matmul:
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


def fp8_rhs_feed_dtype(
    lhs_cfgs: InputConfigs, rhs_cfgs: InputConfigs
) -> jnp.dtype | None:
    """fp8 dtype that holds fp4 rhs exactly and multiplies bf16 lhs natively.

    Feeding it to the MXU (scale applied after the matmul) halves the weight
    pushes of the bf16 dequantize-before-matmul path, which bind at skinny m.
    """
    if _skinny_gmm_path not in ("fp8_feed", "fp8_feed_wide"):
        return None
    fp8 = jnp.float8_e4m3fn.dtype
    rhs_q = rhs_cfgs.quant_dtype
    tpu_info = pltpu.get_tpu_info()
    if (
        rhs_cfgs.should_dequantize_before_matmul
        # Lane-aligned blocks only: narrower ones would mean tiny-K matmuls.
        and rhs_cfgs.quant_block_size % tpu_info.num_lanes == 0
        and not lhs_cfgs.should_quantize
        and rhs_q is not None
        and jnp.issubdtype(rhs_q, jnp.floating)
        and jax.dtypes.itemsize_bits(rhs_q) == 4
        and tpu_info.is_matmul_supported(lhs_cfgs.dtype, fp8)
    ):
        return fp8
    return None


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

    def fp8_rhs_dtype(self, bucket_m: int) -> jnp.dtype | None:
        """fp8 dtype to feed the MXU this bucket's rhs in, or None for bf16."""
        if bucket_m > FP8_RHS_MAX_BUCKET_M:
            return None
        return fp8_rhs_feed_dtype(self.lhs_cfgs, self.rhs_cfgs)

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
        pipeline_mode=pl.Buffered(buffer_count=cfgs.tiles.rhs_buffer_count),
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
        tpu_info = pltpu.get_tpu_info()
        mxu_size = tpu_info.mxu_column_size

        # Step 1: Input pre-processing.
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
        tiled_rhs = tiled_rhs_ref.get_weight()

        # This should only be taken in the case where we don't requantize
        # the scales and thus we need to dequantize inside VMEM to avoid small
        # contracting dimmensions
        rhs_tile_n = tiled_rhs.shape[1]
        rhs_qbs = cfgs.rhs_cfgs.quant_block_size
        fp8_rhs = cfgs.fp8_rhs_dtype(bucket_m)
        dequant_before = (
            fp8_rhs is None
            and cfgs.rhs_cfgs.should_dequantize_before_matmul
            and not (bucket_m <= cfgs.rhs_cfgs.hybrid_after_max_rows)
        )
        dequant_after = cfgs.rhs_cfgs.has_scale and not dequant_before
        if fp8_rhs is not None:
            # Skinny m: the MXU weight push binds, and fp8 halves it.
            tiled_rhs = tiled_rhs.astype(fp8_rhs)
        elif dequant_before:
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

                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                    end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)

                    block_acc = jnp.matmul(
                        tiled_lhs[:, start_k:end_k],
                        tiled_rhs[start_k:end_k, start_n:end_n],
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)

                    if dequant_after:
                        b_id = start_k // rhs_qbs
                        rhs_scale_replicated = tiled_rhs_ref.get_scale(
                            replicate_size=bucket_m
                        )[b_id, :, start_n : start_n + col_size]
                        block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        else:
            # Quantized matmul path.
            lhs_q_dtype = cast(jnp.dtype, cfgs.lhs_cfgs.quant_dtype)
            q_block_size = cfgs.lhs_cfgs.quant_block_size

            if jnp.issubdtype(lhs_q_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(lhs_q_dtype).max)
                preferred_element_type = jnp.float32
            else:
                dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
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

                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                    # Perform lhs quantization. Note that for every block_lhs,
                    # same computation will be performed tiles_n//mxu_size times.
                    # But we can let compiler perform CSE and avoid recomputation.
                    block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
                    block_scale = block_abs_max / dtype_max

                    # If block_scale=0, it will cause division by zero and return either
                    # NaN or Inf. Since this can cause numeric issue when downcasting to
                    # quantized value, we convert them into 0.
                    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
                    # Convert lhs into quantized dtype.
                    block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

                    # Unlike unquantized path, compiler may not perform implicit type
                    # conversion due to numeric concerns. As this can cause unsupported
                    # matmul error, explicit type conversion is performed.
                    if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
                        block_rhs = block_rhs.astype(lhs_q_dtype)

                    block_acc = jnp.matmul(
                        block_lhs_q,
                        block_rhs,
                        preferred_element_type=preferred_element_type,
                    ).astype(acc_ref.dtype)

                    block_acc *= block_scale.astype(acc_ref.dtype)

                    # Apply rhs subchannel scale per quant block.
                    if dequant_after:
                        b_id = start_k // rhs_qbs
                        rhs_scale_replicated = tiled_rhs_ref.get_scale(
                            replicate_size=bucket_m
                        )[b_id, :, start_n : start_n + col_size]
                        block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        acc = jnp.concatenate(acc_list, axis=1)

        # Step 3: Output post-processing.
        if not is_first_k_step:
            acc = jnp.pad(acc, ((cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
            acc += acc_ref[...]
        acc_m = acc.shape[0]

        if is_last_k_step:
            if cfgs.rhs_cfgs.has_bias:
                tiled_rhs_bias = tiled_rhs_ref.get_bias()
                acc += tiled_rhs_bias.astype(acc.dtype)

            acc = apply_act_fn(acc, cfgs.fuse_act)

            # Mask out rows that does not belong to the current group.
            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
            acc_masked = jnp.where(mask, acc, 0)

            # Write the final output to the output ref.
            tiled_out_2d_ref = tiled_out_ref.reshape(-1, cfgs.tiles.tile_n)
            tiled_out_2d_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype)

            # If this is the first tile for grid[n_id, :, :], we initialize the
            # partial out to zeros. Otherwise, partial out from last tile of
            # grid[n_id-1, :, :] can be used and cause numeric issues.
            partial_out_zeros = jnp.zeros_like(partial_out_ref)

            # Accumulate the partial output from the previous step.
            tiled_out_ref[0] += jnp.where(
                gm_id == 0, partial_out_zeros, partial_out_ref[...]
            )

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
        else:
            acc_ref[:acc_m] = acc

    def run_matmul_step(bucket_m: int):

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

    bucket_sizes = cfgs.tiles.bucket_sizes
    if cfgs.tiles.buckets is None:
        # Smallest bucket covering rows [0, m_end_local); m_end_local >= 1.
        bucket_idx = (m_end_local - 1) // cfgs.tiles.bucket_base
    else:
        # First bucket holding rows [0, m_end_local).
        bucket_idx = sum(
            (m_end_local > b).astype(jnp.int32) for b in bucket_sizes[:-1]
        )
    branches = [
        functools.partial(run_matmul_step, bucket_m=b) for b in bucket_sizes
    ]
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

    run_gmm_pipeline(
        lhs_ref,
        rhs_ref,
        out_ref,
        partial_out_ref,
        acc_ref,
        metadata_ref,
        num_gm,
        cfgs=cfgs,
    )

    if cfgs.zero_init:
        zero_out_end(out_ref, cast(jax.Array, semaphore_ref), zero_size, dims=cfgs.dims)


def run_gmm_pipeline(
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    out_ref: jax.Array,  # [size_m, size_n]
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    cfgs: GmmConfigs,
    rhs_weight_window: jax.Array | None = None,
    last_step_hook: Callable[[], None] | None = None,
):
    """Runs the grouped matmul over the gm tiles that fill_metadata laid out.

    last_step_hook runs at the start of the last grid step, when the pipeline
    fetches nothing more (kernel_mlp issues the next gmm's first DMA there).

    rhs_weight_window: [2, tile_k, tile_n] VMEM whose slot 0 already holds the
    first grid step's rhs weight block (see kernel_mlp); the pipeline then
    double-buffers rhs weights in it instead of fetching that block itself.
    """
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if cfgs.rhs_cfgs.should_unpack:
        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)

        rhs_spec = FusedWeightsRef(
            gate=rhs_spec,
            up=rhs_spec,
        )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    grid = (num_n, num_gm, num_k)
    body = functools.partial(inner_kernel, cfgs=cfgs)
    if last_step_hook is not None:
        inner_body = body

        def body(*refs):
            is_last = functools.reduce(
                jnp.logical_and,
                [pl.program_id(i) == pl.num_programs(i) - 1 for i in range(3)],
            )
            pl.when(is_last)(last_step_hook)
            inner_body(*refs)

    if rhs_weight_window is None:
        pltpu.emit_pipeline(
            body, grid=grid, in_specs=(lhs_spec, rhs_spec), out_specs=out_spec
        )(lhs_in, rhs_ref, out_in, scratches=scratches)
        return

    assert cfgs.fuse_act is None
    rhs_spec = dataclasses.replace(
        rhs_spec,
        weight=dataclasses.replace(
            rhs_spec.weight,
            pipeline_mode=pl.Buffered(
                buffer_count=rhs_weight_window.shape[0], prefetched_count=1
            ),
        ),
    )
    pipeline_fn, make_allocations = pltpu.emit_pipeline_with_allocations(
        body, grid=grid, in_specs=(lhs_spec, rhs_spec), out_specs=out_spec
    )

    def run(allocations):
        lhs_bref, rhs_bref, *out_brefs = allocations
        rhs_bref = dataclasses.replace(
            rhs_bref, weight=rhs_bref.weight.with_window_ref(rhs_weight_window)
        )
        pipeline_fn(
            lhs_in,
            rhs_ref,
            out_in,
            scratches=scratches,
            allocations=[lhs_bref, rhs_bref, *out_brefs],
        )

    pl.run_scoped(run, make_allocations(lhs_in, rhs_ref, out_in))


def kernel_mlp(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs1_ref: WeightsRef,  # [size_group, size_k, size_n1]
    rhs2_ref: WeightsRef,  # [size_group, size_n1 out, size_n2]
    # Out
    out_ref: jax.Array,  # [size_m, size_n2]
    h_ref: jax.Array,  # [size_m, out size_n1]; the first gmm's output
    # Scratch memory
    partial_out1_ref: jax.Array,
    acc1_ref: jax.Array,
    partial_out2_ref: jax.Array,
    acc2_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """Two chained gmms over the same groups in one call (see gmm_v2_mlp)."""
    num_gm = fill_metadata(
        lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs1
    )

    # The second gmm's first rhs block (a whole expert, see _mlp_fusable)
    # streams in under the first gmm's last compute step instead of after it.
    def run(window_ref, sem_ref):
        first = pltpu.make_async_copy(
            rhs2_ref.weight.at[metadata_ref.gm_id_to_group_id[0]],
            window_ref.at[0],
            sem_ref,
        )
        has_work = num_gm > 0
        # Started on the first gmm's last step: HBM is idle there, while
        # earlier it would slow the first gmm's own weight stream.
        run_gmm_pipeline(
            lhs_ref,
            rhs1_ref,
            h_ref,
            partial_out1_ref,
            acc1_ref,
            metadata_ref,
            num_gm,
            cfgs=cfgs1,
            last_step_hook=first.start,
        )

        @pl.when(has_work)
        def _():
            first.wait()

        run_gmm_pipeline(
            h_ref,
            rhs2_ref,
            out_ref,
            partial_out2_ref,
            acc2_ref,
            metadata_ref,
            num_gm,
            cfgs=cfgs2,
            rhs_weight_window=window_ref,
        )

    tiles2 = cfgs2.tiles
    pl.run_scoped(
        run,
        pltpu.VMEM((2, tiles2.tile_k, tiles2.tile_n), rhs2_ref.weight.dtype),
        pltpu.SemaphoreType.DMA(()),
    )


def rhs_weight_block_bytes(tk: int, tn: int, rhs_cfgs: InputConfigs) -> int:
    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype
    return tk * tn * jax.dtypes.itemsize_bits(rhs_dtype) // 8


def gmm_vmem_estimate(
    tm: int,
    tn: int,
    tk: int,
    *,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    fuse_act: str | None,
) -> int:
    """VMEM a gmm_v2 pipeline with these tiles needs."""
    lhs_bits = jax.dtypes.itemsize_bits(lhs_cfgs.dtype)
    fuse_act_factor = 1 if fuse_act is None else 2
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
    rhs_weight_vmem = rhs_weight_block_bytes(tk, tn, rhs_cfgs)
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
        gmm_vmem_estimate, lhs_cfgs=lhs_cfgs, rhs_cfgs=rhs_cfgs, fuse_act=fuse_act
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
    if rhs_cfgs.dequantize_after_matmul:
        # Skinny (decode) calls: an expert holds a few rows, so let a tile end
        # at the next lhs sublane block instead of running all tile_m rows
        # through the matmul and the per-block scale epilogue.
        sublane = _get_lhs_sublane_size(lhs_cfgs.dtype, dims.size_m)
        bucket_base = max(sublane, tile_m // max_num_buckets)
        if tile_m % bucket_base:
            bucket_base = tile_m
        return TileSizes(
            tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base
        )
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
        new_tile_m = tile_m + bucket_base
        if new_tile_m > dims.size_m:
            break
        if _gmm_vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
            break
        tile_m = new_tile_m

    buckets = None
    max_after = rhs_cfgs.hybrid_after_max_rows
    if max_after:
        # Skinny tiles (scaled after the matmul) end at the next lhs sublane
        # block; the rest at multiples of bucket_base.
        sublane = _get_lhs_sublane_size(lhs_cfgs.dtype, dims.size_m)
        max_after = min(max_after, tile_m)
        small = range(sublane, max_after + 1, sublane)
        large = range(bucket_base, tile_m + 1, bucket_base)
        buckets = tuple(sorted({*small, *(b for b in large if b > max_after)}))
        if buckets[-1] != tile_m:
            buckets = None
    if _fp8_decode_tiles and buckets is None:
        decode_tiles = _fp8_decode_tile_sizes(
            dims, lhs_cfgs, rhs_cfgs, tile_m, tile_k, tile_n, bucket_base, num_k_tiles
        )
        if decode_tiles is not None:
            return decode_tiles
    # A small-size_m tile_m that did not grow makes every group pay for tile_m
    # rows, and rows cost MXU time on the fp8 rhs feed: use fp8-sized tiles in
    # two buckets. Two such tiles cost what one bf16 tile does, and more
    # buckets would bloat the program (each bucket is a copy of the body).
    # "fp8_feed_wide" takes them up to FP8_RHS_WIDE_MAX_M rows whether or not
    # tile_m grew: a grown tile_m (buckets of bucket_base 64) or one that VMEM
    # held at 64 would otherwise drop every bucket above FP8_RHS_MAX_BUCKET_M
    # to the bf16 dequantize-before-matmul path.
    sublane = _get_lhs_sublane_size(lhs_cfgs.dtype, dims.size_m)
    if fp8_rhs_feed_dtype(lhs_cfgs, rhs_cfgs) is not None and (
        (
            bucket_base == tile_m
            and dims.size_m < 2 * tile_m  # growth stopped by size_m, not VMEM
        )
        or (
            _skinny_gmm_path == "fp8_feed_wide"
            and buckets is None
            and dims.size_m <= FP8_RHS_WIDE_MAX_M
        )
    ):
        if tile_m > FP8_RHS_MAX_BUCKET_M and tile_m % FP8_RHS_TILE_M == 0:
            tile_m = bucket_base = FP8_RHS_TILE_M
        # Partial buckets assume a single k tile (run_matmul_step).
        if num_k_tiles == 1 and tile_m == 2 * sublane:
            bucket_base = sublane

    tiles = TileSizes(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        bucket_base=bucket_base,
        buckets=buckets,
    )
    if envs.TPU_GMM_V2_TUNED_TILES:
        tiles = tuned_tiles(
            tiles, dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act
        )
    return tiles


# From this many rows, a decode gate_up gmm moves an expert's whole gate_up
# width per grid step (one 2-deep weight stream, out of the fused MLP kernel):
# at the MoE decode shapes (H 6144, I 3072, 16 local experts) nearly every
# expert is hit, and halving the grid steps beats the fused kernel's
# prefetched down block.
TUNED_FULL_N_MIN_M = 256
# Grown prefill tiles end at multiples of this many rows: 64-row buckets add
# two more dequantize-before-matmul bodies to the program and cost 11-27% at
# 4096-32768 rows, while 128-row tiles of a 64-row group cost no more.
TUNED_PREFILL_BUCKET_M = 128


def tuned_tiles(
    tiles: TileSizes,
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None,
) -> TileSizes:
    """TPU_GMM_V2_TUNED_TILES: retile fp4-rhs, bf16-lhs calls.

    Only the pipeline depth, tile_n and the bucket sizes change. tile_k stays
    whole and a row's MXU feed (fp8 at <= FP8_RHS_MAX_BUCKET_M rows, else bf16
    dequantize-before) is unchanged, so every output element is computed the
    same way.
    """
    if fp8_rhs_feed_dtype(lhs_cfgs, rhs_cfgs) is None or tiles.tile_k != dims.size_k:
        return tiles
    if tiles.tile_m <= FP8_RHS_MAX_BUCKET_M:
        # Decode fp8 tiles: gate_up only (down's fused-MLP window is 2-deep).
        if fuse_act is None:
            return tiles
        new = dataclasses.replace(tiles, rhs_buffer_count=2)
        if dims.size_m >= TUNED_FULL_N_MIN_M:
            num_lanes = pltpu.get_tpu_info().num_lanes
            full = dataclasses.replace(
                new, tile_n=align_to(dims.size_n // 2, num_lanes)
            )
            block = rhs_weight_block_bytes(full.tile_k, full.tile_n, rhs_cfgs)
            vmem = (
                gmm_vmem_estimate(
                    full.tile_m,
                    full.tile_n,
                    full.tile_k,
                    lhs_cfgs=lhs_cfgs,
                    rhs_cfgs=rhs_cfgs,
                    fuse_act=fuse_act,
                )
                - 2 * block  # 2 weight blocks in flight, not 3 (gate and up)
            )
            if vmem <= vmem_limit_bytes:
                new = full
    elif (
        tiles.buckets is None
        and tiles.bucket_base < TUNED_PREFILL_BUCKET_M
        and tiles.tile_m % TUNED_PREFILL_BUCKET_M == 0
    ):
        new = dataclasses.replace(tiles, bucket_base=TUNED_PREFILL_BUCKET_M)
    else:
        return tiles
    logger.info_once(
        "TPU_GMM_V2_TUNED_TILES=1: gmm_v2 retiled (first: size_m %d, %s -> %s)",
        dims.size_m,
        tiles,
        new,
    )
    return new


def _fp8_decode_tile_sizes(
    dims, lhs_cfgs, rhs_cfgs, tile_m, tile_k, tile_n, bucket_base, num_k_tiles
) -> TileSizes | None:
    """r6's decode tiles (`set_fp8_decode_tiles`): a decode-sized buffer (the
    average group, size_m / size_group rows, fits one tile) keeps small
    groups on the fp8 feed instead of a >= 64-row bf16 bucket. Partial
    buckets assume a single k tile (run_matmul_step), and 16-row buckets a
    sublane tiling of at most 16 rows. None where they do not apply."""
    sublane = _get_lhs_sublane_size(lhs_cfgs.dtype, dims.size_m)
    if not (
        tile_m > bucket_base
        and tile_m >= FP8_DECODE_TILE_M
        and dims.size_m <= FP8_DECODE_TILE_M * dims.size_group
        and dims.size_m <= FP8_DECODE_MAX_M
        and num_k_tiles == 1
        and FP8_DECODE_BUCKETS[0] % sublane == 0
        and fp8_rhs_feed_dtype(lhs_cfgs, rhs_cfgs) is not None
    ):
        return None
    if dims.size_m <= FP8_SMALL_DECODE_MAX_M:
        return TileSizes(
            tile_m=FP8_RHS_TILE_M,
            tile_k=tile_k,
            tile_n=tile_n,
            bucket_base=FP8_RHS_TILE_M // 2,
        )
    return TileSizes(
        tile_m=FP8_DECODE_TILE_M,
        tile_k=tile_k,
        tile_n=tile_n,
        bucket_base=FP8_DECODE_BUCKETS[0],
        buckets=FP8_DECODE_BUCKETS,
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

    after_matmul_ok = (
        has_scale
        and not maybe_quantize_lhs
        and block_size >= MIN_DEQUANT_AFTER_MATMUL_BLOCK
    )
    skinny = after_matmul_ok and dims.size_m <= SKINNY_DEQUANT_AFTER_MATMUL_M
    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
        dequantize_after_matmul=skinny,
        hybrid_after_max_rows=(
            HYBRID_DEQUANT_AFTER_MATMUL_MAX_ROWS
            if after_matmul_ok and not skinny
            else 0
        ),
    )

    if (after_matmul_ok or skinny) and fp8_rhs_feed_dtype(
        InputConfigs(quant_dtype=None, quant_block_size=512, dtype=lhs.dtype),
        dataclasses.replace(
            rhs_cfgs, dequantize_after_matmul=False, hybrid_after_max_rows=0
        ),
    ) is not None:
        # The fp8 rhs feed (`skinny_gmm_path`) takes the skinny buckets the
        # bf16 after-matmul paths would.
        rhs_cfgs = dataclasses.replace(
            rhs_cfgs, dequantize_after_matmul=False, hybrid_after_max_rows=0
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
        quant_block_size=512,
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


def pipeline_scratch_shapes(cfgs: GmmConfigs) -> list:
    """VMEM scratch of run_gmm_pipeline: partial_out_ref, acc_ref."""
    tiles = cfgs.tiles
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    return [
        pltpu.VMEM((cfgs.dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
    ]


def metadata_scratch_shape(cfgs: GmmConfigs) -> MetadataRef:
    dims = cfgs.dims
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, cfgs.tiles.tile_m) - 1
    return MetadataRef(
        gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
        gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
    )


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

    # The kernel reshapes lhs by its sublane block, so pad the rows up to it.
    # The extra rows sit past every group, so they add no work, and they are
    # sliced back off the output below.
    size_m = lhs.shape[0]
    size_lhs_sublane = _get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = align_to(size_m, size_lhs_sublane)
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

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

    # Prepare block specs.
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Initialize scratch shapes.
    scratch_shapes = [*pipeline_scratch_shapes(cfgs), metadata_scratch_shape(cfgs)]

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


# kernel_mlp hands down's pipeline a prefetched block (newer Pallas TPU API).
_HAS_PIPELINE_PREFETCH = "prefetched_count" in {
    f.name for f in dataclasses.fields(pl.Buffered)
} and hasattr(pltpu, "emit_pipeline_with_allocations")


def _mlp_tiles(cfgs: GmmConfigs) -> bool:
    """Decode-sized tiles kernel_mlp fuses: fp8 rhs feed at tile_m, or (with
    `set_fp8_decode_tiles`, r6) up to FP8_DECODE_TILE_M rows whose smallest
    bucket takes the fp8 feed."""
    tiles = cfgs.tiles
    if _fp8_decode_tiles:
        return (
            tiles.tile_m <= FP8_DECODE_TILE_M
            and cfgs.fp8_rhs_dtype(tiles.bucket_sizes[0]) is not None
        )
    return cfgs.fp8_rhs_dtype(tiles.tile_m) is not None


def _mlp_fusable(cfgs1: GmmConfigs, cfgs2: GmmConfigs, vmem_limit_bytes: int) -> bool:
    """Whether kernel_mlp can run both gmms on one fill_metadata layout."""

    def vmem(cfgs):
        t = cfgs.tiles
        return gmm_vmem_estimate(
            t.tile_m,
            t.tile_n,
            t.tile_k,
            lhs_cfgs=cfgs.lhs_cfgs,
            rhs_cfgs=cfgs.rhs_cfgs,
            fuse_act=cfgs.fuse_act,
        )

    # The prefetch window (2 down weight blocks) lives through both pipelines;
    # down's pipeline double-buffers its weights in it.
    t2 = cfgs2.tiles
    block2 = rhs_weight_block_bytes(t2.tile_k, t2.tile_n, cfgs2.rhs_cfgs)
    window = 2 * block2

    # kernel_mlp's scratch keeps both pipelines' partial_out and acc resident.
    def scratch(cfgs):
        return sum(
            math.prod(x.shape) * jnp.dtype(x.dtype).itemsize
            for x in pipeline_scratch_shapes(cfgs)
        )

    fits = (
        vmem(cfgs1) + scratch(cfgs2) + window <= vmem_limit_bytes
        and vmem(cfgs2) - 3 * block2 + scratch(cfgs1) + window <= vmem_limit_bytes
    )
    return (
        fits
        and
        # Decode-sized (fp8 rhs feed) tiles only: the fused program holds both
        # gmms' bucket bodies, and large ones would bloat it.
        _mlp_tiles(cfgs1)
        and _mlp_tiles(cfgs2)
        and not cfgs1.zero_init
        and not cfgs2.zero_init
        and cfgs1.tiles.tile_m == cfgs2.tiles.tile_m
        and cfgs1.dims.size_lhs_sublane == cfgs2.dims.size_lhs_sublane
        and cfgs1.dims.size_group == cfgs2.dims.size_group
        and cfgs1.out_size_n == cfgs2.dims.size_k
        # kernel_mlp prefetches one whole expert of the second rhs.
        and cfgs2.fuse_act is None
        and not cfgs2.rhs_cfgs.should_unpack
        and cfgs2.tiles.tile_k == cfgs2.dims.size_k
        and cfgs2.tiles.tile_n == cfgs2.dims.size_n
        and cfgs1.out_size_n % pltpu.get_tpu_info().num_lanes == 0
    )


@jax.jit(static_argnames=["fuse_act", "preferred_element_type", "maybe_quantize_lhs"])
def gmm_v2_mlp(
    lhs: jax.Array,  # [size_m, size_k]
    rhs1: jax.Array,  # [size_group, size_k, size_n1]
    rhs2: jax.Array,  # [size_group, size_n1 // (2 if fuse_act else 1), size_n2]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs1_scale: jax.Array | None = None,
    rhs2_scale: jax.Array | None = None,
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    fuse_act: str | None = None,
    preferred_element_type: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
) -> jax.Array:
    """gmm_v2(gmm_v2(lhs, rhs1, fuse_act), rhs2) for an MoE expert MLP.

    For decode-sized tiles the two run in one kernel (kernel_mlp), which loads
    the second gmm's first weight block under the first gmm's compute: as two
    calls, each exposes its first block's DMA. Otherwise it is two gmm_v2 calls.
    """

    def two_calls():
        h = gmm_v2(
            lhs,
            rhs1,
            group_sizes,
            rhs_scale=rhs1_scale,
            group_offset=group_offset,
            zero_initialize=False,
            fuse_act=fuse_act,
            maybe_quantize_lhs=maybe_quantize_lhs,
        )
        return gmm_v2(
            h[:, : rhs2.shape[1]],
            rhs2,
            group_sizes,
            rhs_scale=rhs2_scale,
            group_offset=group_offset,
            zero_initialize=False,
            preferred_element_type=preferred_element_type,
            maybe_quantize_lhs=maybe_quantize_lhs,
        )

    size_m = lhs.shape[0]
    padded_size_m = align_to(size_m, _get_lhs_sublane_size(lhs.dtype, size_m))
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    common = dict(
        tile_info=calculate_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        acc_dtype=None,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=False,
    )
    cfgs1 = make_gmm_configs(
        lhs,
        rhs1,
        rhs1_scale,
        None,
        group_sizes,
        group_offset,
        out_dtype=None,
        fuse_act=fuse_act,
        **common,
    )
    if cfgs1.out_size_n != rhs2.shape[1] or not _HAS_PIPELINE_PREFETCH:
        return two_calls()[:size_m]
    h_shape = jax.ShapeDtypeStruct((padded_size_m, cfgs1.out_size_n), cfgs1.out_dtype)
    cfgs2 = make_gmm_configs(
        h_shape,
        rhs2,
        rhs2_scale,
        None,
        group_sizes,
        group_offset,
        out_dtype=preferred_element_type,
        **common,
    )
    if not _mlp_fusable(cfgs1, cfgs2, vmem_limit_bytes):
        return two_calls()[:size_m]

    def weights_ref(rhs, rhs_scale):
        if rhs_scale is not None:
            rhs_scale = rhs_scale.astype(jnp.float32)
        spec = WeightsRef(
            weight=pl.BlockSpec(memory_space=pltpu.HBM),
            scale=None if rhs_scale is None else pl.BlockSpec(memory_space=pltpu.HBM),
            bias=None,
        )
        return WeightsRef(weight=rhs, scale=rhs_scale, bias=None), spec

    rhs1_ref, rhs1_spec = weights_ref(rhs1, rhs1_scale)
    rhs2_ref, rhs2_spec = weights_ref(rhs2, rhs2_scale)
    num_lanes = pltpu.get_tpu_info().num_lanes
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    cost1, cost2 = get_cost_estimate(cfgs1), get_cost_estimate(cfgs2)
    out, _ = pl.pallas_call(
        functools.partial(kernel_mlp, cfgs1=cfgs1, cfgs2=cfgs2),
        out_shape=(
            jax.ShapeDtypeStruct(
                (padded_size_m, align_to(cfgs2.out_size_n, num_lanes)),
                cfgs2.out_dtype,
            ),
            h_shape,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm, rhs1_spec, rhs2_spec],
            out_specs=[hbm, hbm],
            scratch_shapes=[
                *pipeline_scratch_shapes(cfgs1),
                *pipeline_scratch_shapes(cfgs2),
                metadata_scratch_shape(cfgs1),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=f"mlp-{get_scope_name(cfgs1)}-{get_scope_name(cfgs2)}",
        cost_estimate=pl.CostEstimate(
            flops=cost1.flops + cost2.flops,
            bytes_accessed=cost1.bytes_accessed + cost2.bytes_accessed,
            transcendentals=0,
        ),
        metadata={
            **{f"gmm1.{k}": v for k, v in get_metadata(cfgs1).items()},
            **{f"gmm2.{k}": v for k, v in get_metadata(cfgs2).items()},
        },
    )(group_sizes, group_offset, lhs, rhs1_ref, rhs2_ref)
    return out[:size_m, : cfgs2.out_size_n]
