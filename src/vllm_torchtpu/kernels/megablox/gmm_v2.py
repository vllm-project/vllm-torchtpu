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

import contextlib
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

    def select(self, i: int) -> "WeightsRef":
        """Expert ``i`` of a multi-expert block (leading dim), as a plain ref."""
        return WeightsRef(
            weight=self.weight.at[i],
            scale=None if self.scale is None else self.scale.at[i],
            bias=None if self.bias is None else self.bias.at[i],
        )


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

    def select(self, i: int) -> "FusedWeightsRef":
        return FusedWeightsRef(gate=self.gate.select(i), up=self.up.select(i))


def split_part(n0: int, n1: int, part_width: int) -> tuple[int, int, int]:
    """(part, local start, local end) of output columns [n0, n1), which must
    lie inside one part of ``part_width`` columns."""
    part = n0 // part_width
    if n1 > (part + 1) * part_width or n1 <= n0:
        raise ValueError(f"columns [{n0}, {n1}) cross a {part_width}-column part")
    return part, n0 - part * part_width, n1 - part * part_width


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SplitWeightsRef(RhsRef):
    """One weight tile delivered as several column parts, each its own
    pipeline input, so each part is its own concurrent DMA. Blocks are read
    from the part that holds them; the values are those of the whole tile."""

    parts: tuple

    @property
    def part_width(self) -> int:
        return self.parts[0].weight.shape[-1]

    def get_weight(self) -> jax.Array:
        return jnp.concatenate([p.get_weight() for p in self.parts], axis=-1)

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        return jnp.concatenate([p.get_scale(replicate_size) for p in self.parts], axis=-1)

    def get_bias(self) -> jax.Array:
        return jnp.concatenate([p.get_bias() for p in self.parts], axis=-1)

    def weight_block(self, k0: int, k1: int, n0: int, n1: int) -> jax.Array:
        part, a, b = split_part(n0, n1, self.part_width)
        return self.parts[part].weight[k0:k1, a:b]

    def scale_block(self, b_id, replicate_size: int, n0: int, n1: int) -> jax.Array:
        part, a, b = split_part(n0, n1, self.part_width)
        return self.parts[part].get_scale(replicate_size=replicate_size)[b_id, :, a:b]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array
    # experts_per_visit > 1: global row bounds of each expert of a visit's
    # block, E + 1 per gm tile; unused (size 1) otherwise.
    gm_id_to_sub_bounds: jax.Array


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
    # (start, width) output columns computed from rhs[..., start:start + width];
    # dims.size_n is then the window width. None computes every column.
    n_window: tuple[int, int] | None = None
    # Quantize each lhs k-block once per tile, before the n loop, instead of
    # inside it (where the same ops repeat tile_n // mxu_size times and the
    # repeats are left to CSE). Same ops on the same values.
    hoist_lhs_quant: bool = False
    # Phase trace scopes inside each visit (0 off; >= 2 adds gmm_quant, gmm_mxu,
    # gmm_epilogue under the existing per-visit bm<rows>_* scopes). They appear
    # on xprof's "XLA TraceMe" line only with libtpu's
    # --xla_enable_custom_call_region_trace=true. Profiling builds only: a
    # scoped build is not guaranteed bit-identical.
    trace_scopes: int = 0
    # Deliver each weight tile as this many column parts (one pipeline input,
    # hence one concurrent DMA, each). Non-fused GMM with one N tile only.
    rhs_split: int = 1
    # Consecutive experts per pipeline step (one block of E experts' weights,
    # one visit over their rows). Amortizes the per-step fixed cost.
    experts_per_visit: int = 1
    # Store the bf16 output as uint32 [M, N/2], word j = bits(col j) << 16 |
    # bits(col j + N/2): each row's values in whole 32-bit words, so a row
    # gather (the SparseCore combine) moves only that row. Same bf16 values.
    pack_out: bool = False

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


def trace_scope(cfgs: "GmmConfigs", name: str, level: int):
    """A named device scope when cfgs.trace_scopes >= level, else a no-op."""
    return jax.named_scope(name) if cfgs.trace_scopes >= level else contextlib.nullcontext()


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

    # One expert per step squeezes the group dim; a multi-expert visit loads a
    # block of E consecutive experts (block index = the visit's block id).
    lead = None if cfgs.experts_per_visit == 1 else cfgs.experts_per_visit
    rhs_weight_spec = pl.BlockSpec(
        (lead, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (lead, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (lead, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_cols = cfgs.tiles.tile_n // 2 if cfgs.pack_out else cfgs.tiles.tile_n
    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, out_cols),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


def pack_halves(v: jax.Array) -> jax.Array:
    """bf16 [..., n] -> uint32 [..., n/2]: word j = bits(v[j]) << 16 | bits(v[j + n/2]).

    bf16 -> f32 appends 16 zero bits, so the f32 word holds the bf16 bits in
    its high half; only 32-bit ops are used."""
    h = v.shape[-1] // 2
    hi = lax.bitcast_convert_type(v[..., :h].astype(jnp.float32), jnp.uint32)
    lo = lax.bitcast_convert_type(v[..., h:].astype(jnp.float32), jnp.uint32)
    return jnp.bitwise_or(
        jnp.bitwise_and(hi, jnp.uint32(0xFFFF0000)), jnp.right_shift(lo, jnp.uint32(16))
    )


def unpack_halves(u: jax.Array) -> jax.Array:
    """Inverse of pack_halves: uint32 [..., h] -> bf16 [..., 2h] (exact)."""
    hi = lax.bitcast_convert_type(jnp.bitwise_and(u, jnp.uint32(0xFFFF0000)), jnp.float32)
    lo = lax.bitcast_convert_type(jnp.left_shift(u, jnp.uint32(16)), jnp.float32)
    return jnp.concatenate([hi.astype(jnp.bfloat16), lo.astype(jnp.bfloat16)], axis=-1)


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
    tiled_rhs_ref_ = tiled_rhs_ref

    def _matmul(
        is_first_k_step: bool,
        is_last_k_step: bool,
        bucket_m: int,
        rhs_ref: RhsRef | None = None,
        row_start: jax.Array | None = None,
    ):
        # Sub-expert mode (experts_per_visit > 1): rows [row_start,
        # row_start + bucket_m) against one expert of the block; returns the
        # pre-epilogue accumulator (single k step), the caller merges.
        sub_mode = row_start is not None
        tiled_rhs_ref = tiled_rhs_ref_ if rhs_ref is None else rhs_ref
        tpu_info = pltpu.get_tpu_info()
        mxu_size = tpu_info.mxu_column_size

        # Step 1: Input pre-processing.
        if sub_mode:
            tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[
                pl.ds(row_start, bucket_m)
            ]
        else:
            tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
        split = isinstance(tiled_rhs_ref, SplitWeightsRef)
        # A split tile is read block by block from its parts (rhs_block below).
        tiled_rhs = None if split else tiled_rhs_ref.get_weight()

        # This should only be taken in the case where we don't requantize
        # the scales and thus we need to dequantize inside VMEM to avoid small
        # contracting dimmensions
        rhs_tile_n = cfgs.tiles.tile_n if split else tiled_rhs.shape[1]

        def rhs_block(k0, k1, n0, n1):
            if split:
                return tiled_rhs_ref.weight_block(k0, k1, n0, n1)
            return tiled_rhs[k0:k1, n0:n1]

        def rhs_scale_block(b_id, n0, col):
            if split:
                return tiled_rhs_ref.scale_block(b_id, bucket_m, n0, n0 + col)
            return tiled_rhs_ref.get_scale(replicate_size=bucket_m)[b_id, :, n0 : n0 + col]
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

                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                    end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)

                    block_acc = jnp.matmul(
                        tiled_lhs[:, start_k:end_k],
                        rhs_block(start_k, end_k, start_n, end_n),
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)

                    if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_replicated = rhs_scale_block(b_id, start_n, col_size)
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
            def quantize_block(start_k):
                with trace_scope(cfgs, "gmm_quant", 2):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)
                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
                    block_scale = block_abs_max / dtype_max

                    # If block_scale=0, it will cause division by zero and return either
                    # NaN or Inf. Since this can cause numeric issue when downcasting to
                    # quantized value, we convert them into 0.
                    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
                    # Convert lhs into quantized dtype.
                    return (block_lhs * block_scale_inv).astype(lhs_q_dtype), block_scale

            hoisted = (
                {k: quantize_block(k) for k in range(0, cfgs.tiles.tile_k, q_block_size)}
                if cfgs.hoist_lhs_quant
                else None
            )

            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                    block_rhs = rhs_block(start_k, end_k, start_n, end_n)

                    # Perform lhs quantization. Unless hoisted, for every block_lhs
                    # the same computation is performed tiles_n//mxu_size times and
                    # the repeats are left to the compiler's CSE.
                    if hoisted is not None:
                        block_lhs_q, block_scale = hoisted[start_k]
                    else:
                        block_lhs_q, block_scale = quantize_block(start_k)

                    # Unlike unquantized path, compiler may not perform implicit type
                    # conversion due to numeric concerns. As this can cause unsupported
                    # matmul error, explicit type conversion is performed.
                    if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
                        block_rhs = block_rhs.astype(lhs_q_dtype)

                    with trace_scope(cfgs, "gmm_mxu", 2):
                        block_acc = jnp.matmul(
                            block_lhs_q,
                            block_rhs,
                            preferred_element_type=preferred_element_type,
                        ).astype(acc_ref.dtype)

                    block_acc *= block_scale.astype(acc_ref.dtype)

                    # Apply rhs subchannel scale per quant block.
                    if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_replicated = rhs_scale_block(b_id, start_n, col_size)
                        block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        acc = jnp.concatenate(acc_list, axis=1)
        if sub_mode:
            return acc

        # Step 3: Output post-processing.
        epilogue_scope = trace_scope(cfgs, "gmm_epilogue", 2)
        epilogue_scope.__enter__()
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
            out_cols = tiled_out_ref.shape[-1]
            tiled_out_2d_ref = tiled_out_ref.reshape(-1, out_cols)
            if cfgs.pack_out:
                tiled_out_2d_ref[:acc_m] = pack_halves(acc_masked.astype(cfgs.out_dtype))
            else:
                tiled_out_2d_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype)

            # If this is the first tile for grid[n_id, :, :], we initialize the
            # partial out to zeros. Otherwise, partial out from last tile of
            # grid[n_id-1, :, :] can be used and cause numeric issues.
            partial_out_zeros = jnp.zeros_like(partial_out_ref)

            # Accumulate the partial output from the previous step.
            if cfgs.pack_out:
                # the same bf16 add, on the unpacked values
                tiled_out_ref[0] = pack_halves(
                    unpack_halves(tiled_out_ref[0])
                    + unpack_halves(
                        jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])
                    )
                )
            else:
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
        epilogue_scope.__exit__(None, None, None)

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

        if cfgs.tiles.tile_k >= cfgs.dims.size_k:
            # One k step: only first_last can run; emit just it (code size
            # grows with the number of buckets).
            matmul_first_last()
        elif bucket_m == cfgs.tiles.tile_m:
            lax.cond(
                is_first_k_step,
                lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
                lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
            )
        else:
            # partial m is only invoked at last matmul.
            lax.cond(is_first_k_step, matmul_first_last, matmul_last)

    if cfgs.experts_per_visit > 1:
        _multi_expert_visit(
            _matmul,
            tiled_rhs_ref,
            tiled_out_ref,
            partial_out_ref,
            metadata_ref,
            gm_id=gm_id,
            m_offset=m_offset,
            m_start_local=m_start_local,
            m_end_local=m_end_local,
            cfgs=cfgs,
        )
        return

    branches = []
    for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base):
        branches.append(functools.partial(run_matmul_step, bucket_idx=bucket_idx))
    # Smallest bucket covering rows [0, m_end_local).
    bucket_idx = (m_end_local - 1) // cfgs.tiles.bucket_base
    lax.switch(bucket_idx, branches)


def sub_expert_window(lo, hi, *, sublane: int, bucket_base: int, tile_m: int):
    """Row window [start, start + bucket) of one sub-expert's rows [lo, hi).

    start is lo rounded down to the sublane tile; the bucket is the smallest
    multiple of bucket_base covering [start, hi), and start is pulled back so
    the window stays inside the tile. Returns (bucket_idx, start) for the
    bucket bucket_base * (bucket_idx + 1). Works on ints and int arrays.
    """
    num_buckets = tile_m // bucket_base
    start = (lo // sublane) * sublane
    need = hi - start
    bucket_idx = (need - 1) // bucket_base
    if isinstance(bucket_idx, int):
        bucket_idx = max(0, min(num_buckets - 1, bucket_idx))
        start = min(start, tile_m - bucket_base * (bucket_idx + 1))
    else:
        bucket_idx = jnp.clip(bucket_idx, 0, num_buckets - 1)
        start = jnp.minimum(start, tile_m - bucket_base * (bucket_idx + 1))
    return bucket_idx, start


def _multi_expert_visit(
    matmul,
    tiled_rhs_ref,
    tiled_out_ref,
    partial_out_ref,
    metadata_ref,
    *,
    gm_id,
    m_offset,
    m_start_local,
    m_end_local,
    cfgs: GmmConfigs,
):
    """One pipeline step over a block of E consecutive experts' rows.

    Each sub-expert s with rows [lo, hi) in this tile runs the base matmul on
    a sublane-aligned window covering [lo, hi) with its own weights, and its
    rows are merged into the output tile: rows below lo keep what earlier
    sub-experts wrote (zero below the tile's first row), rows from hi up are
    zero. Each row is computed by the same ops as the one-expert kernel.

    Signed zeros: the one-expert kernel adds the carried partial sublane
    (+0.0 on the current expert's rows) to the first sublane group of each of
    its tiles, and the next tile adds a group-sharing expert's last partial
    group to +0.0. Both turn -0.0 into +0.0 on exactly those rows. The merge
    adds a (runtime) +0.0 to the same rows (rule A: first group of each
    one-expert tile; rule B: last partial group when a later tile exists),
    and the carry here only touches rows below the tile's first row, so the
    output is bitwise equal including the sign of zero.
    """
    e = cfgs.experts_per_visit
    sub = e + 1
    sublane = cfgs.dims.size_lhs_sublane
    out_cols = tiled_out_ref.shape[-1]
    bucket_base = cfgs.tiles.bucket_base
    num_buckets = cfgs.tiles.tile_m // bucket_base
    tiled_out_2d_ref = tiled_out_ref.reshape(-1, out_cols)

    tile_m = cfgs.tiles.tile_m
    total = metadata_ref.gm_id_to_m_offset[pl.num_programs(1)]
    # +0.0 that the compiler cannot fold away (x + 0.0 -> x drops the sign rule).
    dyn_zero = jnp.where(gm_id < 0, 1.0, 0.0).astype(cfgs.out_dtype)

    def gbound(j):
        return metadata_ref.gm_id_to_sub_bounds[gm_id * sub + j]

    def sub_expert(s, carry):
        # A loop, not an unrolled block: one copy of the bucketed matmul for
        # all E sub-experts keeps the kernel's code size near the base's.
        gstart = gbound(s)
        gend = gbound(s + 1)
        lo = jnp.clip(gstart - m_offset, m_start_local, m_end_local)
        hi = jnp.clip(gend - m_offset, m_start_local, m_end_local)
        rhs_s = tiled_rhs_ref.select(s)

        def run_sub(bucket_idx):
            bucket_m = bucket_base * (bucket_idx + 1)

            @jax.named_scope(f"bm{bucket_m}_sub")
            def body():
                _, start = sub_expert_window(
                    lo, hi, sublane=sublane, bucket_base=bucket_base, tile_m=tile_m
                )
                start = pl.multiple_of(start, sublane)
                acc = matmul(True, True, bucket_m, rhs_ref=rhs_s, row_start=start)
                if cfgs.rhs_cfgs.has_bias:
                    acc += rhs_s.get_bias().astype(acc.dtype)
                acc = apply_act_fn(acc, cfgs.fuse_act)
                rows = start + lax.broadcasted_iota(jnp.int32, acc.shape, 0)
                val = acc.astype(cfgs.out_dtype)
                grow = m_offset + rows
                norm_a = (grow - (gstart // sublane) * sublane) % tile_m < sublane
                norm_b = jnp.logical_and(
                    grow >= (gend // sublane) * sublane,
                    jnp.logical_and(gend % sublane != 0, gend != total),
                )
                val = jnp.where(jnp.logical_or(norm_a, norm_b), val + dyn_zero, val)
                if cfgs.pack_out:
                    val = pack_halves(val)  # rows stay rows: row masks apply as is
                rows = start + lax.broadcasted_iota(jnp.int32, val.shape, 0)
                mine = jnp.logical_and(lo <= rows, rows < hi)
                keep = jnp.logical_and(rows < lo, lo > m_start_local)
                old = tiled_out_2d_ref[pl.ds(start, bucket_m)]
                zeros = jnp.zeros_like(old)
                tiled_out_2d_ref[pl.ds(start, bucket_m)] = jnp.where(
                    mine, val, jnp.where(keep, old, zeros)
                )

            body()

        @pl.when(hi > lo)
        def _():
            bucket_idx, _ = sub_expert_window(
                lo, hi, sublane=sublane, bucket_base=bucket_base, tile_m=tile_m
            )
            lax.switch(
                bucket_idx,
                [functools.partial(run_sub, i) for i in range(num_buckets)],
            )

        return carry

    lax.fori_loop(0, e, sub_expert, 0)

    # Partial-sublane carry as in the one-expert last-k epilogue, applied only
    # to the rows below this tile's first row (the previous tile's rows).
    partial_out_zeros = jnp.zeros_like(partial_out_ref)
    first_group = tiled_out_ref[0]
    group_rows = lax.broadcasted_iota(jnp.int32, first_group.shape, 0)
    carry_in = jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])
    if cfgs.pack_out:
        carried = pack_halves(unpack_halves(first_group) + unpack_halves(carry_in))
    else:
        carried = first_group + carry_in
    tiled_out_ref[0] = jnp.where(group_rows < m_start_local, carried, first_group)
    last_row = m_end_local // sublane
    partial_out_ref[...] = jnp.where(
        m_end_local % sublane == 0,
        partial_out_zeros,
        tiled_out_ref[last_row],
    )


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


def fill_metadata_blocks(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """fill_metadata for experts_per_visit = E > 1 (non-EP: group offset 0).

    Groups are taken in blocks of E consecutive experts. Each block's rows
    (contiguous, since lhs rows are sorted by expert) are cut into m tiles
    exactly as fill_metadata cuts one group's rows; every tile records the
    block id (its rhs block index) and the global row bounds of the block's E
    experts, so the kernel can give each row its own expert's weights.
    """
    e = cfgs.experts_per_visit
    num_blocks = cfgs.dims.size_group // e
    sub = e + 1
    metadata_ref.gm_id_to_m_offset[0] = 0

    def outer_block_loop(block_id, carry):
        num_gm, start_m_offset = carry
        sizes = [lhs_group_sizes_ref[block_id * e + j] for j in range(e)]
        block_size = functools.reduce(lambda a, b: a + b, sizes)
        end_m_offset = start_m_offset + block_size
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        curr_num_gm = pl.cdiv(block_size + local_offset, cfgs.tiles.tile_m)
        curr_num_gm = jnp.where(block_size > 0, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        def tm_loop(tm_id, curr_m_offset):
            lo = curr_m_offset % cfgs.dims.size_lhs_sublane
            tm_size = jnp.minimum(cfgs.tiles.tile_m - lo, end_m_offset - curr_m_offset)
            metadata_ref.gm_id_to_group_id[tm_id] = block_id
            metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
            metadata_ref.gm_id_to_m_offset[tm_id + 1] = curr_m_offset + tm_size
            bound = start_m_offset
            metadata_ref.gm_id_to_sub_bounds[tm_id * sub] = bound
            for j in range(e):
                bound = bound + sizes[j]
                metadata_ref.gm_id_to_sub_bounds[tm_id * sub + j + 1] = bound
            return curr_m_offset + tm_size

        lax.fori_loop(num_gm, next_num_gm, tm_loop, start_m_offset)
        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, num_blocks, outer_block_loop, (0, 0))
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

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if cfgs.rhs_cfgs.should_unpack:
        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    if cfgs.n_window is not None:
        # A column window of the weights, scales and bias as a ref view (as
        # the fused activation's up half below): the DMAs read only these
        # columns and nothing is copied.
        start, width = cfgs.n_window
        rhs_ref = jax.tree.map(lambda x: x.at[..., start : start + width], rhs_ref)

    # Fill metadata buffer and return number of group & m interations.
    if cfgs.experts_per_visit > 1:
        num_gm = fill_metadata_blocks(lhs_group_sizes_ref, metadata_ref, cfgs=cfgs)
    else:
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

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)

        rhs_spec = FusedWeightsRef(
            gate=rhs_spec,
            up=rhs_spec,
        )
    elif cfgs.rhs_split > 1:
        width = cfgs.tiles.tile_n // cfgs.rhs_split
        rhs_ref = SplitWeightsRef(
            parts=tuple(
                jax.tree.map(lambda x, j=j: x.at[..., j * width : (j + 1) * width], rhs_ref)
                for j in range(cfgs.rhs_split)
            )
        )
        part_spec = jax.tree.map(
            lambda bs: pl.BlockSpec(
                (*bs.block_shape[:-1], width),
                bs.index_map,
                **({"pipeline_mode": bs.pipeline_mode} if getattr(bs, "pipeline_mode", None) else {}),
            ),
            rhs_spec,
            is_leaf=lambda x: isinstance(x, pl.BlockSpec),
        )
        rhs_spec = SplitWeightsRef(parts=(part_spec,) * cfgs.rhs_split)

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, cast(jax.Array, semaphore_ref), zero_size, dims=cfgs.dims)


def fused_mlp_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs1_ref: WeightsRef,  # [size_group, size_k, 2 * inter] (gate | up)
    rhs2_ref: WeightsRef,  # [size_group, inter, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n] (or packed [size_m, size_n // 2])
    # Scratch
    partial1_ref: jax.Array,
    acc1_ref: jax.Array,
    act_ref: jax.Array,  # [tile_m // sublane, sublane, inter]: gmm1's output tile
    partial2_ref: jax.Array,
    acc2_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """Fused expert MLP (gmm1 + activation + gmm2) in one pipeline: each step
    loads one expert's w1 and w2 and runs both GMMs on that visit's rows.

    Both stages are inner_kernel itself, so each runs exactly the two-kernel
    path's ops on the same values: gmm1 writes its visit tile into act_ref (with
    its own first-sublane-group carry, partial1_ref), and gmm2 reads its lhs
    rows from act_ref at the same row offset as it would read gmm1's HBM output.
    Every lhs row gmm2 reads for this visit's own rows holds the value gmm1
    computed; the two-kernel path differs only where the next visit's carry adds
    +0 to this visit's last partial sublane group (-0 becomes +0). gmm2's output
    does not depend on the sign of a zero in its lhs: each row's quantization
    uses |x|, and its accumulator starts at +0, so a zero product sum is +0
    either way. The output is therefore bit-identical (interpret check with
    signed-zero-rich cases: scratch/pmoe/q38_pmoe_interp_fmlp.py).
    """
    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs1)
    (lhs_spec, rhs1_spec), _ = generate_block_specs(metadata_ref, cfgs1)
    (_, rhs2_spec), out_spec = generate_block_specs(metadata_ref, cfgs2)
    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs1.out_size_n :], rhs1_ref)
    rhs1_ref = FusedWeightsRef(gate=rhs1_ref, up=rhs_up_ref)
    rhs1_spec = FusedWeightsRef(gate=rhs1_spec, up=rhs1_spec)
    sub = cfgs1.dims.size_lhs_sublane

    def body(tiled_lhs_ref, tiled_rhs1_ref, tiled_rhs2_ref, tiled_out_ref, p1, a1, act, p2, a2, md):
        inner_kernel(tiled_lhs_ref, tiled_rhs1_ref, act, p1, a1, md, cfgs=cfgs1)
        inner_kernel(act, tiled_rhs2_ref, tiled_out_ref, p2, a2, md, cfgs=cfgs2)

    pipeline_fn = pltpu.emit_pipeline(
        body,
        grid=(1, num_gm, 1),
        in_specs=(lhs_spec, rhs1_spec, rhs2_spec),
        out_specs=out_spec,
    )
    lhs_in = lhs_ref.reshape(-1, sub, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, sub, out_ref.shape[-1])
    pipeline_fn(
        lhs_in, rhs1_ref, rhs2_ref, out_in,
        scratches=[partial1_ref, acc1_ref, act_ref, partial2_ref, acc2_ref, metadata_ref],
    )


def gmm_mlp_v2(
    lhs: jax.Array,  # [size_m, size_k]
    w1: jax.Array,  # [size_group, size_k, 2 * inter]
    w2: jax.Array,  # [size_group, inter, size_n]
    group_sizes: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    *,
    fuse_act: str = "silu",
    tile_info1: "TileSizes | TileFn | None" = None,
    tile_info2: "TileSizes | TileFn | None" = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    pack_out: bool = False,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """gmm_v2(gmm_v2(lhs, w1, fuse_act), w2) as one kernel (fused expert MLP),
    bitwise equal to the two calls with zero_initialize=False, one expert per
    visit and the same tilings. Non-EP only (group offset 0)."""
    tile_info1 = calculate_tiling if tile_info1 is None else tile_info1
    tile_info2 = calculate_tiling if tile_info2 is None else tile_info2
    size_m = lhs.shape[0]
    sub = _get_lhs_sublane_size(lhs.dtype, size_m)
    padded = align_to(size_m, sub)
    if padded != size_m:
        lhs = jnp.pad(lhs, ((0, padded - size_m), (0, 0)))
    group_offset = jnp.array([0], dtype=jnp.int32)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    inter = w1.shape[2] // 2
    common = dict(vmem_limit_bytes=vmem_limit_bytes, acc_dtype=None, maybe_quantize_lhs=True,
                  zero_initialize=False, rhs_quant_dtype=rhs_quant_dtype)
    cfgs1 = make_gmm_configs(lhs, w1, w1_scale, None, group_sizes, group_offset, tile_info=tile_info1,
                             out_dtype=lhs.dtype, fuse_act=fuse_act, **common)
    act_shape = jax.ShapeDtypeStruct((lhs.shape[0], inter), lhs.dtype)
    cfgs2 = make_gmm_configs(act_shape, w2, w2_scale, None, group_sizes, group_offset, tile_info=tile_info2,
                             out_dtype=lhs.dtype, pack_out=pack_out, **common)
    t1, t2 = cfgs1.tiles, cfgs2.tiles
    if not (
        t1.tile_m == t2.tile_m
        and t1.bucket_base == t2.bucket_base
        and t1.tile_k >= cfgs1.dims.size_k
        and t2.tile_k >= cfgs2.dims.size_k
        and t1.tile_n >= cfgs1.out_size_n
        and t2.tile_n >= cfgs2.out_size_n
        and cfgs1.dims.size_lhs_sublane == cfgs2.dims.size_lhs_sublane
    ):
        raise ValueError(f"gmm_mlp_v2 needs one m tiling and single k/n tiles in both GMMs ({t1=}, {t2=})")
    dims = cfgs1.dims
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, t1.tile_m) - 1
    if w1_scale is not None:
        w1_scale = w1_scale.astype(jnp.float32)
    if w2_scale is not None:
        w2_scale = w2_scale.astype(jnp.float32)
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_n = align_to(cfgs2.out_size_n, num_lanes)
    out_init = (
        jax.ShapeDtypeStruct((dims.size_m, aligned_n // 2), jnp.uint32)
        if pack_out
        else jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs2.out_dtype)
    )
    scratch_shapes = [
        pltpu.VMEM((sub, t1.tile_n), cfgs1.out_dtype),  # partial1
        pltpu.VMEM((t1.tile_m, 2 * t1.tile_n), cfgs1.acc_dtype),  # acc1
        pltpu.VMEM((t1.tile_m // sub, sub, cfgs1.out_size_n), cfgs1.out_dtype),  # act
        (
            pltpu.VMEM((sub, t2.tile_n // 2), jnp.uint32)
            if pack_out
            else pltpu.VMEM((sub, t2.tile_n), cfgs2.out_dtype)
        ),  # partial2
        pltpu.VMEM((t2.tile_m, t2.tile_n), cfgs2.acc_dtype),  # acc2
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
            gm_id_to_sub_bounds=pltpu.SMEM((1,), jnp.int32),
        ),
    ]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(fused_mlp_kernel_main, cfgs1=cfgs1, cfgs2=cfgs2),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                hbm,
                WeightsRef(weight=hbm, scale=hbm if w1_scale is not None else None, bias=None),
                WeightsRef(weight=hbm, scale=hbm if w2_scale is not None else None, bias=None),
            ],
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True),
        name="gmm_mlp_v2-" + get_scope_name(cfgs1) + "+" + get_scope_name(cfgs2),
    )(
        group_sizes,
        group_offset,
        lhs,
        WeightsRef(weight=w1, scale=w1_scale, bias=None),
        WeightsRef(weight=w2, scale=w2_scale, bias=None),
    )
    return out[:size_m, : cfgs2.out_size_n // 2 if pack_out else cfgs2.out_size_n]


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_dtype = lhs_cfgs.dtype
    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype

    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
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

    def _gmm_vmem_estimate(tm: int, tn: int, tk: int) -> int:
        # 1. LHS tile (double-buffered HBM load)
        lhs_tile_bytes = lhs_bits // 8
        lhs_vmem = 2 * tm * tk * lhs_tile_bytes
        # If LHS is quantized on-the-fly, we need an extra single-buffered cast
        # buffer in VMEM.
        if lhs_cfgs.should_quantize:
            lhs_quant_bits = jax.dtypes.itemsize_bits(
                cast(jnp.dtype, lhs_cfgs.quant_dtype)
            )
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
    window = "" if cfgs.n_window is None else f"-nw_{cfgs.n_window[0]}"
    window += "-hq" if cfgs.hoist_lhs_quant else ""
    window += f"-sc{cfgs.trace_scopes}" if cfgs.trace_scopes else ""
    window += f"-rs{cfgs.rhs_split}" if cfgs.rhs_split > 1 else ""
    window += f"-ev{cfgs.experts_per_visit}" if cfgs.experts_per_visit > 1 else ""
    window += "-po" if cfgs.pack_out else ""
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}{window}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
        f"-bb_{tiles.bucket_base}"
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
    n_window: tuple[int, int] | None = None,
    hoist_lhs_quant: bool = False,
    trace_scopes: int = 0,
    rhs_split: int = 1,
    experts_per_visit: int = 1,
    pack_out: bool = False,
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

    if n_window is not None:
        start, width = n_window
        num_lanes = pltpu.get_tpu_info().num_lanes
        if fuse_act is not None:
            raise ValueError("n_window does not combine with fuse_act")
        if not (
            0 <= start
            and 0 < width
            and start + width <= dims.size_n
            and start % num_lanes == 0
            and width % num_lanes == 0
        ):
            raise ValueError(f"{n_window=} must be a lane-aligned part of {dims.size_n=}")
        dims = dataclasses.replace(dims, size_n=width)

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

    if rhs_split > 1:
        # Parts must hold whole mxu-wide column chunks: the inner loop reads one
        # chunk at a time from one part.
        if not (
            fuse_act is None
            and tiles.tile_n == dims.size_n
            and dims.size_n % rhs_split == 0
            and (dims.size_n // rhs_split) % pltpu.get_tpu_info().mxu_column_size == 0
            and dims.size_k % tiles.tile_k == 0
            and not rhs_cfgs.should_dequantize_before_matmul
            and get_packing_factor(rhs_cfgs.dtype, rhs_cfgs.quant_dtype) == 1
        ):
            raise ValueError(
                f"{rhs_split=} needs a non-fused GMM with one lane-aligned N tile "
                f"({dims=}, {tiles=})"
            )

    if experts_per_visit > 1:
        check_experts_per_visit(
            experts_per_visit,
            size_group=dims.size_group,
            size_lhs_group=dims.size_lhs_group,
            size_k=dims.size_k,
            tile_k=tiles.tile_k,
            rhs_split=rhs_split,
        )

    if pack_out and not (
        fuse_act is None
        and n_window is None
        and rhs_split == 1
        and not zero_initialize
        and jnp.dtype(out_dtype) == jnp.bfloat16
        and tiles.tile_n == dims.size_n
        and (dims.size_n // 2) % pltpu.get_tpu_info().num_lanes == 0
    ):
        raise ValueError(
            f"pack_out needs a bf16, non-fused, unwindowed GMM with one lane-aligned "
            f"N tile and no zero init ({dims=}, {tiles=}, {out_dtype=})"
        )

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.dtype(out_dtype),
        acc_dtype=jnp.dtype(acc_dtype),
        zero_init=zero_initialize,
        fuse_act=fuse_act,
        n_window=n_window,
        hoist_lhs_quant=hoist_lhs_quant,
        trace_scopes=trace_scopes,
        rhs_split=rhs_split,
        experts_per_visit=experts_per_visit,
        pack_out=pack_out,
    )


def sub_bounds_size(cfgs: GmmConfigs) -> int:
    """SMEM words for gm_id_to_sub_bounds: E + 1 bounds per m tile, with at
    most (size_group / E) + cdiv(size_m, tile_m) - 1 tiles (1 when unused)."""
    e = cfgs.experts_per_visit
    if e == 1:
        return 1
    max_tiles = cfgs.dims.size_group // e + pl.cdiv(cfgs.dims.size_m, cfgs.tiles.tile_m) - 1
    return max_tiles * (e + 1)


def check_experts_per_visit(
    e: int, *, size_group: int, size_lhs_group: int, size_k: int, tile_k: int,
    rhs_split: int,
) -> None:
    """Multi-expert visits need whole blocks of E experts covering every
    group (no expert-parallel offset), one k step, and no column split."""
    if not (
        e in (2, 4)
        and size_group % e == 0
        and size_lhs_group == size_group
        and tile_k >= size_k
        and rhs_split == 1
    ):
        raise ValueError(
            f"experts_per_visit={e} needs E in (2, 4), E | size_group, all "
            f"groups local, a single k tile and rhs_split 1 ({size_group=}, "
            f"{size_lhs_group=}, {size_k=}, {tile_k=}, {rhs_split=})"
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
        "n_window",
        "hoist_lhs_quant",
        "trace_scopes",
        "rhs_split",
        "experts_per_visit",
        "pack_out",
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
    n_window: tuple[int, int] | None = None,
    hoist_lhs_quant: bool = False,
    trace_scopes: int = 0,
    rhs_split: int = 1,
    experts_per_visit: int = 1,
    pack_out: bool = False,
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
        n_window: Optional (start, width): compute only output columns
            [start, start + width), reading only those rhs columns (lane
            aligned; not with fuse_act). Each column is computed exactly as
            without the window.
        hoist_lhs_quant: Quantize each lhs k-block once per tile before the n
            loop instead of once per mxu-wide column chunk. Same values.
        trace_scopes: >= 2 adds phase trace scopes (gmm_quant, gmm_mxu,
            gmm_epilogue) for in-kernel profiling with libtpu's
            --xla_enable_custom_call_region_trace=true. Profiling only.
        rhs_split: Deliver each weight tile as this many column parts, each
            its own pipeline input (a concurrent DMA). Non-fused GMM with one
            N tile; every output column is computed as without the split.
        experts_per_visit: Visit E consecutive experts per pipeline step (one
            E-expert weight block, one pass over their rows), amortizing the
            fixed per-step cost. All groups local (no group offset), one k
            tile. Every row is computed exactly as with one expert per step.
        pack_out: Return uint32 [size_m, size_n / 2] holding the bf16 output
            column pairs (j, j + size_n / 2) as one word (pack_halves), so each
            row is whole 32-bit words. Same values (unpack_halves recovers them).

    Returns:
        Output of shape [size_m, size_n] ([size_m, width] with n_window).
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
        n_window=n_window,
        hoist_lhs_quant=hoist_lhs_quant,
        trace_scopes=trace_scopes,
        rhs_split=rhs_split,
        experts_per_visit=experts_per_visit,
        pack_out=pack_out,
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
        (
            pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n // 2), jnp.uint32)
            if cfgs.pack_out
            else pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype)
        ),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
            gm_id_to_sub_bounds=pltpu.SMEM((sub_bounds_size(cfgs),), jnp.int32),
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
    if cfgs.pack_out:
        out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n // 2), jnp.uint32)
    else:
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
    )(group_sizes, group_offset, lhs, rhs_weights)[
        :size_m, : cfgs.out_size_n // 2 if cfgs.pack_out else cfgs.out_size_n
    ]
