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

"""Vectorized VMEM load/store and state ring-buffer helpers for GDN v3."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, memory_ref


def load_strided_heads(
    vmem_ref: jax.Ref, num_heads: int, head_dim: int, lane_offset: int = 0
) -> jax.Array:
    """Use strided LDST to split heads along the last dim and transpose.

    Args:
        vmem_ref: One slot's VMEM reference of shape [chunk_size, 1, cols].
        num_heads: Number of heads to split out.
        head_dim: Width of one head, in elements.
        lane_offset: Lane the first head starts at, for a region that is part
            of a larger tensor (k and v within the fused qkv).

    Returns:
        [num_heads, chunk_size, head_dim]
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = vmem_ref.shape[-1] // num_lanes
    lanes_per_head = head_dim // num_lanes

    flat_ref = vmem_ref.reshape(-1, num_lanes)
    head_list = []
    for head in range(num_heads):
        head_lanes = [
            flat_ref[lane_offset + head * lanes_per_head + lane :: lanes_per_col]
            for lane in range(lanes_per_head)
        ]
        head_list.append(jnp.concat(head_lanes, axis=-1))
    return jnp.stack(head_list, axis=0)


def load_compact_heads(
    vmem_ref: jax.Ref, num_heads: int, head_dim: int, dim_offset: int = 0
) -> jax.Array:
    """Use contiguous slices to split heads along the last dim and stack.

    Args:
        vmem_ref: VMEM reference of shape [seq_tile_size, chunk_size, 1, cols].
        num_heads: Number of heads to split out.
        head_dim: Width of one head, in elements.
        dim_offset: Element the first head starts at, for a region that is
            part of a larger tensor (k and v within the fused qkv).

    Returns:
        [seq_tile_size, num_heads, chunk_size, 1, head_dim]
    """
    head_list = []
    for head in range(num_heads):
        start = dim_offset + head * head_dim
        head_list.append(vmem_ref[..., start : start + head_dim])
    return jnp.stack(head_list, axis=1)


def load_compact_to_large(vmem_ref: jax.Ref) -> jax.Array:
    """Use strided load to convert compact to large layout without transpose."""

    # NOTE: Only support 32-bits for now.
    assert vmem_ref.dtype.itemsize == 4
    assert vmem_ref.shape[-2] == 1
    col_size = vmem_ref.shape[-1]
    new_shape = vmem_ref.shape[:-2] + (col_size,)
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes

    vreg_list = []
    vmem_ref = vmem_ref.reshape(-1, col_size)
    for col_start in range(0, col_size, num_lanes):
        col_end = min(col_start + num_lanes, col_size)
        vreg = vmem_ref[..., col_start:col_end]
        vreg_list.append(vreg)
    return jnp.concat(vreg_list, axis=-1).reshape(new_shape)


def _region_rows_per_block(
    slot_ref: jax.Ref, region: config.StateRegion
) -> tuple[int, int]:
    """(typed rows per source block, typed lane count) of a slot tile."""
    out_lanes = slot_ref.shape[-1] // region.lane_split
    block_bytes = math.prod(slot_ref.shape[1:]) * jnp.dtype(slot_ref.dtype).itemsize
    rows_pb = block_bytes // (jnp.dtype(region.view_dtype).itemsize * out_lanes)
    return rows_pb, out_lanes


def load_state_region(
    slot_ref: jax.Ref, region: config.StateRegion, shape: tuple[int, ...]
) -> jax.Array:
    """Loads one slot's state from its raw source-layout tile.

    Args:
        slot_ref: One slot's VMEM tile of shape [region.nblocks,
            region.nrows, *source payload dims, source lanes] in the
            source dtype.
        region: The region's copy-plan (typed view parameters).
        shape: Logical state shape.

    Returns:
        The logical state of ``shape``.
    """
    if region.nblocks == 1:
        rows_per_vreg = 32 // jnp.dtype(region.view_dtype).itemsize
        vreg_rows = -(-region.rows_used // rows_per_vreg) * rows_per_vreg
        arr = typed_ldst.load_typed(
            slot_ref.at[0],
            view_dtype=region.view_dtype,
            lane_split=region.lane_split,
            row0=region.block_row0,
            rows=vreg_rows,
        )
    else:
        assert region.block_row0 == 0, region.block_row0
        parts = [
            typed_ldst.load_typed(
                slot_ref.at[j],
                view_dtype=region.view_dtype,
                lane_split=region.lane_split,
            )
            for j in range(region.nblocks)
        ]
        arr = jnp.concat(parts, axis=0)
    arr = arr[: region.rows_used]
    if region.pack == 2:
        assert region.rows_perm is None
        arr = typed_ldst.unpack_bf16_pairs(arr)
    elif region.pack == 4:
        assert region.rows_perm is None
        arr = typed_ldst.unpack_fp8_quads(arr).astype(jnp.float32)
    elif region.pack == 8:
        assert region.rows_perm is None
        arr = typed_ldst.unpack_fp4_octets(arr).astype(jnp.float32)
    if region.rows_perm is not None:
        # Static row gather from the stored order to the logical order;
        # sublane-dim slices + concat only, no lane crossing.
        arr = jnp.concat([arr[p][None] for p in region.rows_perm], axis=0)
    return arr.reshape(shape)


def store_state_region(
    slot_ref: jax.Ref, region: config.StateRegion, values: jax.Array
) -> None:
    """Stores one slot's logical state into its raw source-layout tile.

    Typed rows past ``region.rows_used`` are zeroed so the tile's whole
    region has deterministic bytes when copied out.
    """
    rows_pb, out_lanes = _region_rows_per_block(slot_ref, region)
    if region.pack == 2:
        assert region.rows_perm is None
        arr = typed_ldst.pack_bf16_pairs(
            values.astype(jnp.bfloat16).reshape(-1, out_lanes)
        )
    elif region.pack == 4:
        assert region.rows_perm is None
        arr = typed_ldst.pack_fp8_quads(
            values.reshape(-1, out_lanes).astype(jnp.float8_e4m3fn)
        )
    elif region.pack == 8:
        assert region.rows_perm is None
        arr = typed_ldst.pack_fp4_octets(
            values.reshape(-1, out_lanes).astype(jnp.float4_e2m1fn)
        )
    else:
        arr = values.astype(region.view_dtype).reshape(-1, out_lanes)
    if region.rows_perm is not None:
        # Inverse of the load-side gather: logical row i is stored at
        # typed row rows_perm[i].
        inverse = [0] * len(region.rows_perm)
        for logical, stored in enumerate(region.rows_perm):
            inverse[stored] = logical
        arr = jnp.concat([arr[i][None] for i in inverse], axis=0)
    if region.nblocks == 1:
        rows_per_vreg = 32 // jnp.dtype(region.view_dtype).itemsize
        vreg_rows = -(-region.rows_used // rows_per_vreg) * rows_per_vreg
        if region.rows_used < vreg_rows:
            arr = jnp.pad(arr, ((0, vreg_rows - region.rows_used), (0, 0)))
        typed_ldst.store_typed(
            slot_ref.at[0],
            arr,
            lane_split=region.lane_split,
            row0=region.block_row0,
            rows=vreg_rows,
        )
    else:
        assert region.block_row0 == 0, region.block_row0
        capacity = region.nblocks * rows_pb
        if region.rows_used < capacity:
            arr = jnp.pad(arr, ((0, capacity - region.rows_used), (0, 0)))
        for j in range(region.nblocks):
            typed_ldst.store_typed(
                slot_ref.at[j],
                arr[j * rows_pb : (j + 1) * rows_pb],
                lane_split=region.lane_split,
            )


def head_rows_supported(cfg: config.GDNConfig) -> bool:
    """Whether load_recurrent_heads / store_recurrent_heads apply to the plan:
    an unscaled, unpermuted bf16 or f32 recurrent region whose typed rows
    fill its blocks exactly."""
    plan = cfg.state_plan
    if plan is None or plan.scale is not None:
        return False
    region = plan.recurrent
    if region.rows_perm is not None or region.pack not in (1, 2):
        return False
    return region.rows_used % cfg.num_v_heads == 0


def _recurrent_head_pieces(
    slot_ref: jax.Ref, cfg: config.GDNConfig, h0: int, nh: int
) -> list[tuple[int, int, int]]:
    """(block, row0, rows) typed-row pieces holding value heads [h0, h0 + nh)
    of one checkpoint's recurrent tile."""
    region = cfg.state_plan.recurrent
    rows_pb, _ = _region_rows_per_block(slot_ref, region)
    assert region.rows_used == region.nblocks * rows_pb, (
        region.rows_used,
        region.nblocks,
        rows_pb,
    )
    rows_per_head = region.rows_used // cfg.num_v_heads
    r, r1 = h0 * rows_per_head, (h0 + nh) * rows_per_head
    pieces = []
    while r < r1:
        j, off = divmod(r, rows_pb)
        n = min(r1, (j + 1) * rows_pb) - r
        pieces.append((j, off + region.block_row0, n))
        r += n
    return pieces


def load_recurrent_heads(
    slot_ref: jax.Ref,
    cfg: config.GDNConfig,
    h0: int,
    nh: int,
    has_initial_state: jax.Array,
) -> jax.Array:
    """Decodes value heads [h0, h0 + nh) of one slot's recurrent state tile
    to f32, zero for a slot without an initial state.

    The select runs on the packed source words, before the widening, so it
    touches half the vregs the f32 state takes.
    """
    region = cfg.state_plan.recurrent
    parts = [
        typed_ldst.load_typed(
            slot_ref.at[j],
            view_dtype=region.view_dtype,
            lane_split=region.lane_split,
            row0=off,
            rows=n,
        )
        for j, off, n in _recurrent_head_pieces(slot_ref, cfg, h0, nh)
    ]
    arr = parts[0] if len(parts) == 1 else jnp.concat(parts, axis=0)
    arr = jnp.where(has_initial_state, arr, jnp.zeros_like(arr))
    if region.pack == 2:
        arr = typed_ldst.unpack_bf16_pairs(arr)
    return arr.astype(jnp.float32).reshape(nh, cfg.kq_head_dim, cfg.v_head_dim)


def store_recurrent_heads(
    slot_ref: jax.Ref, cfg: config.GDNConfig, h0: int, nh: int, values: jax.Array
) -> None:
    """Encodes value heads [h0, h0 + nh) of a checkpoint into one slot's
    recurrent state tile, the rows load_recurrent_heads reads back."""
    region = cfg.state_plan.recurrent
    _, out_lanes = _region_rows_per_block(slot_ref, region)
    if region.pack == 2:
        arr = typed_ldst.pack_bf16_pairs(
            values.astype(jnp.bfloat16).reshape(-1, out_lanes)
        )
    else:
        arr = values.astype(region.view_dtype).reshape(-1, out_lanes)
    r = 0
    for j, off, n in _recurrent_head_pieces(slot_ref, cfg, h0, nh):
        typed_ldst.store_typed(
            slot_ref.at[j],
            arr[r : r + n],
            lane_split=region.lane_split,
            row0=off,
            rows=n,
        )
        r += n


def _load_conv_state(slot_ref: jax.Ref, cfg: config.GDNConfig, idx: int) -> jax.Array:
    """One slot's conv state, decoded through the plan when there is one."""
    # NOTE: Conv1D mandates fp32 due to its usage of compact layout.
    if cfg.state_plan is None:
        return slot_ref[idx, 0].astype(jnp.float32)
    return load_state_region(
        slot_ref.at[idx, 0],
        cfg.state_plan.conv,
        (cfg.prev_kernel_size, 1, cfg.dim_size),
    ).astype(jnp.float32)


def _load_recurrent_state(
    recurrent_slot_ref: jax.Ref,
    conv_state_slot_ref: jax.Ref,
    cfg: config.GDNConfig,
    idx: int,
) -> jax.Array:
    """One slot's recurrent state, converted to FP32 for compute."""
    if cfg.state_plan is None:
        return recurrent_slot_ref[idx, 0].astype(jnp.float32)
    state = load_state_region(
        recurrent_slot_ref.at[idx, 0],
        cfg.state_plan.recurrent,
        (cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim),
    ).astype(jnp.float32)
    if cfg.state_plan.scale is not None:
        vreg_rows = cfg.state_plan.scale.rows_used
        # The scale rides in the conv state's input tile, at position 0 like
        # the recurrent state.
        scale_2d = load_state_region(
            conv_state_slot_ref.at[idx, 0],
            cfg.state_plan.scale,
            (1, vreg_rows, cfg.v_head_dim),
        ).astype(jnp.float32)
        scale_3d = compute_gdn.fused_transpose_broadcast(
            scale_2d, src_dim=1, dst_dim=0
        )[: cfg.num_v_heads]
        state = state * scale_3d
    return state


def store_recurrent_state(
    recurrent_slot_ref: jax.Ref,
    conv_state_slot_ref: jax.Ref,
    cfg: config.GDNConfig,
    state: jax.Array,
) -> None:
    """Stores one slot's recurrent state (and scale when quantized)."""
    assert cfg.state_plan is not None
    if cfg.state_plan.scale is None:
        store_state_region(
            recurrent_slot_ref,
            cfg.state_plan.recurrent,
            state,
        )
        return
    qmax = 448.0 if cfg.state_plan.recurrent.pack == 4 else 6.0
    amax = jnp.max(jnp.abs(state), axis=1, keepdims=True)
    scale_3d = jnp.maximum(amax / qmax, jnp.finfo(jnp.float32).tiny)
    inv_scale = 1.0 / scale_3d
    scaled_2d = (state * inv_scale).reshape(-1, cfg.v_head_dim)
    store_state_region(
        recurrent_slot_ref,
        cfg.state_plan.recurrent,
        scaled_2d,
    )
    vreg_rows = cfg.state_plan.scale.rows_used
    if cfg.num_v_heads < vreg_rows:
        scale_3d_padded = jnp.pad(
            scale_3d, ((0, vreg_rows - cfg.num_v_heads), (0, 0), (0, 0))
        )
    else:
        scale_3d_padded = scale_3d
    scale_2d = compute_gdn.fused_transpose_broadcast(
        scale_3d_padded, src_dim=0, dst_dim=1
    )[0]
    store_state_region(
        conv_state_slot_ref,
        cfg.state_plan.scale,
        scale_2d,
    )


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_in_slot_ref: jax.Ref,
    recurrent_in_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
    stack_recurrent: bool = True,
    load_recurrent: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array | tuple[jax.Array, ...] | None]:
    """Load correct states from HBM or prior tile, and masks invalid states.

    Reference metadata to select the appropriate prior states. If `is_first_tile`
    is True, it selects states read from HBM. If it is False, it selects
    carry states from previous tile. If `has_initial_state` is False, states are
    zero initialized.

    When `cfg.state_plan` is set, the state refs hold raw source-layout
    tiles (see `memory_ref.ExternalStateInBufferedRef`) and are decoded
    through the plan's typed region views instead of read directly.

    Args:
        metadata_ref: Metadata reference containing grid and sequence mappings.
        p_id: Current Pallas program ID.
        conv_state_in_slot_ref: Convolution state read from HBM of shape
            [seq_tile_size, 1, prev_kernel_size, 1, dim_size].
        recurrent_in_slot_ref: Recurrent state read from HBM of shape
            [seq_tile_size, 1, num_v_heads, kq_head_dim, v_head_dim].
        carry_conv_scratch_ref: Optional inter-tile convolution carry of shape
            [seq_tile_size, prev_kernel_size, 1, dim_size].
        carry_recurrent_scratch_ref: Optional inter-tile recurrent state carry of
            shape [seq_tile_size, num_v_heads, kq_head_dim, v_head_dim].
        cfg: GDN configuration object.

    Returns:
        real_sizes: Valid token count per sequence tile of shape [seq_tile_size].
        prev_conv_state: Selected convolution state of shape [seq_tile_size,
        prev_kernel_size, 1, dim_size] in float32.
        prev_recurrent_state: Selected recurrent state of shape [seq_tile_size,
        num_v_heads, kq_head_dim, v_head_dim], or with `stack_recurrent`
        False the per-slot states as a tuple, so the tile never materializes
        the stacked copy; None with `load_recurrent` False, when the caller
        decodes the recurrent state itself (pooled tiles only).
    """

    real_sizes_list = []
    prev_conv_state_list = []
    prev_recurrent_state_list = []

    for idx in range(cfg.seq_tile_size):
        s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
        real_sizes = metadata_ref.p_id_to_r_size[p_id, idx]
        is_first_tile = metadata_ref.p_id_is_first_tile[p_id, idx]
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

        # NOTE: The input VMEM window holds one initial state per sequence at
        # position 0.
        # Both scratches are allocated together (see GDNConfig.scratch_shapes).
        # Only the pool decode is worth guarding; a dense slot read is cheap
        # enough that the guard costs more than it saves.
        if carry_conv_scratch_ref is None or cfg.state_plan is None:
            prev_conv_state = jnp.where(
                has_initial_state, _load_conv_state(conv_state_in_slot_ref, cfg, idx), 0
            )
            if not load_recurrent:
                assert carry_conv_scratch_ref is None and cfg.state_plan is not None
                real_sizes_list.append(real_sizes)
                prev_conv_state_list.append(prev_conv_state)
                continue
            prev_recurrent_state = jnp.where(
                has_initial_state,
                _load_recurrent_state(
                    recurrent_in_slot_ref, conv_state_in_slot_ref, cfg, idx
                ),
                0,
            )
            if carry_conv_scratch_ref is not None:
                assert carry_recurrent_scratch_ref is not None
                prev_conv_state = jnp.where(
                    is_first_tile, prev_conv_state, carry_conv_scratch_ref[idx]
                )
                prev_recurrent_state = jnp.where(
                    is_first_tile,
                    prev_recurrent_state,
                    carry_recurrent_scratch_ref[idx],
                )
        else:
            # Later tiles resume from the carry, so decode both sources once
            # instead of on every tile. One guard, not one per state.
            # pl.when traces the body here, so `idx` is this iteration's.
            assert carry_recurrent_scratch_ref is not None

            @pl.when(is_first_tile)
            def _():
                @pl.when(has_initial_state)
                def _():
                    carry_conv_scratch_ref[idx] = _load_conv_state(
                        conv_state_in_slot_ref, cfg, idx
                    )
                    carry_recurrent_scratch_ref[idx] = _load_recurrent_state(
                        recurrent_in_slot_ref, conv_state_in_slot_ref, cfg, idx
                    )

                @pl.when(jnp.logical_not(has_initial_state))
                def _():
                    carry_conv_scratch_ref[idx] = jnp.zeros_like(
                        carry_conv_scratch_ref[idx]
                    )
                    carry_recurrent_scratch_ref[idx] = jnp.zeros_like(
                        carry_recurrent_scratch_ref[idx]
                    )

            prev_conv_state = carry_conv_scratch_ref[idx]
            prev_recurrent_state = carry_recurrent_scratch_ref[idx]

        real_sizes_list.append(real_sizes)
        prev_conv_state_list.append(prev_conv_state)
        prev_recurrent_state_list.append(prev_recurrent_state)

    real_sizes = jnp.stack(real_sizes_list, axis=0)
    prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
    if not load_recurrent:
        return real_sizes, prev_conv_state, None
    if not stack_recurrent:
        return real_sizes, prev_conv_state, tuple(prev_recurrent_state_list)
    prev_recurrent_state = jnp.stack(prev_recurrent_state_list, axis=0)

    return real_sizes, prev_conv_state, prev_recurrent_state


def load_activation_as_compact(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Load activations from VMEM as a compact layout."""

    k_offset = cfgs.num_kq_heads * cfgs.kq_head_dim
    v_offset = 2 * k_offset
    num_lanes = pltpu.get_tpu_info().num_lanes

    if cfgs.kq_head_dim % num_lanes == 0 and cfgs.v_head_dim % num_lanes == 0:
        q_list = [
            qkv_vreg[..., h * cfgs.kq_head_dim : (h + 1) * cfgs.kq_head_dim]
            for h in range(cfgs.num_kq_heads)
        ]
        k_list = [
            qkv_vreg[
                ...,
                k_offset + h * cfgs.kq_head_dim : k_offset + (h + 1) * cfgs.kq_head_dim,
            ]
            for h in range(cfgs.num_kq_heads)
        ]
        v_list = [
            qkv_vreg[
                ...,
                v_offset + h * cfgs.v_head_dim : v_offset + (h + 1) * cfgs.v_head_dim,
            ]
            for h in range(cfgs.num_v_heads)
        ]
        q_compact = jnp.stack(q_list, axis=1)
        k_compact = jnp.stack(k_list, axis=1)
        v_compact = jnp.stack(v_list, axis=1)
    else:
        qkv_vmem_ref[...] = qkv_vreg
        q_compact = load_compact_heads(
            qkv_vmem_ref, cfgs.num_kq_heads, cfgs.kq_head_dim
        )
        k_compact = load_compact_heads(
            qkv_vmem_ref, cfgs.num_kq_heads, cfgs.kq_head_dim, k_offset
        )
        v_compact = load_compact_heads(
            qkv_vmem_ref, cfgs.num_v_heads, cfgs.v_head_dim, v_offset
        )
    b_compact = jnp.expand_dims(b_vmem_ref[...], axis=1)
    if cfgs.is_kda:
        # KDA's gate is per-channel, so it splits into heads like q/k/v do
        # instead of occupying a single lane per head.
        a_compact = load_compact_heads(a_vmem_ref, cfgs.num_v_heads, cfgs.kq_head_dim)
    else:
        a_compact = jnp.expand_dims(a_vmem_ref[...], axis=1)
    return q_compact, k_compact, v_compact, b_compact, a_compact


def load_activation_as_large(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Load activations from VMEM as a large layout."""

    qkv_vmem_ref[...] = qkv_vreg

    num_lanes = pltpu.get_tpu_info().num_lanes
    kq_lanes = cfgs.num_kq_heads * (cfgs.kq_head_dim // num_lanes)

    q_large_list = []
    k_large_list = []
    v_large_list = []
    a_large_list = []
    for idx in range(cfgs.seq_tile_size):
        qkv_slot = qkv_vmem_ref.at[idx]
        q_large_list.append(
            load_strided_heads(qkv_slot, cfgs.num_kq_heads, cfgs.kq_head_dim)
        )
        k_large_list.append(
            load_strided_heads(qkv_slot, cfgs.num_kq_heads, cfgs.kq_head_dim, kq_lanes)
        )
        v_large_list.append(
            load_strided_heads(
                qkv_slot, cfgs.num_v_heads, cfgs.v_head_dim, 2 * kq_lanes
            )
        )
        if cfgs.is_kda:
            # Per-channel gate: split into heads and transpose like q/k/v.
            a_large_list.append(
                load_strided_heads(
                    a_vmem_ref.at[idx], cfgs.num_v_heads, cfgs.kq_head_dim
                )
            )

    q_large = jnp.stack(q_large_list, axis=0)
    k_large = jnp.stack(k_large_list, axis=0)
    v_large = jnp.stack(v_large_list, axis=0)
    b_large = jnp.expand_dims(load_compact_to_large(b_vmem_ref), axis=1)
    if cfgs.is_kda:
        a_large = jnp.stack(a_large_list, axis=0)
    else:
        a_large = jnp.expand_dims(load_compact_to_large(a_vmem_ref), axis=1)

    return q_large, k_large, v_large, b_large, a_large
