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
    parts = [
        typed_ldst.load_typed(
            slot_ref.at[j], view_dtype=region.view_dtype, lane_split=region.lane_split
        )
        for j in range(region.nblocks)
    ]
    arr = parts[0] if region.nblocks == 1 else jnp.concat(parts, axis=0)
    arr = arr[: region.rows_used]
    # When a bfloat16 state uses a uint32 VMEM view (2 bf16 elements per
    # uint32 word), unpack back to bfloat16 before reshaping to `shape`.
    if region.view_dtype == jnp.uint32 and math.prod(shape) == arr.size * 2:
        assert region.rows_perm is None
        arr = pltpu.bitcast(arr, jnp.bfloat16)
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
    if region.view_dtype == jnp.uint32 and values.dtype == jnp.float32:
        assert region.rows_perm is None
        arr = pltpu.bitcast(values.astype(jnp.bfloat16), jnp.uint32)
    else:
        arr = values.astype(region.view_dtype)
    arr = arr.reshape(-1, out_lanes)
    if region.rows_perm is not None:
        # Inverse of the load-side gather: logical row i is stored at
        # typed row rows_perm[i].
        inverse = [0] * len(region.rows_perm)
        for logical, stored in enumerate(region.rows_perm):
            inverse[stored] = logical
        arr = jnp.concat([arr[i][None] for i in inverse], axis=0)
    capacity = region.nblocks * rows_pb
    if region.rows_used < capacity:
        arr = jnp.pad(arr, ((0, capacity - region.rows_used), (0, 0)))
    for j in range(region.nblocks):
        typed_ldst.store_typed(
            slot_ref.at[j],
            arr[j * rows_pb : (j + 1) * rows_pb],
            lane_split=region.lane_split,
        )


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
    slot_ref: jax.Ref, cfg: config.GDNConfig, idx: int
) -> jax.Array:
    """One slot's recurrent state, converted to FP32 for compute."""
    if cfg.state_plan is None:
        return slot_ref[idx, 0].astype(jnp.float32)
    return load_state_region(
        slot_ref.at[idx, 0],
        cfg.state_plan.recurrent,
        (cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim),
    ).astype(jnp.float32)


def load_window_log(
    log_slot_ref: jax.Ref, log_conv_slot_ref: jax.Ref, cfg: config.GDNConfig
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Loads one slot's draft log (LOG mode) from its raw source tiles.

    Args:
        log_slot_ref: The slot's `StateSourcePlan.log` tile.
        log_conv_slot_ref: The slot's `StateSourcePlan.log_conv` tile.
        cfg: GDN configuration object.

    Returns:
        gates [K, 1, 1, d_v] (lane h: decay of value head h after exp),
        keys [K, num_kq_heads, 1, kq_head_dim], updates [K, num_v_heads, 1,
        v_head_dim] and conv inputs [K, 1, dim_size] f32, for draft
        positions 1..K.
    """
    plan = cfg.state_plan
    assert plan is not None and plan.log is not None and plan.log_conv is not None
    num_pos, n_kq, n_v = cfg.log_positions, cfg.num_kq_heads, cfg.num_v_heads
    keys_row, updates_row, rows_used = config.log_layout_rows(num_pos, n_kq, n_v)
    rows = load_state_region(log_slot_ref, plan.log, (rows_used, cfg.v_head_dim))
    gates = rows[:keys_row].reshape(num_pos, 1, 1, cfg.v_head_dim)
    keys = rows[keys_row:updates_row].reshape(num_pos, n_kq, 1, cfg.kq_head_dim)
    updates = rows[updates_row:].reshape(num_pos, n_v, 1, cfg.v_head_dim)
    conv_inputs = load_state_region(
        log_conv_slot_ref, plan.log_conv, (num_pos, 1, cfg.dim_size)
    ).astype(jnp.float32)
    return gates, keys, updates, conv_inputs


def store_window_log(
    log_slot_ref: jax.Ref,
    log_conv_slot_ref: jax.Ref,
    gates: jax.Array,
    keys: jax.Array,
    updates: jax.Array,
    conv_inputs: jax.Array,
    cfg: config.GDNConfig,
) -> None:
    """Stores one slot's draft log; the inverse of `load_window_log`."""
    plan = cfg.state_plan
    assert plan is not None and plan.log is not None and plan.log_conv is not None
    d = cfg.v_head_dim
    rows = jnp.concat(
        [gates.reshape(-1, d), keys.reshape(-1, d), updates.reshape(-1, d)], axis=0
    )
    store_state_region(log_slot_ref, plan.log, rows)
    store_state_region(log_conv_slot_ref, plan.log_conv, conv_inputs)


def fold_window_log(
    recurrent: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    conv: jax.Array,  # [prev_kernel_size, 1, dim_size]
    gates: jax.Array,
    keys: jax.Array,
    updates: jax.Array,
    conv_inputs: jax.Array,
    read_offset: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Applies the first `read_offset` logged draft updates to checkpoint 0.

    Each update is the recurrence's own two statements
    (`compute_gdn.recurrent_gdn_per_seq`), state * g then + k^T * v_new, on
    the operands it logged, and the gate and key go through the same
    lane-to-sublane transposes, so the result is bit for bit the state the
    window wrote for that position in SELECT mode. The conv state shifts in
    the same number of logged conv inputs. Positions past `read_offset` are
    computed and discarded by a select, which keeps the loop static.

    Returns:
        (recurrent, conv) states after window position `read_offset`.
    """
    num_pos = cfg.log_positions
    # [d_v, K, 1, 1] -> [num_v_heads, K, 1, 1], like the gating in recurrent_gdn.
    gating = compute_gdn.fused_transpose_broadcast(
        gates.reshape(1, num_pos, 1, cfg.v_head_dim), src_dim=3, dst_dim=0
    )[: cfg.num_v_heads]
    # [K, num_kq_heads, kq_head_dim, 1], like k_compact_t in recurrent_gdn.
    keys_t = compute_gdn.fused_transpose_broadcast(keys, src_dim=3, dst_dim=2)

    rows = jnp.concat([conv, conv_inputs], axis=0)
    kprev = cfg.prev_kernel_size
    new_conv = conv
    for j in range(num_pos):
        applied = j < read_offset
        gating_curr = gating[:, j]
        k_curr_t = jnp.repeat(keys_t[j], cfg.v_per_kq_head, axis=0)
        v_new = updates[j]
        state_updated = recurrent * gating_curr
        state_new = k_curr_t * v_new
        recurrent = jnp.where(applied, state_updated + state_new, recurrent)
        new_conv = jnp.where(applied, rows[j + 1 : j + 1 + kprev], new_conv)
    return recurrent, new_conv


def fold_conv_log(
    conv: jax.Array,  # [prev_kernel_size, 1, dim_size]
    conv_inputs: jax.Array,  # [K, 1, dim_size]
    read_offset: jax.Array,
    cfg: config.GDNConfig,
) -> jax.Array:
    """The conv half of `fold_window_log`: shifts in read_offset logged inputs."""
    rows = jnp.concat([conv, conv_inputs], axis=0)
    kprev = cfg.prev_kernel_size
    new_conv = conv
    for j in range(cfg.log_positions):
        new_conv = jnp.where(j < read_offset, rows[j + 1 : j + 1 + kprev], new_conv)
    return new_conv


def load_states_factored(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_in_slot_ref: jax.Ref,
    recurrent_in_slot_ref: jax.Ref,
    log_in_slot_refs: tuple[jax.Ref, jax.Ref],
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, list[jax.Array], list[jax.Array]]:
    """Loads a BATCHED tile's states for the factored window body.

    The conv state is folded as in `load_and_select_states`; the recurrent
    state stays checkpoint block 0's (the base), and the log's typed rows
    and the count of its updates that apply go to `compute_gdn.factored_gdn`.
    A sequence without an initial state gets zeros and a count of 0.

    Returns:
        real_sizes [seq_tile_size], conv [seq_tile_size, prev_kernel_size, 1,
        dim_size], base [seq_tile_size, num_v_heads, kq_head_dim,
        v_head_dim], and per slot the log rows [rows_used, v_head_dim] and
        the update count.
    """
    assert cfg.uses_factored_window and cfg.state_plan is not None
    plan = cfg.state_plan
    _, _, rows_used = config.log_layout_rows(
        cfg.log_positions, cfg.num_kq_heads, cfg.num_v_heads
    )
    log_ref, log_conv_ref = log_in_slot_refs
    real_sizes, convs, bases, log_rows, counts = [], [], [], [], []
    for idx in range(cfg.seq_tile_size):
        s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]
        read_offset = metadata_ref.s_idx_to_read_offset[s_idx]
        conv_inputs = load_state_region(
            log_conv_ref.at[idx, 0], plan.log_conv, (cfg.log_positions, 1, cfg.dim_size)
        ).astype(jnp.float32)
        conv = fold_conv_log(
            _load_conv_state(conv_state_in_slot_ref, cfg, idx), conv_inputs, read_offset, cfg
        )
        base = _load_recurrent_state(recurrent_in_slot_ref, cfg, idx)
        real_sizes.append(metadata_ref.p_id_to_r_size[p_id, idx])
        convs.append(jnp.where(has_initial_state, conv, 0))
        bases.append(jnp.where(has_initial_state, base, 0))
        log_rows.append(
            load_state_region(log_ref.at[idx, 0], plan.log, (rows_used, cfg.v_head_dim))
        )
        counts.append(jnp.where(has_initial_state, read_offset, 0))
    return (
        jnp.stack(real_sizes, axis=0),
        jnp.stack(convs, axis=0),
        jnp.stack(bases, axis=0),
        log_rows,
        counts,
    )


def store_window_log_rows(
    log_slot_ref: jax.Ref,
    log_conv_slot_ref: jax.Ref,
    rows: jax.Array,
    conv_inputs: jax.Array,
    cfg: config.GDNConfig,
) -> None:
    """`store_window_log` for log rows already in the typed row order."""
    plan = cfg.state_plan
    assert plan is not None and plan.log is not None and plan.log_conv is not None
    store_state_region(log_slot_ref, plan.log, rows)
    store_state_region(log_conv_slot_ref, plan.log_conv, conv_inputs)


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_in_slot_ref: jax.Ref,
    recurrent_in_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
    log_in_slot_refs: tuple[jax.Ref, jax.Ref] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
        log_in_slot_refs: LOG mode: the (log, log_conv) tiles of checkpoint
            block 0, whose first read_offset updates are applied to the
            loaded states.
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
        num_v_heads, kq_head_dim, v_head_dim].
    """

    real_sizes_list = []
    prev_conv_state_list = []
    prev_recurrent_state_list = []

    assert cfg.is_window_log == (log_in_slot_refs is not None)

    for idx in range(cfg.seq_tile_size):
        s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
        real_sizes = metadata_ref.p_id_to_r_size[p_id, idx]
        is_first_tile = metadata_ref.p_id_is_first_tile[p_id, idx]
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

        def load_states(idx=idx, s_idx=s_idx):
            conv = _load_conv_state(conv_state_in_slot_ref, cfg, idx)
            recurrent = _load_recurrent_state(recurrent_in_slot_ref, cfg, idx)
            if log_in_slot_refs is not None:
                log_ref, log_conv_ref = log_in_slot_refs
                recurrent, conv = fold_window_log(
                    recurrent,
                    conv,
                    *load_window_log(log_ref.at[idx, 0], log_conv_ref.at[idx, 0], cfg),
                    metadata_ref.s_idx_to_read_offset[s_idx],
                    cfg,
                )
            return conv, recurrent

        # NOTE: The input VMEM window holds one initial state per sequence at
        # position 0.
        # Both scratches are allocated together (see GDNConfig.scratch_shapes).
        # Only the pool decode is worth guarding; a dense slot read is cheap
        # enough that the guard costs more than it saves.
        if carry_conv_scratch_ref is None or cfg.state_plan is None:
            loaded_conv, loaded_recurrent = load_states()
            prev_conv_state = jnp.where(has_initial_state, loaded_conv, 0)
            prev_recurrent_state = jnp.where(has_initial_state, loaded_recurrent, 0)
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
                loaded_conv, loaded_recurrent = load_states()
                carry_conv_scratch_ref[idx] = jnp.where(
                    has_initial_state, loaded_conv, 0
                )
                carry_recurrent_scratch_ref[idx] = jnp.where(
                    has_initial_state, loaded_recurrent, 0
                )

            prev_conv_state = carry_conv_scratch_ref[idx]
            prev_recurrent_state = carry_recurrent_scratch_ref[idx]

        real_sizes_list.append(real_sizes)
        prev_conv_state_list.append(prev_conv_state)
        prev_recurrent_state_list.append(prev_recurrent_state)

    real_sizes = jnp.stack(real_sizes_list, axis=0)
    prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
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

    qkv_vmem_ref[...] = qkv_vreg

    k_offset = cfgs.num_kq_heads * cfgs.kq_head_dim
    v_offset = 2 * k_offset

    q_compact = load_compact_heads(qkv_vmem_ref, cfgs.num_kq_heads, cfgs.kq_head_dim)
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
