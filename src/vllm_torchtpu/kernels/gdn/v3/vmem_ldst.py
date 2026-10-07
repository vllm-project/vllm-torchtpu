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

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, legacy_unified, memory_ref


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


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_slot_ref: jax.Ref,
    recurrent_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Load correct states from HBM or prior tile, and masks invalid states.

    Reference metadata to select the appropriate prior states. If `is_first_tile`
    is True, it selects states read from HBM. If it is False, it selects
    carry states from previous tile. If `has_initial_state` is False, states are
    zero initialized.

    When `cfg.state_plan` is set, the state refs hold raw source-layout
    tiles (see `legacy_unified.ExternalStateBufferedRef`) and are decoded
    through the plan's typed region views instead of read directly.

    Args:
        metadata_ref: Metadata reference containing grid and sequence mappings.
        p_id: Current Pallas program ID.
        conv_state_slot_ref: Convolution state read from HBM of shape
            [seq_tile_size, state_window_alloc, prev_kernel_size, 1, dim_size];
            the incoming state is at window position `cfg.state_in_pos`.
        recurrent_slot_ref: Recurrent state read from HBM of shape [seq_tile_size,
            state_window_alloc, num_v_heads, kq_head_dim // packing, v_head_dim]
            (uint32 words); the incoming state is at `cfg.state_in_pos`.
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

    for idx in range(cfg.seq_tile_size):
        s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
        real_sizes = metadata_ref.p_id_to_r_size[p_id, idx]
        is_first_tile = metadata_ref.p_id_is_first_tile[p_id, idx]
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

        if cfg.state_plan is not None:
            # TODO: Delete this after unified pool migration completes.
            prev_conv_state, prev_recurrent_state = legacy_unified.select_states(
                conv_state_slot_ref,
                recurrent_slot_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
                cfg,
                idx,
                has_initial_state,
                is_first_tile,
            )
        else:
            # NOTE: Conv1D mandates fp32 due to its usage of compact layout.
            hbm_conv_state = conv_state_slot_ref[idx, cfg.state_in_pos].astype(
                jnp.float32
            )
            prev_conv_state = jnp.where(has_initial_state, hbm_conv_state, 0)

            if carry_conv_scratch_ref is not None:
                prev_tile_conv = carry_conv_scratch_ref[idx]
                prev_conv_state = jnp.where(
                    is_first_tile, prev_conv_state, prev_tile_conv
                )

            # Unpack the uint32 words to the state dtype, then upcast because
            # compute runs in fp32.
            hbm_recurrent_state = pltpu.bitcast(
                recurrent_slot_ref[idx, cfg.state_in_pos], cfg.dtypes.recurrent_state
            ).astype(jnp.float32)
            prev_recurrent_state = jnp.where(has_initial_state, hbm_recurrent_state, 0)

            if carry_recurrent_scratch_ref is not None:
                prev_tile_recurrent_scratch = carry_recurrent_scratch_ref[idx]
                prev_recurrent_state = jnp.where(
                    is_first_tile, prev_recurrent_state, prev_tile_recurrent_scratch
                )

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
