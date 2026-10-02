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

# Forked from vllm_torchtpu/kernels/gdn/v3/wrapper.py (inner_kernel only).
# Upstream V3 changes ported by hand through #1422.
# This copy is owned by the PCP workstream and must not import from
# kernels/gdn/v3. Upstream V3 changes are not picked up automatically;
# see README.md in this directory.


import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.gdn_pcp_fused import (
    compute_conv1d,
    compute_gdn,
    config,
    memory_ref,
    vmem_ldst,
)


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Array,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Array,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Array,  # [seq, chunk, 1, num_v_heads]
    conv_state_slot_ref: jax.Array,  # [seq, prev_kernel_size, 1, dim_size]
    recurrent_slot_ref: jax.Array,  # [seq, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Array,  # [seq * chunk, num_v_heads, v_head]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array | None = None,
):
    """Orchestrates computation of Conv1D and GDN for a single tile.

    This kernel acts as a facade adhering to strict separation of concerns. It
    operates VMEM reference without knowledge on DMA logic. Furthermore, the
    kernel invokes vmem_ldst to pre-processes data needed for compute and
    invokes compute_conv1d and compute_gdn for actual compute.

    Args:
        qkv_slot_ref: qkv VMEM ref that stores data loaded from HBM.
        b_slot_ref: b VMEM ref that stores data loaded from HBM.
        a_slot_ref: a VMEM ref that stores data loaded from HBM.
        conv_state_slot_ref: Convolution state VMEM ref that stores data loaded from
            HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
        recurrent_slot_ref: Recurrent state VMEM ref that stores data loaded from
            HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
        out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
        metadata_ref: Metadata reference containing grid and sequence mappings.
        weights_ref: Weight references for Conv1D and GDN in VMEM.
        carry_conv_scratch_ref: Optional VMEM scratch reference for inter-tile
            convolution carry.
        carry_recurrent_scratch_ref: Optional VMEM scratch reference for inter-tile
            recurrent state carry.
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

    # Prepare states.
    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_slot_ref=conv_state_slot_ref,
        recurrent_slot_ref=recurrent_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
    )

    # Step 1: Conv1D.
    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding.
    # Therefore, it is extremely important to leverage compact layout that
    # ensures 1 register only stores data from 1 row.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

    # Prepare conv1d weights.
    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    conv_bias = None
    if weights_ref.conv.bias is not None:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

    qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        lhs=qkv_in_compact,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        cfg=cfg,
    )

    conv_state_slot_ref[...] = new_conv_state
    if carry_conv_scratch_ref is not None:
        # The next tile resumes from the state after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN.

    # Prepare gdn weights.
    padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
    a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

    # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
    # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
    # Passing refs into gdn.py breaks strict separation of concerns.
    if cfg.use_recurrent:
        recurrent_fn = compute_gdn.recurrent_gdn
        q_compact, k_compact, v_compact, b_compact, a_compact = (
            vmem_ldst.load_activation_as_compact(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfg,
            )
        )

        out, new_recurrent_state = recurrent_fn(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )

    else:
        chunked_fn = compute_gdn.chunked_gdn
        q_large, k_large, v_large, b_large, a_large = (
            vmem_ldst.load_activation_as_large(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfg,
            )
        )

        out, new_recurrent_state = chunked_fn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )

    # Store output and recurrent to vmem.
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    # Round to the state dtype, then pack into the uint32 window.
    recurrent_slot_ref[...] = pltpu.bitcast(
        new_recurrent_state.astype(cfg.dtypes.recurrent_state), jnp.uint32
    )

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]
