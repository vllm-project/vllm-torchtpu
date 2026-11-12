# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused prefill MoE's uncapped overflow consumer.

The compact metadata path keeps a launch's local routes in SMEM. When an EP
group's routes exceed that capacity, every rank runs this GU/down pipeline over
the same prepared expert-major metadata instead, streaming bounded metadata
tiles.
"""

import jax.numpy as jnp

from ..prefill_megakernel import expert_family_slot
from .indexed_ragged_gmm import indexed_gate_up
from .moe_down_valu import down_valu, down_valu_compact


def routed_experts_prepared(
    latent,
    sorted_tokens,
    sorted_probs,
    counts,
    expert_gate_up,
    expert_gate_up_scales,
    expert_down,
    expert_down_scales,
    *,
    reduce_tables,
    expert_layer=None,
    global_route_max=None,
):
    """Consume prepared expert-major metadata with the existing GU/down path."""
    if expert_layer is not None:
        expert_layer = expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens = latent.shape[0]
    # The E8M0 exponent bytes go to the GMM as they are (it converts each
    # tile in VMEM): an FP32 [E, blocks, 1, N] copy pads 8x in HBM.
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    # The gate/up weights arrive lane-interleaved (the loader's layout), the
    # order the fused SiTU activation consumes.
    middle = indexed_gate_up(
        latent, sorted_tokens, expert_gate_up, counts, gu_scale, expert_layer
    )
    options = dict(
        tokens=tokens,
        layer=expert_layer,
        ring_dtype=jnp.dtype(jnp.float32),
    )
    if tokens == 8192:
        return down_valu_compact(
            middle,
            expert_down,
            down_scale,
            counts,
            sorted_tokens,
            sorted_probs,
            reduce_tables,
            tile=256,
            columns=896,
            unroll=32,
            metadata_prefix_rows=32768,
            slice_input=False,
            global_route_max=global_route_max,
            **options,
        )
    return down_valu(
        middle,
        expert_down,
        down_scale,
        counts,
        sorted_tokens,
        sorted_probs,
        reduce_tables,
        tile=128,
        columns=1792,
        unroll=8,
        **options,
    )
