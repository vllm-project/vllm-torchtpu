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
"""Shared declared-VMEM accounting helpers for GDN and KDA kernel tests."""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref, tiling

_RANK_2D = 2


def get_vmem_declared_bytes(
    cfg: config.GDNConfig,
    conv_state_dim_size: int,
    num_lanes: int,
    num_sublanes: int,
) -> int:
    """Return declared on-chip VMEM bytes from tiling.get_declared_vmem_bytes."""
    spec = tiling.VmemTileSpec.from_config(
        cfg=cfg,
        conv_state_dim_size=conv_state_dim_size,
        num_lanes=num_lanes,
        num_sublanes=num_sublanes,
    )
    return tiling.get_declared_vmem_bytes(spec)


def weight_refs_vmem_bytes(
    cfg: config.GDNConfig,
    num_lanes: int,
    num_sublanes: int,
) -> int:
    """Compute VMEM weight footprint from WeightRefs shapes with 2D tiling."""
    f32 = jnp.dtype(jnp.float32)
    num_v_heads = cfg.num_v_heads
    kq_head_dim = cfg.kq_head_dim
    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(
            weight=jax.ShapeDtypeStruct((cfg.kernel_size, 1, cfg.dim_size), f32),
            bias=jax.ShapeDtypeStruct((1, cfg.dim_size), f32),
        ),
        gdn=memory_ref.GDNWeightsRef(
            a_log=jax.ShapeDtypeStruct((num_v_heads,), f32),
            dt_bias=jax.ShapeDtypeStruct(
                (num_v_heads, kq_head_dim) if cfg.is_kda else (num_v_heads,), f32
            ),
        ),
    )
    conv_bytes = sum(
        leaf.shape[0]
        * tiling.align_to(int(np.prod(leaf.shape[1:])), num_lanes)
        * jnp.dtype(leaf.dtype).itemsize
        for leaf in jax.tree.leaves(weights.conv)
    )
    gdn_bytes = sum(
        (
            tiling.align_to(leaf.shape[0], num_sublanes)
            * tiling.align_to(leaf.shape[1], num_lanes)
            if len(leaf.shape) == _RANK_2D
            else tiling.align_to(leaf.shape[0], num_lanes)
        )
        * jnp.dtype(leaf.dtype).itemsize
        for leaf in jax.tree.leaves(weights.gdn)
    )
    return conv_bytes + gdn_bytes
