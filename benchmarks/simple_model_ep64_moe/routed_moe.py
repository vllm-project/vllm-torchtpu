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
"""Benchmark bridge for the fused EP v2 implementation from d8e6ae2."""

from functools import partial

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.pallas import get_pallas_mesh, jax_op

from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2, layer


def build_v2_op(ep, topk, scale_block, scale_shapes):
    # The kernel emits a compile-time layout message. Process scope avoids
    # importing vLLM's serving distributed state just to log that message.
    layer.logger.info_once = partial(layer.logger.info_once, scope="process")
    mesh = get_pallas_mesh(("d",), (ep,))
    groups = tuple(tuple(range(begin, begin + 4)) for begin in range(0, ep, 4))
    expert_second_minor = tuple(
        tuple(tpu_torch_compile.get_default_layout(torch.float32, shape)[0])
        == (2, 0, 1)
        for shape in scale_shapes
    )
    spec = PartitionSpec("d")

    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return fused_ep_moe_v2(
            x,
            w1,
            w2,
            s1,
            s2,
            gating.astype(jnp.float32),
            rank=rank,
            topk=topk,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP4,
            rhs_qb=scale_block,
            sharded_plan=True,
            token_replica_groups=groups,
            w1_scales_expert_second_minor=expert_second_minor[0],
            w2_scales_expert_second_minor=expert_second_minor[1],
        )

    op = jax_op(
        "pallas::simple_model_ut_fused_ep_v2",
        moe,
        mesh=mesh,
        input_partition_specs=(spec,) * 7,
        output_partition_specs=spec,
    )

    @op.register_fake
    def fake(x, *_args):
        return torch.empty_like(x)

    return op


MODE = "v2"
SCALE_SINGLETON_AXIS = False


def build_routed(dp, tp, topk, scale_block, scale_shapes):
    op = build_v2_op(dp * tp, topk, scale_block, scale_shapes)

    def route(r, x, router, w1, w2, s1, s2, mask):
        logits = (x @ router).float().masked_fill(mask.bool(), -float("inf"))
        return op(x, w1, w2, s1, s2, logits, r.reshape(1, 1))

    return route
