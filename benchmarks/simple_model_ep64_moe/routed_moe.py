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
"""simple_model benchmark adapter for unmodified PR2821 MoE kernels at 7ab112621.
Only router and TorchTPU flat-input mesh bridging are supplied here.
"""

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.ragged_a2a_moe import ragged_a2a_moe_local


def build_pr_op(mesh, *, topk):
    from torch_tpu._internal.pallas import jax_op

    dp, tp = mesh.shape["dp"], mesh.shape["tp"]
    spec = P("dp", "tp")

    def local(rank, x, router, w1, w2, s1, s2, mask):
        logits = (
            (x.astype(jnp.bfloat16) @ router).astype(jnp.bfloat16).astype(jnp.float32)
        )
        scores = jax.nn.softmax(logits, axis=1)
        cols = jnp.arange(router.shape[1], dtype=jnp.int32)[None, :]
        weights = []
        ids = []
        for _ in range(topk):
            val = jnp.max(scores, axis=1, keepdims=True)
            idx = jnp.min(
                jnp.where(scores == val, cols, router.shape[1]), axis=1, keepdims=True
            )
            weights.append(val)
            ids.append(idx)
            scores = jnp.where(cols == idx, -jnp.inf, scores)
        weights = jnp.concatenate(weights, axis=1)
        weights = (weights / jnp.sum(weights, axis=1, keepdims=True)).astype(
            jnp.bfloat16
        )
        weights = jnp.where(mask != 0, 0, weights)
        return ragged_a2a_moe_local(
            x,
            w1,
            w2,
            s1,
            s2,
            weights,
            jnp.concatenate(ids, axis=1),
            rank * w1.shape[0],
            tp_size=tp,
            dp_size=dp,
            topk=topk,
            activation="silu",
            use_sparse_core=True,
            onehot_moe_permute_threshold=3840,
            skip_padded_tokens=True,
            capacity_factor=4.0,
            route_factor=5.0,
            row_gather=True,
            row_scatter=True,
            streamed_combine=True,
            owner_combine=True,
            sorted_plan=True,
            owner_tail=True,
            routing_dma=True,
            send_index_rows_minor=True,
            plan_kernel=True,
        )

    def wrapped(*args):
        return local(*(a[0, 0] for a in args))[None, None]

    mapped = jax.shard_map(
        wrapped, mesh=mesh, in_specs=(spec,) * 8, out_specs=spec, check_vma=False
    )

    def forward(
        rank: jax.Array,
        x: jax.Array,
        router: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        return mapped(rank, x, router, w1, w2, s1, s2, mask)

    return jax_op(
        "pallas::pr2821_full",
        forward,
        mesh=mesh,
        input_partition_specs=(spec,) * 8,
        output_partition_specs=(spec,),
    )


MODE = "pr"
SCALE_SINGLETON_AXIS = True


def build_routed(dp, tp, topk, scale_block, scale_shapes):
    from torch_tpu._internal.pallas import get_pallas_mesh

    op = build_pr_op(get_pallas_mesh(("dp", "tp"), (dp, tp)), topk=topk)

    @op.register_fake
    def fake(r, x, *operands):
        return torch.empty_like(x)

    def route(*operands):
        return op(*(a[None, None] for a in operands))[0, 0]

    return route
