# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for PR 2821 routed MoE, router and shared expert."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.kernels.router_topk import router_topk
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    _capacity_overflow,
    exchange_capacity,
    ragged_a2a_moe_local,
)


def build_pr_op(mesh, mode="capacity", iterations=0, diagnostics=False):
    def local(rank, x, router, gate, sw1, ss1, sw2, ss2, w1, w2, s1, s2, mask):
        logits = jnp.dot(x, router, preferred_element_type=jnp.float32).astype(x.dtype)
        scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        values, ids = router_topk(scores, 10)
        values = (values / values.sum(-1, keepdims=True)).astype(x.dtype)
        values = jnp.where(mask != 0, 0, values)
        if diagnostics:
            gathered = lax.all_gather(ids, "dp")
            rows, routes = exchange_capacity(
                dp_size=8,
                tp_size=4,
                num_tokens=x.shape[0],
                topk=10,
                local_num_experts=16,
                factor=2.0,
                onehot_moe_permute_threshold=3840,
                use_sparse_core=True,
            )
            overflow = _capacity_overflow(
                gathered, tp_size=4, local_num_experts=16, rows=rows, routes=routes
            )
            local_hit = (gathered // 16) == rank[0]
            recv = local_hit.any(-1).sum()
            all_routes = local_hit.sum()
            return x, jnp.array(
                [rows, routes, overflow.astype(jnp.int32), recv, all_routes, 0, 0, 0],
                jnp.int32,
            )
        out = ragged_a2a_moe_local(
            x,
            w1,
            w2,
            s1,
            s2,
            values,
            ids,
            rank * 16,
            tp_size=4,
            dp_size=8,
            topk=10,
            activation="silu",
            use_sparse_core=True,
            onehot_moe_permute_threshold=3840,
            skip_padded_tokens=True,
            capacity_factor=2.0 if mode == "capacity" else 0.0,
        )
        # Shared expert is outside PR's routed core, as in serving.
        gu = xla_quantized_matmul(x, sw1, ss1.reshape(-1))
        g, u = jnp.split(gu, 2, axis=-1)
        act = (jax.nn.silu(g.astype(jnp.float32)) * u).astype(x.dtype)
        shared = xla_quantized_matmul(act, sw2, ss2.reshape(-1))
        gate_value = jax.nn.sigmoid(
            jnp.dot(x, gate, preferred_element_type=jnp.float32)
            .astype(x.dtype)
            .astype(jnp.float32)
        ).astype(x.dtype)
        shared = (shared * gate_value).astype(x.dtype)
        out = lax.psum(out, "tp") + lax.psum(shared, "tp")
        return jnp.where(mask != 0, 0, out), jnp.zeros((8,), jnp.int32)

    def loop(rank, x, *rest):
        bank = jnp.stack((x, x.at[0].set(x[1]).at[1].set(x[0])), axis=0)

        def body(i, carry):
            y, _ = local(rank, lax.dynamic_index_in_dim(bank, i % 2, 0, False), *rest)
            return y, carry[1] + jnp.sum(y.astype(jnp.float32))

        y, cs = lax.fori_loop(
            0,
            iterations,
            body,
            (jnp.zeros_like(x), jnp.zeros((), jnp.float32)),
            unroll=False,
        )
        return y, cs.reshape((1,))

    run = loop if iterations else local
    spec = P("dp", "tp")

    def mapped_body(*args):
        y, c = run(*(a[0, 0] for a in args))
        return y[None, None], c[None, None]

    mapped = jax.shard_map(
        mapped_body,
        mesh=mesh,
        in_specs=(spec,) * 13,
        out_specs=(spec, spec),
        check_vma=False,
    )

    def forward(
        rank: jax.Array,
        x: jax.Array,
        router: jax.Array,
        gate: jax.Array,
        sw1: jax.Array,
        ss1: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return mapped(rank, x, router, gate, sw1, ss1, sw2, ss2, w1, w2, s1, s2, mask)

    return jax_op(
        f"pallas::ut_pr2821_{mode}_n{iterations}_diag{int(diagnostics)}",
        forward,
        mesh=mesh,
        input_partition_specs=(spec,) * 13,
        output_partition_specs=(spec, spec),
    )
