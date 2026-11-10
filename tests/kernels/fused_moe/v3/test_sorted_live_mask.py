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
"""CPU lowering and invariant tests for 1-based sorted-row live masking."""

import dataclasses
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import (
    AbstractDevice,
    AbstractMesh,
    NamedSharding,
    use_abstract_mesh,
)
from jax.sharding import (
    PartitionSpec as P,
)

from vllm_torchtpu.kernels.fused_moe import v3 as fused_ep
from vllm_torchtpu.kernels.fused_moe.v3 import (
    Config,
    layout,
    routing_tables,
    row_tables_kernel,
)
from vllm_torchtpu.kernels.fused_moe.v3 import (
    ffn_kernel as ffn_kernel_mod,
)
from vllm_torchtpu.kernels.fused_moe.v3 import (
    layer as layer_mod,
)
from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel as gather


def _run_production_decode_scatter(
    routed_row: np.ndarray,
    *,
    tokens_per_shard: int,
    message_tokens: int,
    pairs_per_shard: int,
    routed_rows_table_len: int,
    valid_rows: int,
    token_bias: int,
) -> np.ndarray:
    """Executes `row_tables_kernel.decode_and_clamp_pairs` directly."""
    pairs = jnp.arange(routed_row.shape[0], dtype=jnp.int32)
    biased_tokens, clamped_rows, _, _ = row_tables_kernel.decode_and_clamp_pairs(
        pairs,
        jnp.asarray(routed_row, dtype=jnp.int32),
        inv_pairs=jnp.float32(1.0 / pairs_per_shard),
        inv_tokens=jnp.float32(1.0 / message_tokens),
        pairs_per_shard=pairs_per_shard,
        message_tokens=message_tokens,
        tokens_per_shard=tokens_per_shard,
        routed_rows=routed_rows_table_len,
        valid_rows=valid_rows,
        token_bias=token_bias,
    )
    table = np.zeros((routed_rows_table_len,), dtype=np.int32)
    bt_np = np.asarray(biased_tokens, dtype=np.int32)
    cr_np = np.asarray(clamped_rows, dtype=np.int32)
    keep = cr_np < routed_rows_table_len
    table[cr_np[keep]] = bt_np[keep]
    return table


class SortedLiveMaskTest(parameterized.TestCase):
    @parameterized.parameters(
        {"valid_rows": 257},
        {"valid_rows": -1},
        {"valid_rows": True},
        {"token_bias": -1},
        {"token_bias": True},
        {"token_bias": 1 << 31},
    )
    def test_scatter_row_tables_validates_valid_rows_and_token_bias(self, **bad_kw):
        routed_row = jax.ShapeDtypeStruct((1024,), jnp.int32)
        messages = jax.ShapeDtypeStruct((2048,), jnp.int32)
        token_table = jax.ShapeDtypeStruct((256,), jnp.int32)
        scale_table = jax.ShapeDtypeStruct((256,), jnp.int32)
        with self.assertRaisesRegex(ValueError, "valid_rows|token_bias"):
            jax.jit(
                lambda r, m, t, s: row_tables_kernel.scatter_row_tables(
                    r,
                    m,
                    t,
                    s,
                    top_k=2,
                    tokens_per_shard=64,
                    message_words=256,
                    scale_offset=128,
                    **bad_kw,
                )
            ).trace(routed_row, messages, token_table, scale_table)

    @parameterized.parameters(
        (4, 16, 2, 8, 8, 128),
        (4, 16, 2, 16, 16, 128),
        (4, 32, 4, 16, 32, 128),
        (8, 32, 4, 8, 64, 128),
        (4, 16, 2, 16, 128, 128),
    )
    def test_biased_token_mask_matches_edge_cumsum_exactly(
        self,
        num_shards,
        num_experts,
        top_k,
        tokens_per_shard,
        tile_rows,
        message_tokens,
    ):
        rng = np.random.default_rng(17 + tile_rows + num_shards)
        experts_per_shard = num_experts // num_shards
        pairs_per_shard = message_tokens * top_k
        mesh = AbstractMesh(
            (num_shards,),
            (layout.MESH_AXIS,),
            abstract_device=AbstractDevice("TPU7x", 1, "tpu"),
        )
        routed_rows_per_shard = routing_tables.routed_rows_bound(
            num_shards * tokens_per_shard, top_k, num_experts, tile_rows
        )
        table_len = layout.align_up(routed_rows_per_shard, layout.LANES)
        with use_abstract_mesh(mesh):
            n_sorted = layout.align_up(table_len, gather.gather_rows_multiple(16))
        total_gathered_rows = num_shards * tokens_per_shard

        flat_experts = np.full(
            (num_shards, top_k, message_tokens), num_experts, dtype=np.int32
        )
        for s in range(num_shards):
            flat_experts[s, :, :tokens_per_shard] = rng.integers(
                0, num_experts, size=(top_k, tokens_per_shard), dtype=np.int32
            )
            flat_experts[s, :, tokens_per_shard - 1] = num_experts
        flat_experts[0, 0, 0] = 0
        if top_k > 1:
            flat_experts[0, 1, 0] = experts_per_shard

        pair_experts = flat_experts.reshape(-1)
        tabs = routing_tables.build_routing_tables(
            jnp.asarray(flat_experts.reshape(num_shards * message_tokens, top_k)),
            num_experts=num_experts,
            num_shards=num_shards,
            tokens_per_shard=message_tokens,
            block=128,
            tile_rows=tile_rows,
            routed_rows_per_shard=routed_rows_per_shard,
        )
        global_routed_row = np.asarray(tabs.routed_row, dtype=np.int32)
        unrouted = pair_experts == num_experts
        done_ones = jnp.ones((1, layout.LANES), dtype=jnp.int32)

        for shard in range(num_shards):
            shard_base = shard * routed_rows_per_shard
            local_routed = np.where(
                unrouted | (global_routed_row < shard_base),
                routed_rows_per_shard,
                global_routed_row - shard_base,
            )
            table_zero = _run_production_decode_scatter(
                local_routed,
                tokens_per_shard=tokens_per_shard,
                message_tokens=message_tokens,
                pairs_per_shard=pairs_per_shard,
                routed_rows_table_len=table_len,
                valid_rows=table_len,
                token_bias=0,
            )
            e0 = shard * experts_per_shard
            e1 = (shard + 1) * experts_per_shard
            expert_base_full = np.asarray(tabs.expert_base, dtype=np.int32)[e0:e1]
            expert_base = expert_base_full - expert_base_full[0]
            run_start_rows = np.asarray(tabs.run_start_aligned, dtype=np.int32)[e0:e1]
            push_rows = np.asarray(tabs.run_rows, dtype=np.int32)[e0:e1]
            starts = (expert_base[:, None] + run_start_rows).reshape(-1)
            ends = starts + push_rows.reshape(-1)
            edges = np.zeros((n_sorted + 1,), dtype=np.int32)
            np.add.at(edges, np.minimum(starts, n_sorted), 1)
            np.add.at(edges, np.minimum(ends, n_sorted), -1)
            live_ref = np.cumsum(edges)[:n_sorted] > 0
            padded_zero = np.pad(table_zero, (0, n_sorted - table_len))
            row = np.arange(n_sorted, dtype=np.int32)
            want_index = np.where(live_ref, padded_zero, row % total_gathered_rows)

            table_biased = _run_production_decode_scatter(
                local_routed,
                tokens_per_shard=tokens_per_shard,
                message_tokens=message_tokens,
                pairs_per_shard=pairs_per_shard,
                routed_rows_table_len=table_len,
                valid_rows=routed_rows_per_shard,
                token_bias=1,
            )
            got_index = np.asarray(
                layer_mod._sorted_gather_index(
                    jnp.asarray(table_biased),
                    done_ones,
                    n_sorted=n_sorted,
                    num_gathered_rows=total_gathered_rows,
                )
            )
            np.testing.assert_array_equal(got_index, want_index)

    @parameterized.parameters(
        (1024, 512, 10, 128, False),
        (384, 512, 10, 128, False),
        (48, 512, 10, 16, False),
        (48, 32, 4, 16, True),
    )
    def test_sorted_rows_tpu_lowering_eliminates_edge_scatters(
        self, local_tokens, num_experts, top_k, tile_rows, bridge_mode
    ):
        mesh = AbstractMesh(
            (8,), (layout.MESH_AXIS,), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        experts_per_shard = num_experts // 8
        sh = NamedSharding(mesh, P(layout.MESH_AXIS, None)) if bridge_mode else None
        sh_w = (
            NamedSharding(mesh, P(layout.MESH_AXIS, None, None))
            if bridge_mode
            else None
        )
        x = jax.ShapeDtypeStruct((local_tokens * 8, 4096), jnp.bfloat16, sharding=sh)
        w1 = jax.ShapeDtypeStruct(
            (num_experts, 4096, 2048), jnp.float8_e4m3fn, sharding=sh_w
        )
        w2 = jax.ShapeDtypeStruct(
            (num_experts, 1024, 4096), jnp.float8_e4m3fn, sharding=sh_w
        )
        w1_s = jax.ShapeDtypeStruct((num_experts, 2048), jnp.float32, sharding=sh)
        w2_s = jax.ShapeDtypeStruct((num_experts, 4096), jnp.float32, sharding=sh)
        logits = jax.ShapeDtypeStruct(
            (local_tokens * 8, num_experts), jnp.float32, sharding=sh
        )
        rank = (
            jax.ShapeDtypeStruct(
                (8, 1), jnp.int32, sharding=NamedSharding(mesh, P("d", None))
            )
            if bridge_mode
            else None
        )
        routed_rows_per_shard = routing_tables.routed_rows_bound(
            local_tokens * 8, top_k, num_experts, tile_rows
        )
        table_len = layout.align_up(routed_rows_per_shard, layout.LANES)
        cfg = dataclasses.replace(
            Config.for_shape(
                hidden=4096,
                inter=1024,
                experts_per_shard=experts_per_shard,
                top_k=top_k,
            ),
            sorted_rows=True,
        )

        def call_moe(x, w1, w2, w1_s, w2_s, logits, rank=None):
            return fused_ep.fused_ep_moe(
                x,
                w1,
                w2,
                w1_s,
                w2_s,
                logits,
                top_k=top_k,
                renormalize=True,
                mesh=mesh,
                tile_rows=tile_rows,
                config=cfg,
                shard_index=rank,
            )

        with (
            use_abstract_mesh(mesh),
            mock.patch.object(layer_mod, "WRAP_IN_JIT", not bridge_mode),
            mock.patch.object(layout, "SIDE_EFFECTS", not bridge_mode),
            mock.patch.object(layout, "SHARD_INDEX_OPERAND", bridge_mode),
            mock.patch.dict(ffn_kernel_mod._BUILD_CACHE, {}, clear=True),
            mock.patch.dict(layer_mod._LAYER_CACHE, {}, clear=True),
        ):
            n_sorted = layout.align_up(
                table_len, gather.gather_rows_multiple(cfg.gather_block_rows)
            )
            args = (
                (x, w1, w2, w1_s, w2_s, logits, rank)
                if bridge_mode
                else (x, w1, w2, w1_s, w2_s, logits)
            )
            jaxpr_text = str(jax.make_jaxpr(call_moe)(*args))
            hlo = (
                jax.jit(call_moe)
                .trace(*args)
                .lower(lowering_platforms=("tpu",))
                .as_text("hlo")
            )
        self.assertIn("_b1", jaxpr_text)
        if routed_rows_per_shard < table_len:
            self.assertIn(f"{routed_rows_per_shard}:i32[]", jaxpr_text)
        self.assertNotIn("scatter(", hlo)
        self.assertNotIn(f"s32[{n_sorted + 1}]", hlo)


if __name__ == "__main__":
    absltest.main()
