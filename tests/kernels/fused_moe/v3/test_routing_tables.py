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
"""The routing tables: one routed row per pair and no more, the per-shard
slices,
the layout helpers, and the two front ends bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import (
    device,
    layout,
    routing_tables,
    routing_tables_kernel,
    transport,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES, ROW_BLOCK

SERVED_EXPERTS, SERVED_TOP_K, SERVED_SHARDS = 512, 10, 8


def routed_rows_per_shard(tokens, top_k, experts, tile_rows):
    return routing_tables.routed_rows_bound(tokens, top_k, experts, tile_rows)


def make_routing(indices, *, experts, num_shards, block, tile_rows):
    tokens, top_k = indices.shape
    return routing_tables.build_routing_tables(
        jnp.asarray(indices, jnp.int32),
        num_experts=experts,
        num_shards=num_shards,
        tokens_per_shard=tokens // num_shards,
        block=block,
        tile_rows=tile_rows,
        routed_rows_per_shard=routed_rows_per_shard(tokens, top_k, experts, tile_rows),
    )


def token_table_all(routing, *, num_shards, stride):
    """The per-shard token tables laid back out as one table."""
    return np.concatenate(
        [
            np.asarray(
                routing_tables.shard_token_table(
                    routing, jnp.int32(s), routed_rows_per_shard=stride
                )
            )
            for s in range(num_shards)
        ]
    )


def random_indices(case, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, case["experts"], (case["tokens"], case["top_k"]))


def one_expert(expert):
    return lambda case, seed: np.full((case["tokens"], case["top_k"]), expert, np.int32)


@pytest.mark.cpu_test
class RoutingTablesTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name="e8_s4",
            experts=8,
            num_shards=4,
            tokens=32,
            top_k=2,
            block=8,
            tile_rows=32,
        ),
        dict(
            testcase_name="e16_s8",
            experts=16,
            num_shards=8,
            tokens=64,
            top_k=4,
            block=32,
            tile_rows=32,
        ),
        dict(
            testcase_name="e4_s2",
            experts=4,
            num_shards=2,
            tokens=16,
            top_k=3,
            block=8,
            tile_rows=16,
        ),
        dict(
            testcase_name="e64_s8",
            experts=64,
            num_shards=8,
            tokens=64,
            top_k=6,
            block=16,
            tile_rows=64,
        ),
        dict(
            testcase_name="all_on_the_first_expert",
            experts=8,
            num_shards=4,
            tokens=32,
            top_k=2,
            block=8,
            tile_rows=32,
            indices=one_expert(0),
        ),
        dict(
            testcase_name="all_on_the_last_expert",
            experts=8,
            num_shards=4,
            tokens=32,
            top_k=2,
            block=8,
            tile_rows=32,
            indices=one_expert(7),
        ),
        dict(
            testcase_name="all_on_a_mid_shard_expert",
            experts=8,
            num_shards=4,
            tokens=32,
            top_k=2,
            block=8,
            tile_rows=32,
            indices=one_expert(3),
        ),
        dict(
            testcase_name="served",
            experts=SERVED_EXPERTS,
            num_shards=SERVED_SHARDS,
            tokens=8192,
            top_k=SERVED_TOP_K,
            block=256,
            tile_rows=128,
            seeds=(11,),
        ),
    )
    def test_every_pair_lands_on_a_routed_row_of_its_own(
        self, indices=random_indices, seeds=(0, 1, 2, 3), **case
    ):
        for seed in seeds:
            routing = make_routing(
                indices(case, seed),
                experts=case["experts"],
                num_shards=case["num_shards"],
                block=case["block"],
                tile_rows=case["tile_rows"],
            )
            counts = np.asarray(
                jnp.zeros((routing.routed_rows + 1,), jnp.int32)
                .at[routing.routed_row]
                .add(1)
            )
            self.assertEqual(
                int(counts.max()), 1, f"two pairs share a routed row (seed {seed})"
            )
            self.assertEqual(int(counts.sum()), case["tokens"] * case["top_k"])
            stride = routed_rows_per_shard(
                case["tokens"], case["top_k"], case["experts"], case["tile_rows"]
            )
            token_of_pair = np.arange(case["tokens"] * case["top_k"]) // case["top_k"]
            np.testing.assert_array_equal(
                token_table_all(routing, num_shards=case["num_shards"], stride=stride)[
                    np.asarray(routing.routed_row)
                ],
                token_of_pair,
            )

    def test_an_empty_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one token"):
            routing_tables.build_routing_tables(
                jnp.zeros((0, 2), jnp.int32),
                num_experts=4,
                num_shards=2,
                tokens_per_shard=0,
                block=8,
                tile_rows=16,
                routed_rows_per_shard=16,
            )

    def test_the_token_table_stays_inside_the_token_range(self):
        """The kernel fetches rows by this table with bounds checks off. An
        expert id past the last expert is clamped rather than trusted."""
        experts, num_shards, tokens, top_k = 8, 4, 32, 2
        rng = np.random.default_rng(9)
        indices = rng.integers(0, experts, size=(tokens, top_k))
        indices[5, 0] = experts
        routing = make_routing(
            indices, experts=experts, num_shards=num_shards, block=8, tile_rows=32
        )
        table = token_table_all(
            routing,
            num_shards=num_shards,
            stride=routed_rows_per_shard(tokens, top_k, experts, 32),
        )
        self.assertGreaterEqual(int(table.min()), 0)
        self.assertLess(int(table.max()), tokens)

    @parameterized.named_parameters(
        ("served", 1024, 10), ("odd_top_k", 64, 3), ("tiny", 4, 1)
    )
    def test_the_routing_block_is_the_widest_that_divides(
        self, tokens_per_shard, top_k
    ):
        block = routing_tables.routing_block(tokens_per_shard, top_k)
        self.assertEqual((tokens_per_shard * top_k) % block, 0)
        self.assertLessEqual(block, routing_tables.MAX_ROUTING_BLOCK)
        self.assertEqual(block & (block - 1), 0)
        self.assertTrue(
            2 * block > routing_tables.MAX_ROUTING_BLOCK
            or (tokens_per_shard * top_k) % (2 * block) != 0
        )

    def test_the_routed_rows_bound_holds_the_worst_case_in_whole_tiles(self):
        tokens, top_k, experts, tile_rows = 512, 4, 32, 128
        worst = tokens * top_k + (ROW_BLOCK - 1) * experts + tile_rows
        bound = routing_tables.routed_rows_bound(tokens, top_k, experts, tile_rows)
        self.assertEqual(bound % tile_rows, 0)
        self.assertGreaterEqual(bound, worst)
        self.assertLess(bound - tile_rows, worst)

    @parameterized.named_parameters(
        ("heaviest_first", [0, 24, 8, 0, 40, 8], [4, 1, 2, 5]),
        ("ties_by_ascending_index", [8, 8, 16, 8], [2, 0, 1, 3]),
    )
    def test_the_visit_order(self, rows, order):
        rows = jnp.asarray(rows, jnp.int32)
        visit = routing_tables.visit_order(rows, rows.shape[0])
        np.testing.assert_array_equal(np.asarray(visit)[: len(order)], order)

    @parameterized.named_parameters(("seed0", 0), ("seed3", 3))
    def test_the_shard_cuts_agree_with_the_replicated_tables(self, seed):
        """On every shard: the expert rows are the shard's own experts with
        bases relative to its first. The per-destination runs start at zero,
        follow each other and cover the expert's rows. The row slice and
        the block slice agree on where a push lands. The true rows never
        exceed
        the padded ones."""
        experts, num_shards, tokens = 8, 4, 32
        top_k, block, tile_rows = 2, 8, 32
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, experts, size=(tokens, top_k))
        routing = make_routing(
            indices,
            experts=experts,
            num_shards=num_shards,
            block=block,
            tile_rows=tile_rows,
        )
        per_shard = experts // num_shards
        expert_rows_all = np.asarray(routing.expert_rows_aligned)
        for shard in range(num_shards):
            kw = dict(num_experts=experts, num_shards=num_shards)
            start, blocks, _, push_blocks, push_destination = [
                np.asarray(t)
                for t in routing_tables.shard_transport_tables(routing, shard, **kw)
            ]
            push_rows, arrival_offset = [
                np.asarray(t)
                for t in routing_tables.shard_push_tables(routing, shard, **kw)
            ]
            rows, base = routing_tables.shard_expert_rows(routing, shard, **kw)
            mine = expert_rows_all[shard * per_shard : (shard + 1) * per_shard]
            np.testing.assert_array_equal(np.asarray(rows), mine)
            np.testing.assert_array_equal(np.asarray(base), np.cumsum(mine) - mine)
            for g in range(per_shard):
                np.testing.assert_array_equal(
                    start[g], np.cumsum(blocks[g]) - blocks[g]
                )
                self.assertEqual(int(blocks[g].sum()) * ROW_BLOCK, int(rows[g]))
            np.testing.assert_array_equal(arrival_offset, push_destination * ROW_BLOCK)
            np.testing.assert_array_equal(push_blocks, blocks)
            self.assertTrue((push_rows <= blocks * ROW_BLOCK).all())


@pytest.mark.cpu_test
class GeometryTest(parameterized.TestCase):
    @parameterized.parameters(512, 1024, 4096)
    def test_a_row_is_whole_lane_blocks(self, hidden):
        self.assertEqual(layout.row_lane_blocks(hidden) * LANES, hidden)

    def test_the_packed_word_tiling_is_checked_against_the_chip(self):
        class Info:
            def __init__(self, tiling):
                self.tiling = tiling

            def get_sublane_tiling(self, dtype):
                return self.tiling

        self.assertIsNone(device.check_u32_sublane_tile(Info(layout.SUBLANES)))
        with self.assertRaisesRegex(ValueError, "sublanes"):
            device.check_u32_sublane_tile(Info(layout.SUBLANES * 2))

    def test_generation_scope_is_refused_by_name(self):
        class Info:
            generation = 6

        with self.assertRaisesRegex(ValueError, "generation 6"):
            device.check_generation(Info())


@pytest.mark.multichip
class ShardTablesKernelTest(parameterized.TestCase):
    """The shard-tables kernel against the reference form of the tables:
    every output bit for bit, on device. The kernel reads the routing out
    of the gathered routing messages (each shard's pair indices flat,
    slot-major, in its message's first rows; the rows after are noise it
    must not read); the reference numbers pairs row-major over its [T, K]
    table, so it is fed the same words in that shape."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu" or not device.supported_on(jax.devices()[0]):
            self.skipTest("the tables kernel runs on the supported TPU")
        if len(jax.devices()) < 8:
            self.skipTest("8 devices are needed")

    @parameterized.parameters((8192, 10, 512), (2048, 4, 128), (1024, 8, 256))
    def test_kernel_matches_the_reference(self, tokens, top_k, experts):
        num_shards = 8
        tokens_per_shard = tokens // num_shards
        pairs = tokens_per_shard * top_k
        self.assertEqual(
            pairs
            % routing_tables_kernel.step_pairs_multiple(
                routing_tables_kernel.BLOCKS_PER_STEP
            ),
            0,
        )
        block = routing_tables_kernel.shard_tables_block(pairs)
        stride = routing_tables.routed_rows_bound(tokens, top_k, experts, 128)
        scale_words = layout.scale_table_rows(stride) * layout.LANES
        message_rows = transport.message_rows(tokens_per_shard, pairs)
        rng = np.random.default_rng(7)
        topk = rng.integers(0, experts, (tokens, top_k), dtype=np.int32)
        words = topk.reshape(num_shards, tokens_per_shard, top_k).transpose(0, 2, 1)
        messages = rng.integers(
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
            size=(num_shards, message_rows, transport.MESSAGE_LANES),
            dtype=np.int32,
        )
        messages[:, : pairs // transport.MESSAGE_LANES, :] = words.reshape(
            num_shards, pairs // transport.MESSAGE_LANES, transport.MESSAGE_LANES
        )
        topk_g = jnp.asarray(words.reshape(tokens, top_k))
        messages_g = jnp.asarray(messages)
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:num_shards]), (layout.MESH_AXIS,)
        )
        kw = dict(num_experts=experts, num_shards=num_shards)

        def reference(_):
            shard = jax.lax.axis_index(layout.MESH_AXIS)
            routing = routing_tables.build_routing_tables(
                topk_g,
                tokens_per_shard=tokens_per_shard,
                block=block,
                tile_rows=128,
                routed_rows_per_shard=stride,
                **kw,
            )
            rows, base = routing_tables.shard_expert_rows(routing, shard, **kw)
            arrival = jax.lax.dynamic_slice(
                routing.arrival_row,
                (shard * tokens_per_shard, 0),
                (tokens_per_shard, top_k),
            ).reshape(-1)
            return tuple(
                x[None]
                for x in (
                    *routing_tables.shard_transport_tables(routing, shard, **kw),
                    *routing_tables.shard_push_tables(routing, shard, **kw),
                    rows,
                    base,
                    routing_tables.shard_counts(routing, rows, shard, **kw),
                    routing_tables.visit_order(rows, experts // num_shards),
                    routing.run_rows,
                    arrival,
                    routing_tables.local_routed_rows(
                        routing, shard, routed_rows_per_shard=stride
                    ),
                    jnp.zeros((stride,), jnp.int32),
                    jnp.zeros((scale_words,), jnp.int32),
                )
            )

        def kernel(_):
            (
                run_rows,
                arrival,
                routed_row,
                tables,
                rows,
                base,
                counts,
                visit,
                *zeroed,
            ) = routing_tables_kernel.shard_tables_kernel(
                messages_g,
                pairs_per_shard=pairs,
                routed_rows_per_shard=stride,
                block=block,
                row_block=layout.ROW_BLOCK,
                slot_field=routing_tables.ALIGNMENT_SLOT_FIELD,
                axis=layout.MESH_AXIS,
                zero_tables=(stride, scale_words),
                **kw,
            )
            return tuple(
                x[None]
                for x in (
                    *tables,
                    rows,
                    base,
                    counts,
                    visit,
                    run_rows,
                    arrival.reshape(-1),
                    routed_row,
                    *zeroed,
                )
            )

        spec = jax.sharding.PartitionSpec(layout.MESH_AXIS)

        def run(body):
            return jax.jit(
                jax.shard_map(
                    body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
                )
            )(jnp.zeros((num_shards,), jnp.int32))

        names = (
            "run_start",
            "run_blocks",
            "outgoing_offset",
            "push_blocks",
            "push_destination",
            "push_rows",
            "arrival_offset",
            "expert_rows",
            "expert_base",
            "counts",
            "visit_order",
            "run_rows",
            "arrival_row",
            "routed_row",
            "zero_tokens",
            "zero_scales",
        )
        want, got = run(reference), run(kernel)
        self.assertEqual(len(want), len(names))
        self.assertEqual(len(got), len(names))
        for name, a, b in zip(names, want, got):
            np.testing.assert_array_equal(np.asarray(b), np.asarray(a), err_msg=name)


if __name__ == "__main__":
    absltest.main()
