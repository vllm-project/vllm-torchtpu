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
"""Fused EP MoE v3's shard-tables kernel (``routing_tables_kernel``), run in
TPU interpret mode on CPU (the tables it zeroes on a TPU7x chip too), one
shard at a time with the shard's index as an operand.

The kernel reads three tables (block offsets, packed shifts and expert bases)
back through a one-hot matmul as three biased bytes, which are exact only in
a bounded range. ``check_lookup_range`` refuses a call whose values could
leave it. These tests check the bound's arithmetic at the Qwen3.5 decode
shapes and at a wide mesh, and, with the lookup's window moved down so its
edge is reachable at a small shape, that the kernel really drops pairs past
the edge and that the check refuses those calls and admits exact ones.

Every table the kernel emits is also checked against ``routing_tables``'
reference passes (dense jnp ops over the same pairs).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import routing_tables as tables
from vllm_torchtpu.kernels.fused_moe.v3 import routing_tables_kernel as kernel

LANES = 128
ROW_BLOCK = 8


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


@pytest.fixture
def interpret(monkeypatch):
    """TPU interpret mode with a TPU7x chip record, on the CPU device."""
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", _tpu7x)
    with (
        jax.default_device(jax.devices("cpu")[0]),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
    ):
        yield


@pytest.fixture(
    params=[
        pytest.param("interpret", marks=pytest.mark.cpu_test),
        pytest.param("tpu", marks=pytest.mark.tpu_test),
    ]
)
def interpret_or_chip(request, monkeypatch):
    """Whether the kernel runs in TPU interpret mode on CPU (True) or on the
    TPU chip (False)."""
    if request.param == "tpu":
        yield False
        return
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", _tpu7x)
    with (
        jax.default_device(jax.devices("cpu")[0]),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
    ):
        yield True


def _bounds(num_shards, num_experts, tokens, top_k, tile_rows, distinct=True):
    """check_lookup_range's arguments for a call of `tokens` per shard."""
    pairs = tokens * top_k
    return dict(
        num_experts=num_experts,
        num_shards=num_shards,
        pairs_per_shard=pairs,
        routed_rows_per_shard=tables.routed_rows_bound(
            num_shards * tokens, top_k, num_experts, tile_rows
        ),
        row_block=ROW_BLOCK,
        slot_field=tables.alignment_slot_field(num_shards),
        pairs_per_expert=tokens if distinct else pairs,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [32, 256, 1024, 4096])
def test_the_qwen35_decode_shapes_are_in_range(tokens):
    """EP8, 512 experts, top-k 10, tile 128 (the Qwen3.5-397B decode legs, at
    most 1024 rows per rank at DP4 x TP2 and 256 at TP8), with headroom."""
    kernel.check_lookup_range(**_bounds(8, 512, tokens, 10, 128))


@pytest.mark.cpu_test
def test_a_wide_mesh_is_refused():
    """EP32, 256 experts, top-k 8, 2048 tokens per shard, tile 32: the last
    shard's expert base is about 31 * 526,112, past the biased range."""
    bounds = _bounds(32, 256, 2048, 8, 32)
    assert bounds["routed_rows_per_shard"] == 526_112
    with pytest.raises(ValueError, match="expert base lookup"):
        kernel.check_lookup_range(**bounds)


@pytest.mark.cpu_test
def test_routing_without_distinct_experts_has_the_wider_bound():
    """A shard that may send all its pairs to one expert (a caller's routing)
    shifts arrival rows by up to (shards - 1) * pairs: at EP8 and 1024 tokens
    that leaves the range, while one pair per token per expert does not."""
    kernel.check_lookup_range(**_bounds(8, 512, 1024, 10, 128, distinct=True))
    with pytest.raises(ValueError, match="packed shift lookup"):
        kernel.check_lookup_range(**_bounds(8, 512, 1024, 10, 128, distinct=False))


@pytest.mark.cpu_test
def test_the_shard_index_is_read_from_the_axis_the_peers_are_addressed_on():
    """Without a shard_index operand the kernel reads the shard's index from
    `axis`, and its hosted transport addresses the peers on MESH_AXIS: any
    other axis is refused before anything is built."""
    with pytest.raises(ValueError, match="addresses its peers on"):
        kernel.shard_tables_kernel(
            jnp.zeros((2, 8, LANES), jnp.int32),
            num_experts=16,
            num_shards=2,
            pairs_per_shard=1024,
            routed_rows_per_shard=512,
            block=1024,
            row_block=ROW_BLOCK,
            slot_field=tables.alignment_slot_field(2),
            axis="replica",
        )


# A two-shard world small enough for the interpreter. Each token selects
# TOP_K distinct experts.
SHARDS, EXPERTS, TOP_K, TOKENS, TILE = 2, 16, 4, 256, 32


def _messages(unrouted=0, seed=0):
    """(routing messages [SHARDS, rows, LANES], the pair list): each shard's
    pairs slot-major, as the select kernel writes them; `unrouted` tokens
    route nowhere (index EXPERTS)."""
    rng = np.random.default_rng(seed)
    idx = np.stack(
        [rng.permutation(EXPERTS)[:TOP_K] for _ in range(SHARDS * TOKENS)]
    ).astype(np.int32)
    if unrouted:
        idx[rng.choice(SHARDS * TOKENS, unrouted, replace=False)] = EXPERTS
    pairs = np.concatenate(
        [idx[s * TOKENS : (s + 1) * TOKENS].T.reshape(-1) for s in range(SHARDS)]
    )
    return jnp.asarray(pairs.reshape(SHARDS, -1, LANES)), pairs


def _shard_tables(messages, shard, routed_rows_per_shard, interpret=True, **kwargs):
    pairs_per_shard = TOKENS * TOP_K
    return kernel.shard_tables_kernel(
        messages,
        num_experts=EXPERTS,
        num_shards=SHARDS,
        pairs_per_shard=pairs_per_shard,
        routed_rows_per_shard=routed_rows_per_shard,
        block=kernel.shard_tables_block(pairs_per_shard),
        row_block=ROW_BLOCK,
        slot_field=tables.alignment_slot_field(SHARDS),
        axis="d",
        interpret=interpret,
        shard_index=jnp.asarray([shard], jnp.int32),
        pairs_per_expert=TOKENS,
        **kwargs,
    )


def _own_pairs_dropped(routed_row, pairs, shard, routed_rows_per_shard):
    """How many of the routed pairs this shard's experts own the tables send
    to the drop row."""
    owner = pairs // (EXPERTS // SHARDS)
    own = (pairs < EXPERTS) & (owner == shard)
    return int(np.sum(np.asarray(routed_row)[own] == routed_rows_per_shard))


@pytest.fixture
def low_edge_lookup(monkeypatch):
    """The lookup's window moved down to [2**17 - 2**24, 2**17), so a small
    call's expert bases reach its top edge. Returns the edge."""
    edge = 1 << 17
    monkeypatch.setattr(kernel, "LOOKUP_BIAS", (1 << 24) - edge)
    return edge


@pytest.mark.cpu_test
def test_the_check_admits_exact_calls_only(interpret, low_edge_lookup):
    """The most routed rows per shard the check admits (the edge over the
    shard count): every pair of shard 1's experts keeps its routed row."""
    routed_rows = low_edge_lookup // SHARDS
    assert routed_rows % TILE == 0
    messages, pairs = _messages()
    routed_row = _shard_tables(messages, 1, routed_rows)[2]
    assert _own_pairs_dropped(routed_row, pairs, 1, routed_rows) == 0
    with pytest.raises(ValueError, match="expert base lookup"):
        _shard_tables(messages, 1, routed_rows + TILE)


@pytest.mark.cpu_test
def test_the_check_refuses_the_call_the_lookup_would_corrupt(
    interpret, low_edge_lookup, monkeypatch
):
    """Routed rows per shard at the edge itself put shard 1's expert bases
    past it: with the check taken out, the tables send every pair of shard
    1's experts to the drop row, and with it the call raises."""
    routed_rows = low_edge_lookup
    messages, pairs = _messages()
    with pytest.raises(ValueError, match="expert base lookup"):
        _shard_tables(messages, 1, routed_rows)
    monkeypatch.setattr(kernel, "check_lookup_range", lambda **_: None)
    routed_row = _shard_tables(messages, 1, routed_rows)[2]
    owned = int(np.sum((pairs < EXPERTS) & (pairs >= EXPERTS // SHARDS)))
    assert _own_pairs_dropped(routed_row, pairs, 1, routed_rows) == owned > 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("unrouted", [0, 17], ids=["all_routed", "unrouted_tokens"])
def test_shard_tables_match_the_reference(interpret, unrouted):
    """Every table the kernel emits for each shard, against routing_tables'
    reference passes over the same pairs: routed and arrival rows (a pair
    that routes nowhere takes the drop row and arrival row zero), the FFN
    kernel's transport and push tables, the expert rows and bases, the
    counts, the visit order and the run rows."""
    messages, pairs = _messages(unrouted=unrouted)
    pairs_per_shard = TOKENS * TOP_K
    routed_rows = tables.routed_rows_bound(SHARDS * TOKENS, TOP_K, EXPERTS, TILE)
    ref = tables.build_routing_tables(
        jnp.asarray(pairs.reshape(-1, TOP_K)),
        num_experts=EXPERTS,
        num_shards=SHARDS,
        tokens_per_shard=TOKENS,
        block=kernel.shard_tables_block(pairs_per_shard),
        tile_rows=TILE,
        routed_rows_per_shard=routed_rows,
    )
    live = pairs < EXPERTS
    for shard in range(SHARDS):
        (run_rows, arrival, routed_row, ffn_tables, rows, base, counts, visit) = (
            _shard_tables(messages, shard, routed_rows)[:8]
        )
        own = slice(shard * pairs_per_shard, (shard + 1) * pairs_per_shard)
        want_routed = tables.local_routed_rows(
            ref, shard, routed_rows_per_shard=routed_rows
        )
        np.testing.assert_array_equal(
            np.asarray(routed_row), np.where(live, want_routed, routed_rows)
        )
        want_arrival = np.asarray(ref.arrival_row).reshape(-1)[own]
        np.testing.assert_array_equal(
            np.asarray(arrival).reshape(-1), np.where(live[own], want_arrival, 0)
        )
        want_rows, want_base = tables.shard_expert_rows(
            ref, shard, num_experts=EXPERTS, num_shards=SHARDS
        )
        want_ffn = tables.shard_transport_tables(
            ref, shard, num_experts=EXPERTS, num_shards=SHARDS
        ) + tables.shard_push_tables(ref, shard, num_experts=EXPERTS, num_shards=SHARDS)
        assert len(ffn_tables) == len(want_ffn)
        for got, want in zip(ffn_tables, want_ffn):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        np.testing.assert_array_equal(np.asarray(rows), np.asarray(want_rows))
        np.testing.assert_array_equal(np.asarray(base), np.asarray(want_base))
        want_counts = tables.shard_counts(
            ref, want_rows, shard, num_experts=EXPERTS, num_shards=SHARDS
        )
        np.testing.assert_array_equal(np.asarray(counts), np.asarray(want_counts))
        np.testing.assert_array_equal(
            np.asarray(visit),
            np.asarray(tables.visit_order(want_rows, EXPERTS // SHARDS)),
        )
        np.testing.assert_array_equal(np.asarray(run_rows), np.asarray(ref.run_rows))


# The shard-tables call also zeroes the scatter's token and scale tables, a
# block per grid step. This world's call has 5 steps (2 groups of 8 blocks of
# 128 pairs, twice, and one more).
ZERO_WORDS_PER_STEP = 1024


@pytest.mark.parametrize(
    "zero_tables",
    [(896, 3200), (12_800,)],
    ids=["one_block_and_four", "past_the_steps"],
)
def test_the_tables_to_zero_come_back_zero(interpret_or_chip, zero_tables):
    """Every word of each table comes back zero. At 1024 words a step, an
    896-word table is one ragged block and a 3200-word table four blocks, the
    last ragged; a 12,800-word table would need 13 such blocks, more than the
    grid's 5 steps, so it is zeroed in 3072-word blocks, the last ragged."""
    messages, _ = _messages()
    routed_rows = tables.routed_rows_bound(SHARDS * TOKENS, TOP_K, EXPERTS, TILE)
    outs = _shard_tables(
        messages,
        0,
        routed_rows,
        interpret=interpret_or_chip,
        zero_tables=zero_tables,
        zero_words_per_step=ZERO_WORDS_PER_STEP,
    )
    zeroed = outs[8:]
    assert [tuple(z.shape) for z in zeroed] == [(n,) for n in zero_tables]
    for z in zeroed:
        np.testing.assert_array_equal(np.asarray(z), 0)


@pytest.mark.cpu_test
def test_more_than_two_tables_to_zero_are_refused():
    """The call zeroes the scatter's two tables, no more."""
    messages, _ = _messages()
    routed_rows = tables.routed_rows_bound(SHARDS * TOKENS, TOP_K, EXPERTS, TILE)
    with pytest.raises(ValueError, match="3 tables to zero"):
        _shard_tables(messages, 0, routed_rows, zero_tables=(1024, 1024, 1024))
