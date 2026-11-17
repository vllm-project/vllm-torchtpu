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
TPU interpret mode on CPU, one shard at a time with the shard's index as an
operand.

The kernel reads three tables (block offsets, packed shifts and expert bases)
back through a one-hot matmul as three biased bytes, which are exact only in
a bounded range. ``check_lookup_range`` refuses a call whose values could
leave it. These tests check the bound's arithmetic at the Qwen3.5 decode
shapes and at a wide mesh, and, with the lookup's window moved down so its
edge is reachable at a small shape, that the kernel really drops pairs past
the edge and that the check refuses those calls and admits exact ones.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import routing_tables as tables
from vllm_torchtpu.kernels.fused_moe.v3 import routing_tables_kernel as kernel

pytestmark = pytest.mark.cpu_test

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


@pytest.mark.parametrize("tokens", [32, 256, 1024, 4096])
def test_the_qwen35_decode_shapes_are_in_range(tokens):
    """EP8, 512 experts, top-k 10, tile 128 (the Qwen3.5-397B decode legs, at
    most 1024 rows per rank at DP4 x TP2 and 256 at TP8), with headroom."""
    kernel.check_lookup_range(**_bounds(8, 512, tokens, 10, 128))


def test_a_wide_mesh_is_refused():
    """EP32, 256 experts, top-k 8, 2048 tokens per shard, tile 32: the last
    shard's expert base is about 31 * 526,112, past the biased range."""
    bounds = _bounds(32, 256, 2048, 8, 32)
    assert bounds["routed_rows_per_shard"] == 526_112
    with pytest.raises(ValueError, match="expert base lookup"):
        kernel.check_lookup_range(**bounds)


def test_routing_without_distinct_experts_has_the_wider_bound():
    """A shard that may send all its pairs to one expert (a caller's routing)
    shifts arrival rows by up to (shards - 1) * pairs: at EP8 and 1024 tokens
    that leaves the range, while one pair per token per expert does not."""
    kernel.check_lookup_range(**_bounds(8, 512, 1024, 10, 128, distinct=True))
    with pytest.raises(ValueError, match="packed shift lookup"):
        kernel.check_lookup_range(**_bounds(8, 512, 1024, 10, 128, distinct=False))


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


def _shard_tables(messages, shard, routed_rows_per_shard):
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
        interpret=True,
        shard_index=jnp.asarray([shard], jnp.int32),
        pairs_per_expert=TOKENS,
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
