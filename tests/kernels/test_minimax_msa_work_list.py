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
"""MSA attention work list: the single-tile path equals the general one, and
`_live_groups` marks exactly the first ``num_groups`` groups."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention as msa
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse as lse


def _selections(rng, tokens: int, topk: int) -> list[np.ndarray]:
    sparse = rng.integers(0, 300, (tokens, topk))
    sparse[rng.random(tokens) < 0.5] = -1
    return [
        rng.integers(-1, 6, (tokens, topk)),  # duplicate-heavy
        rng.integers(-1, 5000, (tokens, topk)),  # dispersed
        sparse,
        -np.ones((tokens, topk), np.int64),  # no page
        np.arange(tokens * topk).reshape(tokens, topk),  # all distinct
    ]


@pytest.mark.parametrize(
    "block_q,topk,pages_per_step", [(8, 16, 16), (8, 16, 32), (16, 16, 16), (8, 4, 16)]
)
def test_single_tile_matches_general(
    block_q: int, topk: int, pages_per_step: int
) -> None:
    rng = np.random.default_rng(block_q * topk + pages_per_step)
    groups_per_tile = msa._groups_per_tile(block_q, topk, pages_per_step)
    for sel in _selections(rng, block_q, topk):
        per_tile = (
            jnp.where(sel < 0, msa._NO_PAGE, sel).reshape(1, -1).astype(jnp.int32)
        )
        for union in (msa._dense_unions(per_tile), msa._sorted_unions(per_tile)):
            want = msa._tile_groups(union, groups_per_tile, pages_per_step)
            got = msa._single_tile_groups(union[0], groups_per_tile, pages_per_step)
            np.testing.assert_array_equal(got[0], want[0])  # meta
            np.testing.assert_array_equal(got[1], want[1])  # count


@pytest.mark.parametrize("tiles", [1, 2, 3, 6])
def test_live_groups_are_the_first_num_groups(tiles: int) -> None:
    block_q, topk, pages_per_step = 8, 16, 16
    rng = np.random.default_rng(tiles)
    groups_per_tile = msa._groups_per_tile(block_q, topk, pages_per_step)
    for sel in _selections(rng, tiles * block_q, topk):
        per_tile = jnp.where(sel < 0, msa._NO_PAGE, sel).reshape(tiles, -1)
        per_tile = per_tile.astype(jnp.int32)
        for union in (msa._dense_unions(per_tile), msa._sorted_unions(per_tile)):
            meta, count, num_groups = msa._tile_groups(
                union, groups_per_tile, pages_per_step
            )
            want = np.arange(meta.shape[0]) < int(num_groups)
            np.testing.assert_array_equal(msa._live_groups(meta, count), want)


# An odd length (the last word's high half is padding), ids past the sign bit.
@pytest.mark.parametrize("length", [10, 11])
def test_table16_entries_read_back(length: int) -> None:
    rng = np.random.default_rng(length)
    table = jnp.asarray(rng.integers(0, 1 << 16, length), jnp.int32)
    packed = msa.pack_table16(table, 1 << 16)
    assert msa.pack_table16(table, (1 << 16) + 1) is None

    def kernel(packed_ref, o_ref):
        for i in range(length):
            o_ref[i] = msa.table16_entry(packed_ref, i)

    got = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        out_shape=jax.ShapeDtypeStruct((length,), jnp.int32),
        interpret=True,
    )(packed)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(table))


# One tile, a step's worth (4), steps of 3 and of 1, and a long call.
@pytest.mark.parametrize("tiles", [1, 4, 6, 3, 17])
def test_kernel_group_words_match_xla(tiles: int) -> None:
    """The kernel's scalar group words (`_tile_group_words`) equal
    `_packed_work`'s: meta words and counts of the groups in use, and every
    step start."""
    block_q, topk, pages_per_step = 8, 16, 16
    rng = np.random.default_rng(tiles)
    groups_per_tile = msa._groups_per_tile(block_q, topk, pages_per_step)
    max_groups = tiles * groups_per_tile
    for sel in _selections(rng, tiles * block_q, topk):
        sel = jnp.asarray(sel, jnp.int32)[None]
        want, tiles_per_step = lse._packed_work(sel, block_q, pages_per_step)
        row, steps = lse._packed_work(sel, block_q, pages_per_step, in_kernel=True)
        assert steps == tiles_per_step
        union_len = max_groups * pages_per_step
        np.testing.assert_array_equal(row[0, :union_len], want[0, :union_len])
        num_steps = tiles // tiles_per_step

        kernel = functools.partial(
            lse._tile_group_words,
            groups_per_tile=groups_per_tile,
            pages_per_step=pages_per_step,
            tiles_per_step=tiles_per_step,
        )
        got = pl.pallas_call(
            kernel,
            in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
            out_shape=jax.ShapeDtypeStruct(
                (2 * max_groups + num_steps + 2,), jnp.int32
            ),
            interpret=True,
        )(row[0, union_len:])
        got, want = np.asarray(got), np.asarray(want[0, union_len:])
        meta, count = want[:max_groups], want[max_groups : 2 * max_groups]
        live = np.asarray(msa._live_groups(meta, count))
        used = int(live.sum())
        np.testing.assert_array_equal(got[:used], meta[:used])
        np.testing.assert_array_equal(got[max_groups : max_groups + used], count[:used])
        np.testing.assert_array_equal(got[2 * max_groups : -1], want[2 * max_groups :])
