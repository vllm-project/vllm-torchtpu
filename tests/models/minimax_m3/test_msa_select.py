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
"""Pallas MSA block selection (interpret mode) against a numpy reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.models.minimax_m3.kernels.msa_select import (
    select_blocks,
    select_folded_blocks,
)

BLOCK = 16


def _reference(scores, pos, topk, init_blocks, local_blocks):
    out = np.full((len(pos), topk), -1, np.int32)
    for t, p in enumerate(pos):
        visible = p // BLOCK + 1
        s = scores[t, :visible].astype(np.float64).copy()
        s[max(visible - local_blocks, 0) :] = 1e29
        s[: min(init_blocks, visible)] = 1e30
        # Stable descending order: ties go to the lower block, like argmax.
        order = np.argsort(-s, kind="stable")[:topk]
        out[t, : len(order)] = order
    return out


@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 1), (0, 3)])
def test_matches_reference(init_blocks, local_blocks):
    rng = np.random.default_rng(init_blocks + local_blocks)
    num_tokens, num_blocks, topk = 300, 64, 16
    scores = rng.standard_normal((num_tokens, num_blocks)).astype(np.float32)
    scores[::7, 3] = scores[::7, 9]  # exact ties
    pos = rng.integers(0, num_blocks * BLOCK, num_tokens).astype(np.int32)
    picked = select_blocks(
        jnp.asarray(scores)[:, None],
        jnp.asarray(pos),
        block_size=BLOCK,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_rows=64,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(picked)[:, 0],
        _reference(scores, pos, topk, init_blocks, local_blocks),
    )


def _fold(scores, width, lanes=128):
    """``[T, P]`` -> `msa_indexer.folded_block_scores`' layout; padding lanes
    hold NaN, which the selection must never read."""
    num_tokens, num_blocks = scores.shape
    padded = np.full((num_blocks, width), np.nan, np.float32)
    padded[:, :num_tokens] = scores.T
    return padded.reshape(num_blocks * width // lanes, lanes)


@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 1), (0, 3)])
@pytest.mark.parametrize("num_tokens,width", [(5, 8), (64, 64), (100, 128)])
def test_folded_matches_reference(num_tokens, width, init_blocks, local_blocks):
    rng = np.random.default_rng(num_tokens + init_blocks + local_blocks)
    num_blocks, topk = 256, 16
    # Few distinct values: many exact ties, broken by the lower block.
    scores = rng.integers(0, 6, (num_tokens, num_blocks)).astype(np.float32)
    # Short contexts leave whole chunks of pages unscanned; NaN marks the
    # scores no token may see.
    pos = rng.integers(0, 90 * BLOCK, num_tokens).astype(np.int32)
    scores[np.arange(num_blocks)[None] > (pos // BLOCK)[:, None]] = np.nan
    picked = select_folded_blocks(
        jnp.asarray(_fold(scores, width)),
        jnp.asarray(pos),
        width=width,
        block_size=BLOCK,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        chunk_rows=8,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(picked)[:, 0],
        _reference(scores, pos, topk, init_blocks, local_blocks),
    )


def test_lowers_for_tpu():
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    args = (
        jax.ShapeDtypeStruct((8192, 1, 2048), jnp.float32),
        jax.ShapeDtypeStruct((8192,), jnp.int32),
    )
    select = functools.partial(
        select_blocks, block_size=128, topk=16, init_blocks=0, local_blocks=1
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(select).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("num_tokens,width", [(4, 8), (64, 64), (96, 128)])
def test_folded_lowers_for_tpu(num_tokens, width):
    args = (
        jax.ShapeDtypeStruct((2048 * width // 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((num_tokens,), jnp.int32),
    )
    select = functools.partial(
        select_folded_blocks,
        width=width,
        block_size=128,
        topk=16,
        init_blocks=0,
        local_blocks=1,
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(select).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
