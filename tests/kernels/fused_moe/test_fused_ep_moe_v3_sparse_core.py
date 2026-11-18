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
"""Fused EP MoE v3's SparseCore programs: the arguments they refuse,
checked on CPU before anything is built.
"""

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import row_tables_kernel
from vllm_torchtpu.kernels.fused_moe.v3.config import Config

LANES = 128
BLOCK_ROWS = 16


# ---- refused before anything is built (CPU) ----


@pytest.fixture
def tpu7x_record(monkeypatch):
    """A TPU7x chip record, so the programs size themselves on CPU."""

    def record():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    monkeypatch.setattr(pltpu, "get_tpu_info", record)
    monkeypatch.setattr(tpu_info, "get_tpu_info", record)


def _scatter_shapes(n_pairs, pairs_per_shard, *, tokens, n_scales, routed_rows=512):
    """Abstract operands of scatter_row_tables: one shard's message holds its
    pair words, then n_scales planes of 128-word-aligned scale words."""
    plane_words = -(-tokens // LANES) * LANES
    message_words = pairs_per_shard + n_scales * plane_words
    shards = n_pairs // pairs_per_shard

    def words(n):
        return jax.ShapeDtypeStruct((n,), jnp.int32)

    operands = (
        words(n_pairs),
        words(shards * message_words),
        words(routed_rows),
        words(n_scales * routed_rows),
    )
    return operands, dict(
        top_k=1,
        tokens_per_shard=tokens,
        message_words=message_words,
        scale_offset=pairs_per_shard,
        n_scales=n_scales,
        plane_words=plane_words,
        pairs_per_shard=pairs_per_shard,
    )


@pytest.mark.cpu_test
def test_scatter_refuses_more_pairs_than_its_float_decode_is_exact_for():
    """The sparse cores decode a pair number with float multiplies, exact
    below MAX_PAIRS pairs."""
    pairs_per_shard = row_tables_kernel.MAX_PAIRS // 4
    operands, kwargs = _scatter_shapes(
        8 * pairs_per_shard, pairs_per_shard, tokens=128, n_scales=1
    )
    with pytest.raises(ValueError, match="exact up to"):
        row_tables_kernel.scatter_row_tables(*operands, **kwargs)


@pytest.mark.cpu_test
def test_scatter_refuses_a_short_chunk_with_several_scale_planes(tpu7x_record):
    """A chunk that is a subcore's whole run (32 pairs here) may be shorter
    than 128 words with one scale plane, but not with two: plane 1 of a ring
    slot would start 32 words in, off the 128-word alignment."""
    operands, kwargs = _scatter_shapes(1024, 512, tokens=128, n_scales=2)
    with pytest.raises(ValueError, match="with one scale plane"):
        row_tables_kernel.scatter_row_tables(*operands, **kwargs)


@pytest.mark.cpu_test
def test_the_config_takes_the_scatter_ring_the_scatter_takes():
    """Config refuses a one-slot scatter ring, as scatter_row_tables does (a
    chunk's fetch is issued when an earlier chunk retires)."""
    Config(scatter_ring=2)
    with pytest.raises(ValueError, match="scatter_ring 1"):
        Config(scatter_ring=1)
