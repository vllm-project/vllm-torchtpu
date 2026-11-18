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
"""Fused EP MoE v3's SparseCore programs: what they refuse (on CPU), and
their output on one TPU7x chip against numpy.

``row_gather_kernel.gather_rows_and_scales`` (the combine's gather of result
rows and their scales), ``row_gather_kernel.gather_rows_prefix`` (the sort of
the routed rows, over a routing-dependent prefix) and
``row_tables_kernel.scatter_row_tables`` (the token and scale tables the FFN
kernel reads) run with DMA bounds checks off and reuse their ring slots, so a
wrong slice or slot schedule reads or writes the wrong rows without an error.
Each runs here with more blocks or chunks than ring slots, and its output is
compared value for value with the numpy indexing it stands for.

The programs do not run in the TPU interpreter on CPU (its SparseCore
lowering rejects the index-driven copies), so their output needs a TPU7x
chip; the arguments they refuse are checked on CPU, before anything is built.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import device, row_tables_kernel
from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel as gather
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


# ---- the programs' output on a TPU7x chip ----


@pytest.fixture
def tpu7x():
    """Skip unless the first device is a chip generation the package
    supports (a fixture, so nothing builds a client while pytest collects)."""
    if not device.supported_on(jax.devices()[0]):
        pytest.skip("fused EP MoE v3's SparseCore programs need a TPU7x chip")


def _rows(n, lane_blocks, dtype, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n, lane_blocks, LANES)).astype(np.float32)
    return jnp.asarray(values).astype(dtype)


def _as_bits(a):
    """An fp8 or bf16 array's raw bits, so NaN payloads compare too."""
    a = np.asarray(a)
    return a.view(np.uint8 if a.dtype.itemsize == 1 else np.uint16)


def _gather_index(n_out, n_rows, seed):
    """Every row at least once where it fits, the rest random with repeats."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n_rows, n_out)
    index[: min(n_out, n_rows)] = rng.permutation(n_rows)[: min(n_out, n_rows)]
    return jnp.asarray(index, jnp.int32)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "dtype,with_scales",
    [(jnp.float8_e4m3fn, True), (jnp.bfloat16, False)],
    ids=["fp8_with_scales", "bf16_without"],
)
def test_gather_rows_and_scales_is_rows_at_index(tpu7x, dtype, with_scales):
    _, n_subcores = gather.subcore_mesh()
    n_rows, lane_blocks, ring = 300, 8, 2
    # Four blocks per subcore over a two-slot ring: every slot is reused.
    n_out = n_subcores * BLOCK_ROWS * 4
    rows = _rows(n_rows, lane_blocks, dtype, seed=1)
    scales = (
        jnp.asarray(np.random.default_rng(2).uniform(0.1, 2.0, n_rows), jnp.float32)
        if with_scales
        else None
    )
    index = _gather_index(n_out, n_rows, seed=3)
    got_rows, got_scales = gather.gather_rows_and_scales(
        rows, scales, index, block_rows=BLOCK_ROWS, ring=ring
    )
    idx = np.asarray(index)
    np.testing.assert_array_equal(_as_bits(got_rows), _as_bits(rows)[idx])
    if with_scales:
        assert scales is not None and got_scales is not None
        np.testing.assert_array_equal(np.asarray(got_scales), np.asarray(scales)[idx])
    else:
        assert got_scales is None


@pytest.mark.tpu_test
@pytest.mark.parametrize("live_blocks", [0, 1, 3, 4])
def test_gather_rows_prefix_writes_the_live_prefix(tpu7x, live_blocks):
    """run_live rows per subcore (0, part of the run, the whole run): the
    first n_subcores * run_live output rows are rows[index]."""
    _, n_subcores = gather.subcore_mesh()
    n_rows, lane_blocks, ring, run_blocks = 300, 8, 2, 4
    n_out = n_subcores * BLOCK_ROWS * run_blocks
    rows = _rows(n_rows, lane_blocks, jnp.float8_e4m3fn, seed=4)
    index = _gather_index(n_out, n_rows, seed=5)
    run_live = live_blocks * BLOCK_ROWS
    got = gather.gather_rows_prefix(
        rows,
        index,
        jnp.int32(run_live),
        block_rows=BLOCK_ROWS,
        ring=ring,
    )
    live = n_subcores * run_live
    np.testing.assert_array_equal(
        _as_bits(got)[:live], _as_bits(rows)[np.asarray(index)[:live]]
    )


# (shards, top_k, tokens per shard, message tokens, pair words per shard,
#  scale planes): message tokens past the shard's tokens and pair words past
#  message tokens x top_k are padding that routes nowhere. With 128-pair
#  chunks over a two-chunk ring, "run_is_chunk" is one chunk per subcore
#  (a subcore's whole run of 32 pairs) and the other two are three chunks
#  per subcore, so the ring's slots are reused.
SCATTER_CASES = {
    "run_is_chunk": (2, 4, 80, 96, 512, 1),
    "ring_reuse": (2, 4, 1000, 1024, 6144, 1),
    "two_planes": (4, 2, 600, 640, 3072, 2),
}


def _scatter_case(name, seed=6):
    shards, top_k, tokens, message_tokens, pairs, n_scales = SCATTER_CASES[name]
    rng = np.random.default_rng(seed)
    plane_words = -(-tokens // LANES) * LANES
    scale_offset = pairs
    message_words = scale_offset + n_scales * plane_words
    messages = rng.integers(-(2**31), 2**31 - 1, shards * message_words, np.int64)
    messages = messages.astype(np.int32)
    real = [
        p
        for p in range(shards * pairs)
        if (p % pairs) < message_tokens * top_k
        and (p % pairs) % message_tokens < tokens
    ]
    routed_rows = len(real) // 2 + 37
    landing = rng.permutation(real)[: routed_rows - 37]
    routed_row = np.full(shards * pairs, routed_rows, np.int32)  # the drop row
    foreign = rng.permutation(real)[: len(real) // 4]
    routed_row[foreign] = routed_rows + rng.integers(1, 5000, foreign.size)
    routed_row[landing] = rng.permutation(routed_rows)[: landing.size]
    return dict(
        shards=shards,
        top_k=top_k,
        tokens=tokens,
        message_tokens=message_tokens,
        pairs=pairs,
        n_scales=n_scales,
        plane_words=plane_words,
        scale_offset=scale_offset,
        message_words=message_words,
        messages=messages,
        routed_row=routed_row,
        routed_rows=routed_rows,
    )


def _scatter_reference(c):
    """The two tables in numpy: a routed row's token is its pair's shard's
    first token plus the pair's place in its top-k slot, and its scale words
    are that token's words in its shard's routing message, one per plane."""
    routed_rows, n_scales = c["routed_rows"], c["n_scales"]
    token_table = np.zeros(routed_rows, np.int32)
    scale_table = np.zeros(n_scales * routed_rows, np.int32)
    for p, row in enumerate(c["routed_row"]):
        if row >= routed_rows:
            continue
        shard, local = divmod(p, c["pairs"])
        token = local % c["message_tokens"]
        token_table[row] = shard * c["tokens"] + token
        for plane in range(n_scales):
            word = (
                shard * c["message_words"]
                + c["scale_offset"]
                + plane * c["plane_words"]
                + token
            )
            scale_table[plane * routed_rows + row] = c["messages"][word]
    return token_table, scale_table


@pytest.mark.tpu_test
@pytest.mark.parametrize("name", list(SCATTER_CASES))
def test_scatter_row_tables_matches_the_reference(tpu7x, name):
    c = _scatter_case(name)
    want_tokens, want_scales = _scatter_reference(c)
    got_tokens, got_scales = row_tables_kernel.scatter_row_tables(
        jnp.asarray(c["routed_row"]),
        jnp.asarray(c["messages"]),
        jnp.zeros(c["routed_rows"], jnp.int32),
        jnp.zeros(c["n_scales"] * c["routed_rows"], jnp.int32),
        top_k=c["top_k"],
        tokens_per_shard=c["tokens"],
        message_words=c["message_words"],
        scale_offset=c["scale_offset"],
        n_scales=c["n_scales"],
        plane_words=c["plane_words"],
        pairs_per_shard=c["pairs"],
        message_tokens=c["message_tokens"],
        chunk_max=LANES,
        ring=2,
    )
    np.testing.assert_array_equal(np.asarray(got_tokens), want_tokens)
    np.testing.assert_array_equal(np.asarray(got_scales), want_scales)
