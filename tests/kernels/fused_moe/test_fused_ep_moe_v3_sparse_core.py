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
Each runs here with more blocks or chunks than ring slots (the scatter also
with one chunk per subcore), and its output is compared value for value with
the numpy indexing it stands for.

The programs do not run in the TPU interpreter on CPU (its SparseCore
lowering rejects the index-driven copies), so their output needs a TPU7x
chip. The arguments they refuse, and the order of the scatter's copies and
waits in its traced program, are checked on CPU.
"""

import re

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
@pytest.mark.parametrize("shards", [4, 5], ids=["at_max_pairs", "past_max_pairs"])
def test_scatter_refuses_more_pairs_than_its_float_decode_is_exact_for(
    tpu7x_record, shards
):
    """The sparse cores decode a pair number with float multiplies, exact
    below MAX_PAIRS pairs: a call of exactly MAX_PAIRS pairs is built, one
    more shard's worth is refused."""
    pairs_per_shard = row_tables_kernel.MAX_PAIRS // 4
    operands, kwargs = _scatter_shapes(
        shards * pairs_per_shard, pairs_per_shard, tokens=128, n_scales=1
    )

    def scatter(*args):
        return row_tables_kernel.scatter_row_tables(*args, **kwargs)

    if shards * pairs_per_shard > row_tables_kernel.MAX_PAIRS:
        with pytest.raises(ValueError, match="exact up to"):
            jax.eval_shape(scatter, *operands)
    else:
        jax.eval_shape(scatter, *operands)


@pytest.mark.cpu_test
@pytest.mark.parametrize("divisor", [3, 1000, 1024, 3072, 10239, 81920])
def test_the_float_decode_is_exact_below_max_pairs(divisor):
    """The scatter floors pair // divisor as (pair + FLOOR_NUDGE) times the
    float32 inverse, truncated. That is exact for every pair number below
    MAX_PAIRS, at the divisors the layer uses (pair words and message tokens
    per shard) and at awkward ones."""
    pair = np.arange(row_tables_kernel.MAX_PAIRS, dtype=np.int64)
    inverse = np.float32(1.0 / divisor)
    nudged = pair.astype(np.float32) + np.float32(row_tables_kernel.FLOOR_NUDGE)
    decoded = (nudged * inverse).astype(np.int64)
    bad = np.flatnonzero(decoded != pair // divisor)
    assert not bad.size, (divisor, bad[:3])


@pytest.mark.cpu_test
def test_the_float_decode_first_fails_at_1_28_times_max_pairs():
    """With divisor 1000 the first pair the decode gets wrong is 5,386,999,
    so MAX_PAIRS could not be twice as high."""
    pair = np.arange(5_386_999 + 1, dtype=np.int64)
    inverse = np.float32(1.0 / 1000)
    nudged = pair.astype(np.float32) + np.float32(row_tables_kernel.FLOOR_NUDGE)
    decoded = (nudged * inverse).astype(np.int64)
    bad = np.flatnonzero(decoded != pair // 1000)
    assert bad[0] == 5_386_999


@pytest.mark.cpu_test
def test_scatter_refuses_a_short_chunk_with_several_scale_planes(tpu7x_record):
    """A chunk that is a subcore's whole run (32 pairs here) may be shorter
    than 128 words with one scale plane, but not with two: plane 1 of a ring
    slot would start 32 words in, off the 128-word alignment."""
    operands, kwargs = _scatter_shapes(1024, 512, tokens=128, n_scales=2)
    with pytest.raises(ValueError, match="with one scale plane"):
        row_tables_kernel.scatter_row_tables(*operands, **kwargs)


def _jaxprs(jaxpr):
    """`jaxpr` and every jaxpr nested in its equations, depth first."""
    yield jaxpr
    for eqn in jaxpr.eqns:
        for value in eqn.params.values():
            for inner in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(inner, "jaxpr", inner)
                if hasattr(inner, "eqns"):
                    yield from _jaxprs(inner)


def _base(operand):
    """The ref a DMA operand slices (a TransformedRef's own ref)."""
    return getattr(operand, "ref", operand)


@pytest.mark.cpu_test
@pytest.mark.parametrize("n_scales", [1, 2, 3])
def test_no_scale_plane_is_scattered_before_every_plane_gather_landed(
    tpu7x_record, n_scales
):
    """A ring slot's scale planes are gathered on one shared semaphore, and a
    wait on it returns once one plane's worth of bytes has landed from any
    of the planes' copies. So the scatter may read a plane only after every
    plane's wait: in the traced program, the semaphore of the copies into
    the gathered-scales buffer has no copy outstanding at any copy out of
    that buffer. A chip that happens to land the planes in order gives the
    right tables either way, so the check is on the program."""
    operands, kwargs = _scatter_shapes(
        32 * 4 * LANES, 1024, tokens=1024, n_scales=n_scales
    )
    traced = jax.make_jaxpr(
        lambda *ops: row_tables_kernel.scatter_row_tables(
            *ops, **kwargs, chunk_max=LANES, ring=2
        )
    )(*operands)
    checked = 0
    for jaxpr in _jaxprs(traced.jaxpr):
        dmas = []
        for eqn in jaxpr.eqns:
            if eqn.primitive.name in ("dma_start", "dma_wait"):
                src, dst, sem, _, _ = eqn.params["tree"].unflatten(eqn.invars)
                dmas.append((eqn.primitive.name, _base(src), _base(dst), _base(sem)))
        starts = [d for d in dmas if d[0] == "dma_start"]
        # The gathered-scales buffer: copied into by one start, out of by another.
        staged = {dst for _, _, dst, _ in starts} & {src for _, src, _, _ in starts}
        if not staged:
            continue
        (scales,) = staged
        (gather_sem,) = {sem for _, _, dst, sem in starts if dst is scales}
        outstanding = 0
        for prim, src, dst, sem in dmas:
            if prim == "dma_start" and dst is scales:
                outstanding += 1
            elif prim == "dma_wait" and sem is gather_sem:
                outstanding -= 1
            elif prim == "dma_start" and src is scales:
                assert outstanding == 0, "a scale plane is scattered before it landed"
                checked += 1
    assert checked == n_scales


def _scatter_name(n_scales, **kwargs):
    """The name of the scatter program built for 32 shards of 1024 tokens'
    pairs (a 1024-pair run per subcore) at `n_scales` planes."""
    operands, shape_kwargs = _scatter_shapes(
        32 * 1024, 1024, tokens=1024, n_scales=n_scales
    )
    traced = jax.make_jaxpr(
        lambda *ops: row_tables_kernel.scatter_row_tables(
            *ops, **shape_kwargs, **kwargs
        )
    )(*operands)
    (name,) = set(re.findall(r"fused_ep_row_tables_\w+", str(traced)))
    return name


@pytest.mark.cpu_test
def test_the_scatter_chunk_shrinks_so_its_ring_fits_a_subcore(tpu7x_record):
    """The ring holds ring x chunk words of routed rows and tokens and, per
    scale plane, three more such buffers. At one plane the default chunk is
    the largest of up to 640 pairs that divides the run (512 here); at 32
    planes a 512-pair chunk's ring of 4 would be 0.8 MB, over a TPU7x vector
    subcore's 512 KiB of VMEM, so the chunk shrinks to 256 pairs (0.4 MB).
    An explicit chunk that does not fit is refused."""
    assert _scatter_name(1) == "fused_ep_row_tables_c512x4"
    assert _scatter_name(32) == "fused_ep_row_tables_c256x4_p32"
    with pytest.raises(ValueError, match="bytes of VMEM"):
        _scatter_name(32, chunk=512)


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
    # With no live block the prefix is empty and this compares two empty
    # arrays: that case shows only that the call returns.
    np.testing.assert_array_equal(
        _as_bits(got)[:live], _as_bits(rows)[np.asarray(index)[:live]]
    )


# (shards, top_k, tokens per shard, message tokens, pair words per shard,
#  scale planes, extra token-table rows past the drop row): message tokens
#  past the shard's tokens and pair words past message tokens x top_k are
#  padding that routes nowhere. With 128-pair chunks over a two-chunk ring,
#  "run_is_chunk" is one chunk per subcore (a subcore's whole run of 32
#  pairs) and the others are three chunks per subcore, so the ring's slots
#  are reused. "padded_table" hands a token table longer than its drop row,
#  as the layer does when the bound is not a whole number of 128 rows: the
#  rows from the drop row on must stay zero.
SCATTER_CASES = {
    "run_is_chunk": (2, 4, 80, 96, 512, 1, 0),
    "ring_reuse": (2, 4, 1000, 1024, 6144, 1, 0),
    "two_planes": (4, 2, 600, 640, 3072, 2, 0),
    "padded_table": (2, 4, 1000, 1024, 6144, 1, 96),
}


def _scatter_case(name, seed=6):
    shards, top_k, tokens, message_tokens, pairs, n_scales, extra = SCATTER_CASES[name]
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
        table_rows=routed_rows + extra,
    )


def _scatter_reference(c):
    """The two tables in numpy: a routed row's token is its pair's shard's
    first token plus the pair's place in its top-k slot, and its scale words
    are that token's words in its shard's routing message, one per plane."""
    routed_rows, n_scales, table = c["routed_rows"], c["n_scales"], c["table_rows"]
    token_table = np.zeros(table, np.int32)
    scale_table = np.zeros(n_scales * table, np.int32)
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
            scale_table[plane * table + row] = c["messages"][word]
    return token_table, scale_table


@pytest.mark.tpu_test
@pytest.mark.parametrize("name", list(SCATTER_CASES))
def test_scatter_row_tables_matches_the_reference(tpu7x, name):
    c = _scatter_case(name)
    want_tokens, want_scales = _scatter_reference(c)
    got_tokens, got_scales = row_tables_kernel.scatter_row_tables(
        jnp.asarray(c["routed_row"]),
        jnp.asarray(c["messages"]),
        jnp.zeros(c["table_rows"], jnp.int32),
        jnp.zeros(c["n_scales"] * c["table_rows"], jnp.int32),
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
        routed_rows=c["routed_rows"],
    )
    np.testing.assert_array_equal(np.asarray(got_tokens), want_tokens)
    np.testing.assert_array_equal(np.asarray(got_scales), want_scales)
