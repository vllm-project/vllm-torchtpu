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
"""In-place MSA cache row writes (interpret mode) against the XLA scatter."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack
from vllm_torchtpu.models.minimax_m3.kernels import msa

PAGE = 128
NUM_PAGES = 12
DROPPED = 2**30


def _batch(q_lens, ctx_lens, rng, pad_tokens):
    """Pages and positions of every token of a ragged batch, as msa_forward
    computes them; padding tokens are dropped."""
    perm = rng.permutation(np.arange(1, NUM_PAGES))
    pages, pos, used = [], [], 0
    for q, c in zip(q_lens, ctx_lens):
        n = -(-(q + c) // PAGE)
        table = perm[used : used + n]
        used += n
        p = np.arange(c, c + q)
        pages.append(table[p // PAGE])
        pos.append(p)
    pages = np.concatenate(pages + [np.full(pad_tokens, DROPPED)]).astype(np.int32)
    pos = np.concatenate(pos + [np.zeros(pad_tokens, np.int64)]).astype(np.int32)
    return jnp.asarray(pages), jnp.asarray(pos)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),  # decode: mid-page, page start, last slot
        ([300], [70], 0),  # prefill from mid-page across three pages
        ([1, 200, 3], [127, 0, 125], 4),  # mixed, a run ending at a page edge
        ([], [], 3),  # nothing to write
    ],
)
def test_matches_xla_scatter(q_lens, ctx_lens, pad_tokens, width, dtype):
    rng = np.random.default_rng(sum(q_lens) + width)
    num_tokens = sum(q_lens) + pad_tokens
    cache = pack(
        rng.standard_normal((NUM_PAGES, PAGE, width)).astype(np.float32), dtype
    )
    rows = jnp.asarray(rng.standard_normal((num_tokens, width)) * 300).astype(
        jnp.bfloat16
    )
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)

    expected = msa.write_rows_xla(cache, rows, pages, pos)
    got = msa.write_rows(
        cache, rows, pages, pos, max_num_seqs=max(len(q_lens), 1), interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
    )


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lowers_for_tpu(dtype):
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, width = 256, 256
    args = (
        jax.ShapeDtypeStruct(
            (4096, PAGE // PACKING[dtype], PACKING[dtype], width), dtype
        ),
        jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(msa.write_rows, max_num_seqs=16))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    text = lowered.as_text()
    assert "tpu_custom_call" in text
    assert "minimax_m3_msa_write_rows" in text


def test_fp4_rows_land_in_their_nibbles():
    """bits=4 on a uint8 4-bit cache: runs across and within pages, the rest
    of every page (other slots' nibbles) unchanged."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_write

    rng = np.random.default_rng(3)
    num_pages, page, width = 6, 128, 256
    before = jnp.asarray(rng.integers(0, 16, (num_pages, page, width)), jnp.uint8)
    cache = msa_fp4.pack_rows(before)
    # Request A: slots 120..135 over pages 2 -> 4; request B: 5 tokens in page 1; one dropped token.
    pages = np.array([2] * 8 + [4] * 8 + [1] * 5 + [99], np.int32)
    slots = np.array(list(range(120, 128)) + list(range(8)) + list(range(40, 45)) + [0], np.int32)
    rows = jnp.asarray(rng.integers(0, 16, (len(pages), width)), jnp.uint8)
    out = msa_write.write_rows(
        cache, rows, jnp.asarray(pages), jnp.asarray(slots),
        max_runs=msa_write.max_write_runs(len(pages), 3, page), bits=4, interpret=True,
    )
    expected = np.asarray(before).copy()
    for t, (p, s) in enumerate(zip(pages, slots)):
        if p < num_pages:
            expected[p, s] = np.asarray(rows[t])
    np.testing.assert_array_equal(msa_fp4.unpack_rows(out), expected)


def test_fp4_folded_scales_land_at_their_lanes():
    """Folded 4-bit pages with L scales per slot (K/V of G heads: L = 2G):
    scale j of slot s lands at lane j * S + s of the last word row; other
    lanes and every nibble outside the written slots stay as they were."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_write

    rng = np.random.default_rng(7)
    num_pages, page, g = 4, 128, 2
    width = 2 * page * g
    before = jnp.asarray(rng.integers(0, 16, (num_pages, page, width)), jnp.uint8)
    lanes_before = rng.standard_normal((num_pages, width)).astype(np.float32)
    cache = msa_fp4.pack_rows(before, scales=jnp.asarray(lanes_before))
    # Slots 124..131 over pages 1 -> 3, 3 tokens in page 0, one dropped token.
    pages = np.array([1] * 4 + [3] * 4 + [0] * 3 + [99], np.int32)
    slots = np.array([124, 125, 126, 127, 0, 1, 2, 3, 60, 61, 62, 0], np.int32)
    rows = jnp.asarray(rng.integers(0, 16, (len(pages), width)), jnp.uint8)
    scales = rng.standard_normal((len(pages), 2 * g)).astype(np.float32)
    out = msa_write.write_rows(
        cache, rows, jnp.asarray(pages), jnp.asarray(slots),
        max_runs=msa_write.max_write_runs(len(pages), 3, page), bits=4,
        scales=jnp.asarray(scales), interpret=True,
    )
    expected = np.asarray(before).copy()
    lanes = lanes_before.copy()
    for t, (p, s) in enumerate(zip(pages, slots)):
        if p < num_pages:
            expected[p, s] = np.asarray(rows[t])
            for j in range(2 * g):
                lanes[p, j * page + s] = scales[t, j]
    np.testing.assert_array_equal(msa_fp4.unpack_rows(out), expected)
    np.testing.assert_array_equal(np.asarray(msa_fp4.folded_scales(out)), lanes)


def _staged_batch(rng, q_lens, ctx_lens, pad_tokens, num_pages, dcp, rank):
    """Pages and slots of a ragged batch as `msa_dcp.owner_pages` makes them:
    a token of a block another DCP rank owns (interleave = page) is dropped,
    so a request's tokens split into runs at every page and ownership edge."""
    perm = rng.permutation(num_pages)
    pages, slots, used = [], [], 0
    for q, c in zip(q_lens, ctx_lens):
        pos = np.arange(c, c + q)
        block = pos // PAGE
        n = block[-1] // dcp + 1
        table = perm[used : used + n]
        used += n
        own = block % dcp == rank
        pages.append(np.where(own, table[np.minimum(block // dcp, n - 1)], DROPPED))
        slots.append(pos % PAGE)
    pages = np.concatenate(pages + [np.full(pad_tokens, DROPPED)]).astype(np.int32)
    slots = np.concatenate(slots + [np.zeros(pad_tokens, np.int64)]).astype(np.int32)
    return jnp.asarray(pages), jnp.asarray(slots)


# (q_lens, ctx_lens, pad_tokens, dcp, rank): decode at page start / mid /
# last slot, MTP-verify runs crossing a page edge, prefill chunks from
# mid-page over several pages (a run starting at token 0 with slot > 0, and
# one ending at the last token), DCP2 / DCP8 interleave, empty batches.
_BATCHES = [
    ([1, 1, 1, 1], [0, 5, 127, 128], 3, 1, 0),
    ([8] * 6, [0, 121, 127, 250, 1000, 64], 0, 1, 0),
    ([8] * 6, [0, 121, 127, 250, 1000, 64], 5, 8, 7),
    ([300], [70], 0, 1, 0),
    ([300, 9], [70, 255], 0, 2, 1),
    ([1, 400, 3], [127, 129, 125], 4, 2, 0),
    ([5, 700], [3, 1], 1, 8, 0),
    ([], [], 4, 1, 0),
    ([3], [0], 0, 2, 1),  # every token dropped
]


@pytest.mark.parametrize("batch", _BATCHES)
@pytest.mark.parametrize(
    "layout",
    ["bf16", "fp8", "fp4", "fp4_index", "fp4_kv_g1", "fp4_kv_g4"],
)
def test_matches_staged_writes_bit_exact(batch, layout):
    """`write_rows` (chunks fetched by the kernel) against the staged-gather
    reference `write_rows_staged`, word for word, including the folded
    scale row (scale bits moved as bits: NaN payloads, -0, denormals)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_write

    q_lens, ctx_lens, pad_tokens, dcp, rank = batch
    rng = np.random.default_rng(len(q_lens) * 31 + pad_tokens + dcp + rank)
    num_pages = 40
    pages, slots = _staged_batch(rng, q_lens, ctx_lens, pad_tokens, num_pages, dcp, rank)
    num_tokens = pages.shape[0]
    width, scale_lanes, kw = {
        "bf16": (256, 0, {}),
        "fp8": (256, 0, {}),
        "fp4": (256, 0, {"bits": 4}),
        "fp4_index": (128, 1, {"bits": 4}),
        "fp4_kv_g1": (256, 2, {"bits": 4}),
        "fp4_kv_g4": (1024, 8, {"bits": 4}),
    }[layout]
    if layout in ("bf16", "fp8"):
        dtype = jnp.bfloat16 if layout == "bf16" else jnp.float8_e4m3fn
        cache = pack(
            rng.standard_normal((num_pages, PAGE, width)).astype(np.float32), dtype
        )
        rows = jnp.asarray(rng.standard_normal((num_tokens, width))).astype(dtype)
    else:
        nib = jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, width)), jnp.uint8)
        lanes = None
        if scale_lanes:
            bits = rng.integers(0, 2**32, (num_pages, width), dtype=np.uint64)
            lanes = jnp.asarray(bits.astype(np.uint32).view(np.float32))
        cache = msa_fp4.pack_rows(nib, scales=lanes)
        rows = jnp.asarray(rng.integers(0, 256, (num_tokens, width)), jnp.uint8)
    if scale_lanes:
        special = np.array([0x7FC12345, 0xFFA00001, 0x80000000, 0x00000003], np.uint32)
        bits = rng.integers(0, 2**32, (num_tokens, scale_lanes), dtype=np.uint64)
        bits = bits.astype(np.uint32)
        bits.flat[: min(bits.size, 4)] = special[: min(bits.size, 4)]
        kw["scales"] = jnp.asarray(bits.view(np.float32))
        if scale_lanes == 1:
            kw["scales"] = kw["scales"][:, 0]
    kw["max_runs"] = msa_write.max_write_runs(num_tokens, max(len(q_lens), 1), PAGE)

    got = msa_write.write_rows(cache, rows, pages, slots, interpret=True, **kw)
    expected = msa_write.write_rows_staged(cache, rows, pages, slots, interpret=True, **kw)
    as_bytes = lambda x: np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))
    np.testing.assert_array_equal(as_bytes(got), as_bytes(expected))
    if (np.asarray(pages) < num_pages).any():
        assert (as_bytes(got) != as_bytes(cache)).any()


def test_more_runs_than_max_runs_match_staged():
    """Runs past `max_runs` are dropped by both paths alike."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_write

    rng = np.random.default_rng(11)
    num_pages, width = 12, 128
    cache = msa_fp4.pack_rows(
        jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, width)), jnp.uint8),
        scales=jnp.asarray(rng.standard_normal((num_pages, width)), jnp.float32),
    )
    pages = jnp.asarray(rng.permutation(num_pages)[:10], jnp.int32)
    slots = jnp.asarray(rng.integers(0, PAGE, 10), jnp.int32)
    rows = jnp.asarray(rng.integers(0, 16, (10, width)), jnp.uint8)
    kw = dict(max_runs=4, bits=4, scales=jnp.asarray(rng.standard_normal(10), jnp.float32))
    got = msa_write.write_rows(cache, rows, pages, slots, interpret=True, **kw)
    expected = msa_write.write_rows_staged(cache, rows, pages, slots, interpret=True, **kw)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("width,scale_lanes", [(128, 1), (1024, 8), (256, 0)])
@pytest.mark.parametrize("tokens", [256, 8192])
def test_fp4_lowers_for_tpu(width, scale_lanes, tokens):
    """4-bit writes (folded index / K/V pages, unfolded) lower for TPU."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_write

    rows_per_page = PAGE // 8 + (1 if scale_lanes else 0)
    args = [
        jax.ShapeDtypeStruct((4096, rows_per_page, 4, width), jnp.uint8),
        jax.ShapeDtypeStruct((tokens, width), jnp.uint8),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    ]
    if scale_lanes:
        args.append(jax.ShapeDtypeStruct((tokens, scale_lanes), jnp.float32))
    max_runs = msa_write.max_write_runs(tokens, 64, PAGE)

    def fn(cache, rows, pages, slots, scales=None):
        return msa_write.write_rows(
            cache, rows, pages, slots, max_runs=max_runs, bits=4, scales=scales
        )

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        text = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "minimax_m3_msa_write_rows" in text
