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
