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
"""`QSASparseAttnBlock` row body vs the pure-JAX oracle (CPU interpret).

The block's per-row attention (`attend_row`: micro-page DMAs from the paged FP8
pool through a 2-slot ring, online softmax, stock gate rounding) runs in a
small Pallas harness. Rows cover empty, partial-group, exact-tile and multi-tile
lengths. Positive controls: the other head pair of the packed row, and a
dropped last tile, must both be caught.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.qsa_attn.block import (
    MICRO,
    TILE,
    attend_row,
    reference_rows,
)

NH, DIM, BUDGET, PAGES = 3, 256, 512, 24
SM = DIM**-0.5
SK, SV = 0.5, 0.25
# empty, 1 token, partial micro-page, one exact tile, 2 tiles + partial, long
LENGTHS = [0, 1, 7, TILE, 2 * TILE + 3, 1500]


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    pool = jnp.asarray(
        rng.standard_normal((PAGES, 128, 1, 4, DIM)) * 2, jnp.float32
    ).astype(jnp.float8_e4m3fn)
    n = len(LENGTHS)
    pages = rng.integers(0, PAGES * 128 // MICRO, size=(n, BUDGET + 1)).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((n, NH, DIM)), jnp.bfloat16)
    gate = jnp.asarray(rng.standard_normal((n, NH, DIM)), jnp.bfloat16)
    return pool, jnp.asarray(pages), jnp.asarray(LENGTHS, jnp.int32), q, gate


def _kernel_rows(pool, pages, kv_len, q, gate, *, off, tile_cut=0):
    n = q.shape[0]

    def kernel(pages_ref, len_ref, q_ref, g_ref, pool_ref, out_ref, ring, sems):
        def row(r, carry):
            out_ref[r] = attend_row(
                r,
                pool=pool_ref,
                ring=ring,
                sems=sems,
                pages=pages_ref,
                kv_len=len_ref,
                q=q_ref[r],
                gate=g_ref[r],
                off=off,
                scale_k=SK,
                scale_v=SV,
                sm_scale=SM,
                interpret=True,
            )
            return carry

        jax.lax.fori_loop(0, n, row, 0)

    lens = jnp.maximum(kv_len - tile_cut, jnp.minimum(kv_len, 1)) if tile_cut else kv_len
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((2, TILE, 1, 4, DIM), jnp.float8_e4m3fn),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((n, NH, DIM), jnp.float32),
        interpret=True,
    )(pages, lens, q, gate, pool)


@pytest.mark.parametrize("off", [0, 1])
def test_attend_row_matches_reference(off):
    pool, pages, kv_len, q, gate = _inputs(3 + off)
    got = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=off))
    want = np.asarray(
        reference_rows(q, gate, pool, pages, kv_len, off=off, scale_k=SK, scale_v=SV, sm_scale=SM)
    )
    assert np.all(got[0] == 0)  # empty row
    assert float(np.abs(want).mean()) > 1e-2
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)
    assert float(np.abs(got - want).mean()) < 2e-3


def test_attend_row_catches_wrong_pair_and_dropped_tile():
    pool, pages, kv_len, q, gate = _inputs(11)
    want = np.asarray(
        reference_rows(q, gate, pool, pages, kv_len, off=0, scale_k=SK, scale_v=SV, sm_scale=SM)
    )
    wrong_pair = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=1))
    dropped = np.asarray(_kernel_rows(pool, pages, kv_len, q, gate, off=0, tile_cut=TILE))
    for bad in (wrong_pair, dropped):
        assert not np.allclose(bad[1:], want[1:], atol=2e-2, rtol=2e-2)
