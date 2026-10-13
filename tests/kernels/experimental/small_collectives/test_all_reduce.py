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
"""The small all-reduce kernel on the 8 cores of one host, against the float64
sum and ``jax.lax.psum``, with the core's index read from the axis and handed
in as data (the torch bridge's way)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar

from .oracle import (
    AXIS,
    N,
    bf16_ulp,
    blocks,
    exact_sum,
    run_sharded,
    sum_tolerance,
    within,
)

pytestmark = pytest.mark.multichip


def _all_reduce(mesh, per_core, ranks, *, rank_operand=True, form=None):
    def body(x, r):
        return ar.small_all_reduce(
            x, num_cores=N, axis=AXIS, rank=r if rank_operand else None, form=form
        )

    return run_sharded(mesh, body, per_core, ranks)


def _check(got, per_core):
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for core in range(N):
        ok = within(got[core], want, tolerance)
        assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 12, 16, 32, 64, 96])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_the_exact_sum(mesh, rows, rank_operand):
    per_core = blocks((N, rows, 4096), seed=rows)
    _check(_all_reduce(mesh, per_core, range(N), rank_operand=rank_operand), per_core)


@pytest.mark.parametrize("form", ["oneshot", "twoshot"])
@pytest.mark.parametrize("rows", [4, 16])
def test_both_forms_at_both_sizes(mesh, form, rows):
    per_core = blocks((N, rows, 4096), seed=100 + rows)
    _check(_all_reduce(mesh, per_core, range(N), form=form), per_core)


@pytest.mark.parametrize("hidden", [2048, 6144, 7168])
def test_other_widths(mesh, hidden):
    per_core = blocks((N, 16, hidden), seed=hidden)
    _check(_all_reduce(mesh, per_core, range(N)), per_core)


def test_agrees_with_psum_within_its_rounding(mesh):
    """XLA's bf16 psum may round every partial sum, (n - 1) roundings of at
    most half a step at the sum of magnitudes; the kernel rounds once."""
    per_core = blocks((N, 16, 4096), seed=7)
    psum = run_sharded(
        mesh, lambda x, r: jax.lax.psum(x, AXIS), per_core, range(N)
    ).astype(np.float64)
    got = _all_reduce(mesh, per_core, range(N)).astype(np.float64)
    magnitudes = np.abs(per_core.astype(np.float64)).sum(axis=0)
    tolerance = (N - 1) / 2 * bf16_ulp(magnitudes) + 2 * bf16_ulp(np.abs(psum[0]))
    for core in range(N):
        assert within(got[core], psum[core], tolerance).all()


def test_a_wrong_rank_order_fails_the_oracle(mesh):
    """Positive control: cores 0 and 1 told each other's index write into the
    wrong slots, and the oracle has to see it."""
    per_core = blocks((N, 16, 4096), seed=11)
    swapped = [1, 0, *range(2, N)]
    got = _all_reduce(mesh, per_core, swapped)
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for core in (0, 1):
        assert not within(got[core], want, tolerance).all()


# --- valid rows (fold path) ---------------------------------------------------

TILE_ROWS = 16


def _valid_rows(mesh, per_core, n_valid):
    meta = np.stack([np.full(N, n_valid), np.arange(N)], 1).astype(np.int32)

    def body(x, m):
        return ar.small_all_reduce_valid_rows(
            x,
            m.reshape(2)[:1],
            m.reshape(2)[1:],
            num_cores=N,
            axis=AXIS,
            tile_rows=TILE_ROWS,
        )

    return run_sharded(mesh, body, per_core, meta)


def _plain(mesh, per_core):
    return _all_reduce(mesh, per_core, range(N), form="twoshot")


@pytest.mark.parametrize("hidden", [4096, 6144])
@pytest.mark.parametrize("rows", [16, 32])
def test_valid_rows_are_bitwise_the_plain_two_shot(mesh, rows, hidden):
    """Every n_valid on both sides of fold_rows: rows < n_valid bitwise equal
    to the plain two-shot (and within the oracle), rows >= n_valid +0 on the
    fold path and the plain sum on the full path."""
    _, fold_rows = ar.fold_geometry(rows, hidden, N, TILE_ROWS)
    per_core = blocks((N, rows, hidden), seed=rows + hidden)
    plain = _plain(mesh, per_core)
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for n_valid in sorted({0, 1, fold_rows, fold_rows + 1, rows}):
        got = _valid_rows(mesh, per_core, n_valid)
        for core in range(N):
            g, p = got[core], plain[core]
            assert np.array_equal(
                g[:n_valid].view(np.uint16), p[:n_valid].view(np.uint16)
            ), f"n_valid={n_valid} core {core}: valid rows differ from the two-shot"
            assert within(g[:n_valid], want[:n_valid], tolerance[:n_valid]).all()
            if n_valid <= fold_rows:
                assert not g[n_valid:].view(np.uint16).any(), "padding is not +0"
            else:
                assert np.array_equal(g.view(np.uint16), p.view(np.uint16))


def test_an_understated_valid_count_fails_the_oracle(mesh):
    """Positive control: one valid row reported as padding comes out +0 and
    the oracle has to see it."""
    rows, hidden = 16, 4096
    _, fold_rows = ar.fold_geometry(rows, hidden, N, TILE_ROWS)
    per_core = blocks((N, rows, hidden), seed=5)
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    got = _valid_rows(mesh, per_core, fold_rows - 1)
    assert not within(got[0][:fold_rows], want[:fold_rows], tolerance[:fold_rows]).all()


def test_fold_geometry():
    assert ar.fold_geometry(16, 6144, 16, 16) == (3, 5)  # GLM-5.2 on 16 cores
    assert ar.fold_geometry(16, 6144, 8, 16) == (6, 2)
    assert ar.fold_geometry(8, 6144, 16, 16) is None  # fewer rows than a tile
    assert ar.fold_geometry(16, 2048, 16, 16) is None  # one lane group per chunk


def test_refuses_a_width_that_does_not_split():
    with pytest.raises(ValueError, match="lanes"):
        ar.small_all_reduce(jnp.zeros((16, 4000), jnp.bfloat16), num_cores=N, axis=AXIS)


def test_form_by_bytes():
    assert ar.form_for(4, 4096, 2) == "oneshot"
    assert ar.form_for(5, 4096, 2) == "twoshot"
    assert ar.form_for(8, 2048, 2) == "oneshot"
