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


def test_refuses_a_width_that_does_not_split():
    with pytest.raises(ValueError, match="lanes"):
        ar.small_all_reduce(jnp.zeros((16, 4000), jnp.bfloat16), num_cores=N, axis=AXIS)


def test_form_by_bytes():
    assert ar.form_for(4, 4096, 2) == "oneshot"
    assert ar.form_for(5, 4096, 2) == "twoshot"
    assert ar.form_for(8, 2048, 2) == "oneshot"
