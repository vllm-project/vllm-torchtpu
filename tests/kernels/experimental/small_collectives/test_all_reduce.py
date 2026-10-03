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
"""The small all-reduce kernel on the 8 cores of one host (and 32 cores of
2x2x4), against the float64 sum and ``jax.lax.psum``, with the core's index
read from the axis and handed in as data (the torch bridge's way)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

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


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < N:
        pytest.skip(f"needs at least {N} JAX devices")
    devices = np.array(jax.devices()[:N])
    if jax.default_backend() == "tpu":
        yield Mesh(devices, (AXIS,))
    else:
        with pltpu.force_tpu_interpret_mode():
            yield Mesh(devices, (AXIS,))


def _cossim_and_sqnr_db(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    g = got.astype(np.float64).ravel()
    w = want.astype(np.float64).ravel()
    denom = np.linalg.norm(g) * np.linalg.norm(w)
    cossim = 1.0 if denom == 0.0 else float(np.dot(g, w) / denom)
    noise = np.sum((g - w) ** 2)
    sig = np.sum(w**2)
    sqnr_db = float("inf") if noise == 0.0 else float(10.0 * np.log10(sig / noise))
    return cossim, sqnr_db


def _all_reduce(mesh, per_core, ranks, *, rank_operand=True, form=None):
    n = per_core.shape[0]

    def body(x, r):
        return ar.small_all_reduce(
            x, num_cores=n, axis=AXIS, rank=r if rank_operand else None, form=form
        )

    return run_sharded(mesh, body, per_core, ranks)


def _check(got, per_core):
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for core in range(per_core.shape[0]):
        ok = within(got[core], want, tolerance)
        assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"
        cossim, sqnr_db = _cossim_and_sqnr_db(got[core], want)
        assert cossim >= 0.99999
        assert sqnr_db >= 45.0


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 12, 16, 32, 64, 96])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_the_exact_sum(mesh, rows, rank_operand):
    per_core = blocks((N, rows, 4096), seed=rows)
    _check(_all_reduce(mesh, per_core, range(N), rank_operand=rank_operand), per_core)


@pytest.mark.parametrize("form", ["oneshot", "twoshot", "hierarchical"])
@pytest.mark.parametrize("rows", [4, 16])
def test_all_forms_at_both_sizes(mesh, form, rows):
    per_core = blocks((N, rows, 4096), seed=100 + rows)
    _check(_all_reduce(mesh, per_core, range(N), form=form), per_core)


@pytest.mark.parametrize("hidden", [64, 128, 256, 512, 2048, 4000, 6144, 7168])
@pytest.mark.parametrize("form", ["oneshot", "twoshot", "hierarchical"])
def test_other_widths_and_small_payloads(mesh, hidden, form):
    """Verify H=6144, small payloads (1 KiB..64 KiB), and unaligned widths
    like H=64 and H=4000 across oneshot, twoshot, and hierarchical."""
    per_core = blocks((N, 4, hidden), seed=hidden)
    _check(_all_reduce(mesh, per_core, range(N), form=form), per_core)


def test_h6144_on_32_cores_2x2x4():
    """Verify H=6144 on 32 cores (2x2x4) where 6144 % 128 == 0 even though
    6144 % (32 * 128) != 0, for oneshot, twoshot, and hierarchical."""
    x = jnp.zeros((4, 6144), jnp.bfloat16)
    r = jnp.array([0], jnp.int32)
    for form in ("oneshot", "twoshot", "hierarchical"):
        jaxpr = jax.make_jaxpr(
            lambda a, rk: ar.small_all_reduce(
                a, num_cores=32, axis=AXIS, rank=rk, form=form
            )
        )(x, r)
        assert jaxpr.out_avals[0].shape == (4, 6144)
        assert jaxpr.out_avals[0].dtype == jnp.bfloat16

    if jax.device_count() >= 32:
        devices32 = np.array(jax.devices()[:32])
        per_core = blocks((32, 2, 6144), seed=6144)
        if jax.default_backend() == "tpu":
            mesh32 = Mesh(devices32, (AXIS,))
            for form in ("oneshot", "twoshot", "hierarchical"):
                got = _all_reduce(mesh32, per_core, range(32), form=form)
                _check(got, per_core)
        else:
            with pltpu.force_tpu_interpret_mode():
                mesh32 = Mesh(devices32, (AXIS,))
                for form in ("oneshot", "twoshot", "hierarchical"):
                    got = _all_reduce(mesh32, per_core, range(32), form=form)
                    _check(got, per_core)



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


def test_refuses_a_width_that_does_not_split_when_padding_disabled():
    with pytest.raises(ValueError, match="lanes"):
        ar.small_all_reduce(
            jnp.zeros((16, 4000), jnp.bfloat16),
            num_cores=N,
            axis=AXIS,
            allow_col_padding=False,
        )
    with pytest.raises(ValueError, match="lanes"):
        ar._twoshot(
            jnp.zeros((16, 6144), jnp.bfloat16), 32, AXIS, None, ar.COLLECTIVE_ID
        )


def test_form_by_bytes():
    assert ar.form_for(4, 4096, 2) == "oneshot"
    assert ar.form_for(5, 4096, 2) == "twoshot"
    assert ar.form_for(8, 2048, 2) == "oneshot"
