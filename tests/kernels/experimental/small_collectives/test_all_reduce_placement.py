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
"""The placement-aware all-reduce on the 8 cores of one host: every form
against the float64 sum, bitwise equal on every core, and a skipped DMA that
the oracle has to catch; the peer tables on a fixed 2x2x1 layout."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_reduce_placement as arp,
)
from vllm_torchtpu.layers.adapter.small_all_reduce_op import _resolve_placement_spec

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

# form: (d2d, ordered)
FORMS = {
    "flat_mesh": (False, False),
    "flat_ordered": (False, True),
    "d2d_mesh": (True, False),
    "d2d_ordered": (True, True),
}

# tpu7x 2x2x1 in jax.devices() order: core i is core_on_chip i % 2 of chip i // 2
LAYOUT = [arp.Core(((i // 2) % 2, i // 4, 0), i % 2) for i in range(8)]


def _all_reduce(mesh, per_core, d2d, ordered, drop=()):
    cores = arp.cores_of(mesh.devices.reshape(-1))
    meta = arp.placement_meta(cores, d2d=d2d, ordered=ordered, drop=drop)

    def body(x, m):
        return arp.small_all_reduce_placement(
            x, m, num_cores=N, axis=AXIS, d2d=d2d, drops=bool(drop)
        )

    return run_sharded(mesh, body, per_core, meta.reshape(-1))


def _check(got, per_core):
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for core in range(N):
        ok = within(got[core], want, tolerance)
        assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"
        assert np.array_equal(got[core].view(np.uint16), got[0].view(np.uint16)), (
            f"core {core} differs from core 0"
        )


@pytest.mark.parametrize("rows", [6, 8, 16, 48])
@pytest.mark.parametrize("form", list(FORMS))
def test_matches_the_exact_sum(mesh, form, rows):
    per_core = blocks((N, rows, 4096), seed=rows)
    _check(_all_reduce(mesh, per_core, *FORMS[form]), per_core)


@pytest.mark.parametrize("hidden", [2048, 6144])
@pytest.mark.parametrize("form", list(FORMS))
def test_other_widths(mesh, form, hidden):
    per_core = blocks((N, 16, hidden), seed=hidden)
    _check(_all_reduce(mesh, per_core, *FORMS[form]), per_core)


@pytest.mark.parametrize("d2d", [False, True])
def test_a_skipped_dma_fails_the_oracle(mesh, d2d):
    """Positive control: one DMA not sent (and not waited for) leaves its slot
    holding the previous call's data, and the oracle has to see it."""
    cores = arp.cores_of(mesh.devices.reshape(-1))
    if d2d:
        _, owners = arp.d2d_layout(cores)
        drop = {(owners[1], owners[0])}  # owner 1's phase-1 chunk to owner 0
    else:
        drop = {(0, int(arp.placement_meta(cores, d2d=False, ordered=True)[0, 1]))}
    first = blocks((N, 16, 4096), seed=21)
    _check(_all_reduce(mesh, first, d2d, True), first)
    second = blocks((N, 16, 4096), seed=22)
    got = _all_reduce(mesh, second, d2d, True, drop=drop)
    want, tolerance = exact_sum(second), sum_tolerance(second)
    assert any(not within(got[c], want, tolerance).all() for c in range(N))


@pytest.mark.parametrize("rows", [6, 8, 16, 48])
@pytest.mark.parametrize("reverse", [False, True])
def test_compiled_order_is_bitwise_the_two_shot(mesh, reverse, rows):
    """The issue order changes nothing in the sums: bitwise all_reduce's
    two-shot on every core, and within the oracle."""
    from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar

    cores = arp.cores_of(mesh.devices.reshape(-1))
    pats = arp.ordered_patterns(cores, reverse=reverse)
    per_core = blocks((N, rows, 4096), seed=400 + rows)
    got = run_sharded(
        mesh,
        lambda x, r: arp.small_all_reduce_ordered(
            x, r, num_cores=N, axis=AXIS, patterns=pats
        ),
        per_core,
        range(N),
    )
    plain = run_sharded(
        mesh,
        lambda x, r: ar.small_all_reduce(
            x, num_cores=N, axis=AXIS, rank=r, form="twoshot"
        ),
        per_core,
        range(N),
    )
    _check(got, per_core)
    assert np.array_equal(got.view(np.uint16), plain.view(np.uint16))


def _direct(mesh, per_core, ordered, exact, skip_phase0=()):
    cores = arp.cores_of(mesh.devices.reshape(-1))
    meta = arp.d2d_direct_meta(cores, ordered=ordered, skip_phase0=skip_phase0)

    def body(x, m):
        return arp.small_all_reduce_d2d_direct(
            x, m, num_cores=N, axis=AXIS, ordered=ordered, exact_partials=exact
        )

    return run_sharded(mesh, body, per_core, meta.reshape(-1))


def _bf16_partials_tolerance(per_core):
    """oracle.sum_tolerance plus the direct form's rounding of each chip
    partial to bf16: half a bf16 step at |x_die0 + x_die1| per chip (cores 2c
    and 2c + 1 share chip c here), carried into the final rounding step."""
    terms = per_core.astype(np.float64)
    order_error = (N - 1) * 2.0**-24 * np.abs(terms).sum(axis=0)
    extra = sum(0.5 * bf16_ulp(np.abs(terms[i] + terms[i + 1])) for i in range(0, N, 2))
    return (
        bf16_ulp(np.abs(terms.sum(axis=0)) + order_error + extra) + order_error + extra
    )


def _direct_static(mesh, per_core, exact, skip_phase0=()):
    """The direct form as benchmarked: the rank alone as data, the order
    compiled in from ``d2d_direct_patterns``."""
    cores = arp.cores_of(mesh.devices.reshape(-1))
    pats = arp.d2d_direct_patterns(arp.d2d_direct_meta(cores, ordered=True))

    def body(x, r):
        return arp.small_all_reduce_d2d_direct(
            x,
            r,
            num_cores=N,
            axis=AXIS,
            ordered=True,
            exact_partials=exact,
            patterns=pats,
            skip_phase0=skip_phase0,
        )

    return run_sharded(mesh, body, per_core, range(N))


def _check_bf16_partials(got, per_core):
    want, tolerance = exact_sum(per_core), _bf16_partials_tolerance(per_core)
    for core in range(N):
        assert within(got[core], want, tolerance).all(), f"core {core}"
        assert np.array_equal(got[core].view(np.uint16), got[0].view(np.uint16))


@pytest.mark.parametrize("rows", [6, 16, 48])
@pytest.mark.parametrize("ordered", [False, True])
def test_direct_d2d_with_exact_partials_matches_the_exact_sum(mesh, ordered, rows):
    per_core = blocks((N, rows, 4096), seed=200 + rows)
    _check(_direct(mesh, per_core, ordered, exact=True), per_core)


@pytest.mark.parametrize("rows", [6, 16, 48])
@pytest.mark.parametrize("ordered", [False, True])
def test_direct_d2d_with_bf16_partials_is_within_one_more_rounding(mesh, ordered, rows):
    per_core = blocks((N, rows, 4096), seed=300 + rows)
    _check_bf16_partials(_direct(mesh, per_core, ordered, exact=False), per_core)


@pytest.mark.parametrize("rows", [6, 8, 16, 48])
@pytest.mark.parametrize("exact", [False, True])
def test_direct_d2d_compiled_order(mesh, exact, rows):
    """The benchmarked path: float32 partials within the production
    tolerance, bf16 partials within one more rounding per chip partial."""
    per_core = blocks((N, rows, 4096), seed=500 + rows)
    got = _direct_static(mesh, per_core, exact)
    if exact:
        _check(got, per_core)
    else:
        _check_bf16_partials(got, per_core)


@pytest.mark.parametrize("exact", [False, True])
def test_direct_d2d_compiled_order_on_wide_range_inputs(mesh, exact):
    """Inputs spread over 2^-12 .. 2^12 per element, where float32 sums in a
    different order can differ: still within the oracle."""
    rng = np.random.default_rng(7)
    per_core = (
        blocks((N, 16, 4096), seed=600).astype(np.float32)
        * np.exp2(rng.integers(-12, 13, size=(N, 16, 4096))).astype(np.float32)
    ).astype(jnp.bfloat16)
    got = _direct_static(mesh, per_core, exact)
    if exact:
        _check(got, per_core)
    else:
        _check_bf16_partials(got, per_core)


@pytest.mark.parametrize("exact", [False, True])
def test_direct_d2d_compiled_order_skipped_phase0_fails_the_oracle(mesh, exact):
    first = blocks((N, 16, 4096), seed=41)
    _direct_static(mesh, first, exact)
    second = blocks((N, 16, 4096), seed=42)
    got = _direct_static(mesh, second, exact, skip_phase0=(0,))
    want = exact_sum(second)
    tolerance = sum_tolerance(second) if exact else _bf16_partials_tolerance(second)
    assert not within(got[0], want, tolerance).all()


def test_direct_d2d_skipped_phase0_fails_the_oracle(mesh):
    first = blocks((N, 16, 4096), seed=31)
    _direct(mesh, first, True, exact=True)
    second = blocks((N, 16, 4096), seed=32)
    got = _direct(mesh, second, True, exact=True, skip_phase0={0})
    want, tolerance = exact_sum(second), sum_tolerance(second)
    assert not within(got[0], want, tolerance).all()


def test_issue_order_is_slowest_class_first():
    lat = arp.LATENCY_US
    for me in range(8):
        order = arp.issue_order(LAYOUT, me, [j for j in range(8) if j != me], lat)
        assert sorted(order) == [j for j in range(8) if j != me]
        costs = [arp._latency(LAYOUT, me, j, lat) for j in order]
        assert costs == sorted(costs, reverse=True)
        assert order[-1] == me ^ 1  # the twin, over the intra-chip link, last
    # core 1 (die 1 of chip (0, 0)): the far chip's die 0, then its die 1 (both
    # leave on the x link), then the near chips' die 0 and die 1, each class
    # alternating y and x links, then the twin
    assert arp.issue_order(LAYOUT, 1, [0, 2, 3, 4, 5, 6, 7], lat) == [
        6,
        7,
        4,
        2,
        5,
        3,
        0,
    ]


def test_ordered_patterns_reproduce_every_rank_order():
    p0, p1 = arp.ordered_patterns(LAYOUT)
    for r in range(8):
        pat = p1 if r & 1 else p0
        want = arp.issue_order(
            LAYOUT, r, [j for j in range(8) if j != r], arp.LATENCY_US
        )
        assert [r ^ k for k in pat] == want


def test_d2d_layout_and_meta():
    twin, owners = arp.d2d_layout(LAYOUT)
    assert twin == [1, 0, 3, 2, 5, 4, 7, 6]
    assert owners == [1, 3, 5, 7]
    meta = arp.placement_meta(LAYOUT, d2d=True, ordered=False, drop={(0, 1)})
    assert meta.shape == (8, arp.meta_width(8, True))
    assert list(meta[0, :5]) == [0, 0, 1, 0, 0]  # rank, die 0, twin, slot, no phase 0
    assert list(meta[1, :5]) == [1, 1, 0, 0, 0]
    flat = arp.placement_meta(LAYOUT, d2d=False, ordered=False, drop={(2, 5)})
    assert -1 in flat[2, 1:8] and flat[5, 8 + 2] == 0


def test_d2d_refuses_a_mesh_without_both_cores_of_a_chip():
    with pytest.raises(ValueError, match="both cores"):
        arp.d2d_layout(LAYOUT[:4] + [arp.Core((1, 1, 0), 0)] * 4)


pytestmark = [pytest.mark.multichip, pytest.mark.cpu_test]


def test_resolve_placement_spec():
    cores = LAYOUT
    assert _resolve_placement_spec(cores, "") is None
    assert _resolve_placement_spec(cores, "0") is None
    assert _resolve_placement_spec(cores, "off") is None

    spec = _resolve_placement_spec(cores, "1")
    assert spec is not None
    assert spec[0] == "d2d_direct"
    assert spec[1] is True

    spec = _resolve_placement_spec(cores, "f32")
    assert spec is not None
    assert spec[0] == "d2d_direct"
    assert spec[1] is True

    spec = _resolve_placement_spec(cores, "bf16")
    assert spec is not None
    assert spec[0] == "d2d_direct"
    assert spec[1] is False

    spec = _resolve_placement_spec(cores, "ordered")
    assert spec is not None
    assert spec[0] == "ordered"
    assert spec[1] is False

    # 2-core fallback
    assert _resolve_placement_spec(cores[:2], "1") is None
    assert _resolve_placement_spec(cores[:3], "1") is None

    with pytest.raises(
        ValueError, match="Unknown TPU_SMALL_ALL_REDUCE_PLACEMENT mode: invalid_mode"
    ):
        _resolve_placement_spec(cores, "invalid_mode")
