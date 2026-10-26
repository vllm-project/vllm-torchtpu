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
sum and ``jax.lax.psum``: over the whole axis, and over subgroups of it named
by a peer table (a TP group inside the expert-parallel mesh)."""

import jax
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

# Two interleaved groups of four, members listed out of mesh order.
SUBGROUPS = ((6, 0, 2, 4), (3, 5, 7, 1))


def _tables(groups):
    """Per core: its position in its group, and the group's mesh indices."""
    rank = np.zeros((N, 1), np.int32)
    peers = np.zeros((N, len(groups[0])), np.int32)
    for group in groups:
        for position, core in enumerate(group):
            rank[core], peers[core] = position, group
    return rank, peers


def _grouped(mesh, per_core, groups, *, form=None, rank=None, peers=None):
    n = len(groups[0])
    table_rank, table_peers = _tables(groups)
    rank = table_rank if rank is None else rank
    peers = table_peers if peers is None else peers

    def body(x, r, p):
        return ar.small_all_reduce(
            x, num_cores=n, axis=AXIS, rank=r, peers=p, form=form
        )

    return run_sharded(mesh, body, per_core, rank, peers)


def _check(got, per_core, groups):
    for group in groups:
        want = exact_sum(per_core[list(group)])
        tolerance = sum_tolerance(per_core[list(group)])
        for core in group:
            ok = within(got[core], want, tolerance)
            assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"
            assert np.array_equal(
                got[core].view(np.uint16), got[group[0]].view(np.uint16)
            ), f"core {core} differs from its group"


@pytest.mark.parametrize("rows", [1, 3, 8, 16, 32, 64])
def test_whole_axis_from_the_axis_index(mesh, rows):
    per_core = blocks((N, rows, 7168), seed=rows)
    got = run_sharded(
        mesh,
        lambda x: ar.small_all_reduce(x, num_cores=N, axis=AXIS),
        per_core,
    )
    _check(got, per_core, (tuple(range(N)),))


@pytest.mark.parametrize("rows", [1, 8, 16, 64])
@pytest.mark.parametrize("form", ["oneshot", "twoshot"])
def test_whole_axis_through_a_peer_table(mesh, rows, form):
    per_core = blocks((N, rows, 7168), seed=10 + rows)
    groups = ((0, 1, 4, 5, 6, 7, 2, 3),)
    _check(_grouped(mesh, per_core, groups, form=form), per_core, groups)


@pytest.mark.parametrize("rows", [1, 8, 16, 64])
@pytest.mark.parametrize("form", ["oneshot", "twoshot"])
def test_subgroups_sum_only_their_members(mesh, rows, form):
    per_core = blocks((N, rows, 7168), seed=20 + rows)
    _check(_grouped(mesh, per_core, SUBGROUPS, form=form), per_core, SUBGROUPS)


@pytest.mark.parametrize("hidden", [2048, 4096, 7168])
def test_other_widths(mesh, hidden):
    per_core = blocks((N, 16, hidden), seed=hidden)
    _check(_grouped(mesh, per_core, SUBGROUPS), per_core, SUBGROUPS)


def test_agrees_with_grouped_psum_within_its_rounding(mesh):
    """XLA's bf16 psum may round every partial sum, (n - 1) roundings of at
    most half a step at the sum of magnitudes; the kernel rounds once."""
    per_core = blocks((N, 16, 7168), seed=7)
    groups = [list(g) for g in SUBGROUPS]
    psum = run_sharded(
        mesh, lambda x: jax.lax.psum(x, AXIS, axis_index_groups=groups), per_core
    ).astype(np.float64)
    got = _grouped(mesh, per_core, SUBGROUPS).astype(np.float64)
    for group in SUBGROUPS:
        magnitudes = np.abs(per_core[list(group)].astype(np.float64)).sum(axis=0)
        n = len(group)
        for core in group:
            tolerance = (n - 1) / 2 * bf16_ulp(magnitudes) + 2 * bf16_ulp(
                np.abs(psum[core])
            )
            assert within(got[core], psum[core], tolerance).all()


def test_a_wrong_peer_table_fails_the_oracle(mesh):
    """Positive control: the two groups' tables name each other's first
    member, so the first members trade groups: every count still matches (no
    hang), and the column owner sums the wrong blocks, which the oracle has
    to see."""
    per_core = blocks((N, 16, 7168), seed=11)
    rank, peers = _tables(SUBGROUPS)
    peers[list(SUBGROUPS[0])] = (3, 0, 2, 4)
    peers[list(SUBGROUPS[1])] = (6, 5, 7, 1)
    got = _grouped(mesh, per_core, SUBGROUPS, rank=rank, peers=peers)
    group = list(SUBGROUPS[0])
    want, tolerance = exact_sum(per_core[group]), sum_tolerance(per_core[group])
    assert not within(got[0], want, tolerance).all()


def test_refuses_a_width_that_does_not_split():
    x = jax.numpy.zeros((8, 7168 + 128), jax.numpy.bfloat16)
    with pytest.raises(ValueError):
        ar.small_all_reduce(x, num_cores=8, axis=AXIS)


@pytest.mark.parametrize("rows,kernel", [(16, True), (32, True), (48, False)])
def test_the_serving_body_switches_to_xla_above_the_cap(mesh, rows, kernel):
    """small_all_reduce_op's shard_map body: the kernel's bits up to the cap,
    the grouped psum's above it."""
    from vllm_torchtpu.layers.adapter.small_all_reduce_op import group_all_reduce

    per_core = blocks((N, rows, 7168), seed=30 + rows)
    rank, peers = _tables(SUBGROUPS)
    got = run_sharded(
        mesh, group_all_reduce(SUBGROUPS, 32, AXIS), per_core, rank, peers
    )
    if kernel:
        want = _grouped(mesh, per_core, SUBGROUPS)
    else:
        groups = [list(g) for g in SUBGROUPS]
        want = run_sharded(
            mesh, lambda x: jax.lax.psum(x, AXIS, axis_index_groups=groups), per_core
        )
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))


# The served DP4 x TP4 layout on one host: group positions (0, 1) and (2, 3)
# are the two cores of one chip.
CHIP_PAIRS = ((0, 1, 4, 5), (6, 7, 2, 3))


def _check_two_roundings(got, per_core, groups):
    """The pairs form rounds the pair sums and then the total: held to XLA
    psum's own bound ((n - 1) / 2 steps at the sum of magnitudes, one at the
    result), and every member of a group bit-identical."""
    for group in groups:
        terms = per_core[list(group)].astype(np.float64)
        want = exact_sum(per_core[list(group)])
        magnitudes = np.abs(terms).sum(axis=0)
        tolerance = (len(group) - 1) / 2 * bf16_ulp(magnitudes) + bf16_ulp(np.abs(want))
        for core in group:
            ok = within(got[core], want, tolerance)
            assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"
            assert np.array_equal(
                got[core].view(np.uint16), got[group[0]].view(np.uint16)
            ), f"core {core} differs from its group"


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("groups", [CHIP_PAIRS, SUBGROUPS], ids=["chip-pairs", "interleaved"])
@pytest.mark.parametrize("skip", [False, True])
def test_pairs_form(mesh, rows, chunks, groups, skip):
    per_core = blocks((N, rows, 7168), seed=40 + rows)
    table_rank, table_peers = _tables(groups)

    def body(x, r, p):
        return ar.small_all_reduce(
            x, num_cores=4, axis=AXIS, rank=r, peers=p, form="pairs",
            chunks=chunks, skip_device_barrier=skip,
        )

    got = run_sharded(mesh, body, per_core, table_rank, table_peers)
    _check_two_roundings(got, per_core, groups)


def test_pairs_form_oracle_sees_a_missing_member(mesh):
    """Positive control: the pairs output checked against inputs where one
    member's block is zeroed must fail the two-roundings oracle."""
    per_core = blocks((N, 16, 7168), seed=12)
    rank, peers = _tables(CHIP_PAIRS)

    def body(x, r, p):
        return ar.small_all_reduce(x, num_cores=4, axis=AXIS, rank=r, peers=p, form="pairs")

    got = run_sharded(mesh, body, per_core, rank, peers)
    _check_two_roundings(got, per_core, CHIP_PAIRS)
    wrong = per_core.copy()
    wrong[5] = 0
    with pytest.raises(AssertionError):
        _check_two_roundings(got, wrong, CHIP_PAIRS)


def test_skip_device_barrier_is_bit_identical(mesh):
    per_core = blocks((N, 32, 7168), seed=13)
    a = _grouped(mesh, per_core, CHIP_PAIRS)
    rank, peers = _tables(CHIP_PAIRS)

    def body(x, r, p):
        return ar.small_all_reduce(
            x, num_cores=4, axis=AXIS, rank=r, peers=p, skip_device_barrier=True
        )

    b = run_sharded(mesh, body, per_core, rank, peers)
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


def test_group_form_reads_the_layout(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_SMALL_ALL_REDUCE_PAIRS", 2, raising=False)
    assert ar.group_form(CHIP_PAIRS)[:2] == ("pairs", 2)
    assert ar.group_form(SUBGROUPS)[0] is None
    assert ar.group_form(((0, 1, 2, 3, 4, 5, 6, 7),))[0] is None
    monkeypatch.setattr(envs, "TPU_SMALL_ALL_REDUCE_PAIRS", 0, raising=False)
    assert ar.group_form(CHIP_PAIRS)[0] is None
