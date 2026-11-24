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

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import kda_a2a
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)

CHIPS = 16
CROSS_HALF_LINKS = {
    (1, 2),
    (5, 6),
    (9, 10),
    (13, 14),
}


def _fake_devices():
    return [
        SimpleNamespace(
            id=((x * 2 + y) * 4 + z) * 2 + c, coords=(x, y, z), core_on_chip=c
        )
        for x in range(2)
        for y in range(2)
        for z in range(4)
        for c in range(2)
    ]


def _chip_coords(chip):
    xy, z = divmod(chip, 4)
    return xy >> 1, xy & 1, z


def _neighbours_on_non_wrapped_2x2x4(a, b):
    return (
        sum(abs(p - q) for p, q in zip(_chip_coords(a), _chip_coords(b), strict=True))
        == 1
    )


def _undirected_edges(cycle):
    return {
        tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
        for i in range(len(cycle))
    }


def _max_cross_half_share(cycles, column_blocks):
    load = dict.fromkeys(CROSS_HALF_LINKS, 0)
    for cycle, blocks in zip(cycles, column_blocks, strict=True):
        for edge in _undirected_edges(cycle) & CROSS_HALF_LINKS:
            load[edge] += blocks
    return max(load.values()) / sum(column_blocks)


@pytest.mark.parametrize("cycle", kda_a2a.KDA_RING_CYCLES)
def test_should_be_hamiltonian_on_physical_links_when_using_kda_ring_cycles(cycle):
    assert sorted(cycle) == list(range(CHIPS))
    for a, b in itertools.pairwise((*cycle, cycle[0])):
        assert _neighbours_on_non_wrapped_2x2x4(a, b), (a, b)


def test_should_route_each_family_through_its_kda_cycle_when_building_tables():
    devices = _fake_devices()
    token_ranks = tuple((r * 7 + 3) % 32 for r in range(32))

    tables = kda_a2a.build_kda_a2a_tables(devices, token_ranks, kda_a2a.KDA_RING_CYCLES)

    for family, cycle in enumerate(kda_a2a.KDA_RING_CYCLES):
        expected_peers = [chip * 2 + core for chip in cycle for core in (0, 1)]
        assert tables[family, 1].tolist() == expected_peers
        assert [int(tables[family, 0, p]) for p in expected_peers] == list(range(32))
        assert tables[family, 2].tolist() == list(token_ranks)
    for peer, owner in enumerate(token_ranks):
        assert int(tables[3, 0, owner]) == peer


def test_should_keep_production_cycles_when_tables_built_without_cycles():
    devices = _fake_devices()

    default = kda_a2a.build_kda_a2a_tables(devices, tuple(range(32)))
    explicit = kda_a2a.build_kda_a2a_tables(devices, tuple(range(32)), pm.CYCLES)

    np.testing.assert_array_equal(default, explicit)
    assert default[0, 1].tolist()[:4] == [0, 1, 8, 9]


def test_should_reject_cycle_that_misses_a_chip_when_building_ring_tables():
    broken = (pm.CYCLES[0], pm.CYCLES[1], (*pm.CYCLES[2][:-1], 0))

    with pytest.raises(ValueError, match="Hamiltonian"):
        pm.ring_tables(_fake_devices(), tuple(range(32)), broken)


def test_should_split_columns_by_blocks_when_column_blocks_given():
    streams = pm._half_ring_streams(7168, kda_a2a.KDA_RING_COLUMN_BLOCKS)  # noqa: SLF001 -- stream layout is the contract under test

    assert [(f, d, c, w) for f, d, c, w in streams] == [
        (0, 1, 0, 2176),
        (0, -1, 0, 2176),
        (1, 1, 2176, 2304),
        (1, -1, 2176, 2304),
        (2, 1, 4480, 2688),
        (2, -1, 4480, 2688),
    ]


def test_should_split_columns_in_thirds_when_column_blocks_omitted():
    streams = pm._half_ring_streams(7168)  # noqa: SLF001 -- stream layout is the contract under test

    assert [w for _, d, _, w in streams if d == 1] == [2432, 2432, 2304]


@pytest.mark.parametrize("column_blocks", [(17, 18, 20), (17, 18, 21, 0), (0, 35, 21)])
def test_should_reject_column_blocks_when_they_do_not_split_the_width(column_blocks):
    with pytest.raises(ValueError, match="column_blocks"):
        pm._half_ring_streams(7168, column_blocks)  # noqa: SLF001 -- validation is the contract under test


def test_should_balance_cross_half_links_when_using_kda_ring_set():
    prod = _max_cross_half_share(pm.CYCLES, (19, 19, 18))
    ring_set = _max_cross_half_share(
        kda_a2a.KDA_RING_CYCLES, kda_a2a.KDA_RING_COLUMN_BLOCKS
    )

    assert prod == pytest.approx(38 / 56)
    assert ring_set < prod
