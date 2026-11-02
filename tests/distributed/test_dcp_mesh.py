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
"""The DCP-side accessors and the full-world `(lane, DCP)` Pallas mesh.

The failure mode these guard is silent. A mesh built over the wrong group,
or with its two axes transposed, still builds and still runs -- it just
collects over the wrong partners. So what is pinned here is the plumbing:
which shape comes out, which axis is last, and which rank layouts are
refused outright.

No hardware: `get_pallas_mesh` is faked out, because the question is what it
gets called with.
"""

from unittest.mock import Mock

import jax
import pytest
import torch

from vllm_torchtpu.distributed import dcp, mesh_utils, pcp


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_LAYOUT_CACHE", {})
    monkeypatch.setattr(dcp, "_MESH_CACHE", {})


class _FakeGroup:
    def __init__(self, ranks, rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group


def _dcp_mesh_environment(monkeypatch, group, world_size, rank):
    """Everything `get_or_create_dcp_mesh` reads, except Torchtpu itself."""
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: rank)
    # A mesh ordered by device id would disagree with the rank numbering
    # get_pallas_mesh uses, so the point is that this is never consulted.
    monkeypatch.setattr(
        jax,
        "devices",
        Mock(side_effect=AssertionError("the DCP mesh must not read jax.devices()")),
    )
    from torch_tpu._internal import pallas as pallas_module

    get_pallas_mesh = Mock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(pallas_module, "get_pallas_mesh", get_pallas_mesh)
    return get_pallas_mesh


@pytest.mark.tpu_test
@pytest.mark.parametrize("lanes,dcp_size", [(8, 2), (1, 16), (4, 4), (16, 1)])
def test_mesh_spans_the_world_with_dcp_as_the_last_axis(monkeypatch, lanes, dcp_size):
    """DCP is the fastest-varying factor, so it is the trailing mesh axis.

    Transposing the two axes also produces a valid world-sized mesh, and the
    kernels then gather across TP lanes instead of DCP partners.
    """
    world_size = lanes * dcp_size
    for rank in range(world_size):
        lane, dcp_rank = divmod(rank, dcp_size)
        group = _FakeGroup(
            range(lane * dcp_size, (lane + 1) * dcp_size), rank_in_group=dcp_rank
        )
        native = _dcp_mesh_environment(monkeypatch, group, world_size, rank)

        mesh = dcp.get_or_create_dcp_mesh()

        native.assert_called_once_with(
            axis_names=("dcp_lane", "dcp"), mesh_shape=(lanes, dcp_size)
        )
        assert dcp.get_or_create_dcp_mesh() is mesh
        assert native.call_count == 1
        dcp._MESH_CACHE.clear()


@pytest.mark.tpu_test
def test_an_explicit_axis_name_wins_over_the_default(monkeypatch):
    """The default is a default, not a constant folded into the call."""
    native = _dcp_mesh_environment(monkeypatch, _FakeGroup([0, 1]), 16, 0)

    dcp.get_or_create_dcp_mesh(axis_name="something_else")

    assert native.call_args.kwargs["axis_names"] == ("dcp_lane", "something_else")


@pytest.mark.tpu_test
def test_mesh_rejects_a_strided_dcp_group(monkeypatch):
    """A PCP-style strided group would make the DCP axis the leading one.

    Ranks (0, 4) in an 8-rank world means DCP is the *slowest*-varying factor,
    which `(lane, dcp)` does not describe.
    """
    native = _dcp_mesh_environment(monkeypatch, _FakeGroup([0, 4]), 8, 0)

    with pytest.raises(RuntimeError, match="contiguous runs of ranks"):
        dcp.get_or_create_dcp_mesh()
    native.assert_not_called()
    assert dcp._MESH_CACHE == {}


@pytest.mark.tpu_test
def test_mesh_rejects_a_group_position_that_disagrees_with_the_rank(monkeypatch):
    """rank_in_group has to be `global_rank % dcp_size`, not merely in range."""
    group = _FakeGroup([0, 1], rank_in_group=0)
    native = _dcp_mesh_environment(monkeypatch, group, 8, rank=1)

    with pytest.raises(RuntimeError, match="at position 1, got"):
        dcp.get_or_create_dcp_mesh()
    native.assert_not_called()


@pytest.mark.tpu_test
def test_mesh_rejects_a_dcp_size_that_does_not_divide_the_world(monkeypatch):
    native = _dcp_mesh_environment(monkeypatch, _FakeGroup([0, 1, 2]), 8, 0)

    with pytest.raises(NotImplementedError, match="must divide"):
        dcp.get_or_create_dcp_mesh()
    native.assert_not_called()


def test_layout_describes_the_group_it_is_given(monkeypatch):
    """The shared layout helper must follow its argument, not the PCP group.

    Without this, a DCP-only run -- where the PCP group is trivial -- would get
    a world_size=1 layout and build a single-device mesh.
    """
    pcp_group = _FakeGroup([0])
    dcp_group = _FakeGroup([0, 1, 2, 3], rank_in_group=2)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(
        mesh_utils,
        "_collect_rank_to_device_id",
        lambda group, *_: {r: 10 + r for r in group.ranks},
    )
    monkeypatch.setattr(mesh_utils, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(mesh_utils, "_get_tpu_global_device_id", lambda: 12)

    # The no-argument call is the pre-existing PCP contract, and it still
    # holds -- extracting the shared helper did not change pcp.py's surface.
    assert pcp.get_pcp_group_layout().world_size == 1

    layout = dcp.get_dcp_group_layout()
    assert layout.world_size == 4
    assert layout.rank_in_group == 2
    assert layout.device_ids == (10, 11, 12, 13)
    assert layout.device_id == 12


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_the_plain_accessors_survive_an_uninitialized_group(monkeypatch, world_size):
    """Rank and world size have to answer before distributed init, too.

    `get_dcp_group` returns None then, and callers read these two without
    checking -- so the defaults are load-bearing, not defensive.
    """
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    assert dcp.get_dcp_rank() == 0
    assert dcp.get_dcp_world_size() == 1

    group = _FakeGroup(list(range(world_size)), rank_in_group=world_size - 1)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    assert dcp.get_dcp_rank() == world_size - 1


def test_an_uninitialized_group_still_yields_a_two_axis_mesh(monkeypatch):
    """The pre-init fallback must not change the mesh's rank.

    Callers index the specs by axis name, so a one-axis fallback would fail
    somewhere else entirely.
    """
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    mesh = dcp.get_or_create_dcp_mesh()
    assert mesh.axis_names == ("dcp_lane", "dcp")
    assert mesh.size == 1
