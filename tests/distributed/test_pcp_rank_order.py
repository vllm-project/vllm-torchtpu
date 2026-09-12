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
"""Tests for topology-aware ordering of the PCP group."""

from types import SimpleNamespace

import numpy
import pytest

from vllm_torchtpu.distributed import pcp_rank_order as order_mod

pytestmark = pytest.mark.cpu_test


def _config(**kwargs):
    parallel = SimpleNamespace(
        prefill_context_parallel_size=kwargs.pop("pcp", 8),
        tensor_parallel_size=kwargs.pop("tp", 1),
        data_parallel_size=kwargs.pop("dp", 1),
        nnodes=kwargs.pop("nnodes", 1),
        pipeline_parallel_size=kwargs.pop("pp", 1),
    )
    return SimpleNamespace(parallel_config=parallel)


# --- resolve_pcp_topology_order: not applicable -> None --------------------


def test_no_order_when_pcp_is_disabled():
    assert order_mod.resolve_pcp_topology_order(_config(pcp=1)) is None


def test_no_order_when_multiple_nodes():
    """A ring crossing hosts needs multi-slice ordering over DCN."""
    assert order_mod.resolve_pcp_topology_order(_config(nnodes=2)) is None


# --- resolve_pcp_topology_order: faults raise ------------------------------


def test_raises_when_pipeline_parallelism_is_present():
    """PcpStaticSupportValidator rejects this before workers spawn."""
    with pytest.raises(RuntimeError, match="pipeline_parallel_size"):
        order_mod.resolve_pcp_topology_order(_config(pp=2))


def test_raises_when_data_parallel_replicas_are_present():
    """dp>1 here means EP, whose replicas never share a torch world."""
    with pytest.raises(RuntimeError, match="data_parallel_size"):
        order_mod.resolve_pcp_topology_order(_config(dp=2))


def test_propagates_the_error_when_the_mesh_call_fails(monkeypatch):
    """A missing API or dead process group must not be papered over."""
    import torch

    def _boom(_shape):
        raise RuntimeError("topology_aware_mesh exploded")

    monkeypatch.setattr(torch,
                        "tpu",
                        SimpleNamespace(topology_aware_mesh=_boom),
                        raising=False)
    with pytest.raises(RuntimeError, match="exploded"):
        order_mod.resolve_pcp_topology_order(_config())


def test_raises_when_the_mesh_is_not_a_permutation(monkeypatch):
    """A duplicated rank would silently drop a worker from the groups."""
    import torch

    mesh = numpy.array([0, 1, 2, 3, 4, 5, 6, 6]).reshape(1, 8, 1)
    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(topology_aware_mesh=lambda _shape: mesh),
        raising=False)
    with pytest.raises(RuntimeError, match="not a permutation"):
        order_mod.resolve_pcp_topology_order(_config())


# --- resolve_pcp_topology_order: the supported path ------------------------


def test_returns_both_axes_from_one_mesh(monkeypatch):
    import torch

    mesh = numpy.array([[[0, 1], [6, 7], [4, 5], [2, 3]]])
    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(topology_aware_mesh=lambda _shape: mesh),
        raising=False)
    got = order_mod.resolve_pcp_topology_order(_config(pcp=4, tp=2))

    assert got == {
        "pcp": [[0, 6, 4, 2], [1, 7, 5, 3]],
        "tp": [[0, 1], [6, 7], [4, 5], [2, 3]],
    }
    assert set(got) == {"pcp", "tp"}, "keys must be vLLM group_name values"

    # The invariant the whole change rests on: a TP lane all-reduces partial
    # attention outputs, which is only meaningful if its members hold the same
    # tokens -- that is, the same pcp position. Slicing both axes from one
    # mesh is what guarantees it.
    pcp_of = {r: i for ring in got["pcp"] for i, r in enumerate(ring)}
    for lane in got["tp"]:
        assert len({pcp_of[r] for r in lane}) == 1


def test_matches_the_order_measured_on_hardware(monkeypatch):
    """pcp=8 tp=1, job j-64748dc1 on a v7 2x2x1 host."""
    import torch

    mesh = numpy.array([0, 1, 6, 7, 4, 5, 2, 3]).reshape(1, 8, 1)
    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(topology_aware_mesh=lambda _shape: mesh),
        raising=False)
    got = order_mod.resolve_pcp_topology_order(_config(pcp=8, tp=1))

    assert got["pcp"] == [[0, 1, 6, 7, 4, 5, 2, 3]]


# --- pcp_topology_order ----------------------------------------------------


def test_context_manager_only_touches_axes_it_was_given():
    """Axes absent from the orders dict pass through untouched."""
    from vllm.distributed import parallel_state

    seen = {}
    original = parallel_state.init_model_parallel_group

    def _record(group_ranks, local_rank, backend, **kwargs):
        seen[kwargs.get("group_name")] = [list(g) for g in group_ranks]
        return None

    parallel_state.init_model_parallel_group = _record
    try:
        with order_mod.pcp_topology_order({"pcp": [[0, 1, 2, 3, 6, 7, 4, 5]]}):
            parallel_state.init_model_parallel_group(
                [[0, 1, 2, 3, 4, 5, 6, 7]], 0, "gloo", group_name="pcp")
            parallel_state.init_model_parallel_group(
                [[0, 1, 2, 3, 4, 5, 6, 7]], 0, "gloo", group_name="tp")
    finally:
        parallel_state.init_model_parallel_group = original

    assert seen["pcp"] == [[0, 1, 2, 3, 6, 7, 4, 5]]
    assert seen["tp"] == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_new_group_preserves_rank_order_only_for_replaced_axes():
    """new_group sorts by default, which would discard the resolved order.

    The wrapper keys off the axis, not the rank list: at tp=1 the replaced tp
    lanes are singletons and so are dcp, pp and dp, so matching on contents
    would catch axes vLLM built for itself.
    """
    import torch
    from vllm.distributed import parallel_state

    calls = []
    original_new_group = torch.distributed.new_group
    original_init = parallel_state.init_model_parallel_group

    # sort_ranks has to be a named parameter: pcp_topology_order inspects the
    # signature and refuses to run against a new_group that cannot preserve
    # the order, so a **kwargs-only double trips that guard instead of
    # exercising the wrapper.
    def _record_new_group(ranks=None, sort_ranks=True, **kwargs):
        calls.append(sort_ranks)
        return None

    def _fake_init(group_ranks, local_rank, backend, *args, **kwargs):
        # Stands in for the real thing, which is what calls new_group.
        for ranks in group_ranks:
            torch.distributed.new_group(ranks)

    orders = {
        "pcp": [[0, 1, 6, 7, 4, 5, 2, 3]],
        "tp": [[0], [1], [6], [7], [4], [5], [2], [3]],
    }
    singletons = [[r] for r in range(8)]
    seen = {}

    torch.distributed.new_group = _record_new_group
    parallel_state.init_model_parallel_group = _fake_init
    try:
        with order_mod.pcp_topology_order(orders):
            patched = parallel_state.init_model_parallel_group
            for axis, built in (("tp", singletons), ("dcp", singletons),
                                ("pcp", [list(range(8))]), ("pp", singletons),
                                ("dp", singletons), ("ep", [list(range(8))])):
                calls.clear()
                patched(built, 0, "gloo", group_name=axis)
                seen[axis] = sorted({c for c in calls}, key=str)
    finally:
        torch.distributed.new_group = original_new_group
        parallel_state.init_model_parallel_group = original_init

    # The replaced axes skip the sort; everything vLLM builds for itself keeps
    # torch's default, so its groups are untouched.
    assert seen["pcp"] == [False], seen
    assert seen["tp"] == [False], seen
    for axis in ("dcp", "pp", "dp", "ep"):
        assert seen[axis] == [True], (axis, seen)

    assert torch.distributed.new_group is original_new_group


def test_new_group_is_not_wrapped_when_there_is_nothing_to_replace():
    import torch

    before = torch.distributed.new_group
    with order_mod.pcp_topology_order(None):
        assert torch.distributed.new_group is before
    assert torch.distributed.new_group is before


def test_raises_when_new_group_cannot_preserve_order():
    import torch

    original_new_group = torch.distributed.new_group
    torch.distributed.new_group = lambda ranks=None, backend=None: None
    try:
        with pytest.raises(RuntimeError, match="no sort_ranks parameter"):
            with order_mod.pcp_topology_order({"pcp": [[0, 1]]}):
                pass
    finally:
        torch.distributed.new_group = original_new_group


def test_context_manager_restores_the_original_function():
    """The patch must not outlive group construction."""
    from vllm.distributed import parallel_state

    before = parallel_state.init_model_parallel_group
    with order_mod.pcp_topology_order({"pcp": [[0, 1, 2, 3, 6, 7, 4, 5]]}):
        assert parallel_state.init_model_parallel_group is not before
    assert parallel_state.init_model_parallel_group is before


def test_context_manager_changes_no_group_without_orders():
    """Without orders the patch still installs, but replaces nothing.

    It is installed so that vLLM's own grouping is still logged, which is
    what a topology run gets diffed against.
    """
    from vllm.distributed import parallel_state

    seen = {}
    before = parallel_state.init_model_parallel_group

    def _record(group_ranks, local_rank, backend, **kwargs):
        seen[kwargs.get("group_name")] = [list(g) for g in group_ranks]
        return None

    parallel_state.init_model_parallel_group = _record
    try:
        with order_mod.pcp_topology_order(None):
            parallel_state.init_model_parallel_group(
                [[0, 1, 2, 3, 4, 5, 6, 7]], 0, "gloo", group_name="pcp")
            parallel_state.init_model_parallel_group([[0, 1], [2, 3]],
                                                     0,
                                                     "gloo",
                                                     group_name="tp")
        # Restored even though nothing was replaced.
        assert parallel_state.init_model_parallel_group is _record
    finally:
        parallel_state.init_model_parallel_group = before

    assert seen["pcp"] == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert seen["tp"] == [[0, 1], [2, 3]]


def test_context_manager_replaces_both_axes():
    """pcp and tp are applied together -- that is what keeps them coherent."""
    from vllm.distributed import parallel_state

    seen = {}
    original = parallel_state.init_model_parallel_group

    def _record(group_ranks, local_rank, backend, **kwargs):
        seen[kwargs.get("group_name")] = [list(g) for g in group_ranks]
        return None

    orders = {
        "pcp": [[0, 6, 4, 2], [1, 7, 5, 3]],
        "tp": [[0, 1], [6, 7], [4, 5], [2, 3]],
    }
    parallel_state.init_model_parallel_group = _record
    try:
        with order_mod.pcp_topology_order(orders):
            parallel_state.init_model_parallel_group(
                [[0, 2, 4, 6], [1, 3, 5, 7]], 0, "gloo", group_name="pcp")
            parallel_state.init_model_parallel_group(
                [[0, 1], [2, 3], [4, 5], [6, 7]], 0, "gloo", group_name="tp")
            parallel_state.init_model_parallel_group([[0]],
                                                     0,
                                                     "gloo",
                                                     group_name="dp")
    finally:
        parallel_state.init_model_parallel_group = original

    assert seen["pcp"] == orders["pcp"]
    assert seen["tp"] == orders["tp"]
    assert seen["dp"] == [[0]]


# --- verify_pcp_topology_order ---------------------------------------------


def _patch_groups(monkeypatch, *, rank, pcp=None, tp=None):
    """Point the verifier at synthetic groups without a live torch world."""
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    for name, ranks in (("get_pcp_group", pcp), ("get_tp_group", tp)):
        monkeypatch.setattr(parallel_state,
                            name,
                            lambda ranks=ranks: SimpleNamespace(ranks=ranks),
                            raising=False)


def test_verify_does_nothing_when_no_order_was_applied(monkeypatch):
    """None and {} both mean the mesh never applied, so there is no promise
    to check. Reading a group back here would fault on configurations that
    opted out."""
    import torch.distributed as dist

    def _explode():
        raise AssertionError("verify must not touch the world when idle")

    monkeypatch.setattr(dist, "get_rank", _explode)

    order_mod.verify_pcp_topology_order(None)
    order_mod.verify_pcp_topology_order({})


def test_verify_accepts_the_order_it_asked_for(monkeypatch):
    _patch_groups(monkeypatch, rank=6, pcp=[0, 1, 6, 7, 4, 5, 2, 3], tp=[6])

    order_mod.verify_pcp_topology_order({
        "pcp": [[0, 1, 6, 7, 4, 5, 2, 3]],
        "tp": [[r] for r in range(8)],
    })


def test_verify_raises_when_the_members_match_but_the_order_does_not(
        monkeypatch):
    """The failure this function exists for.

    A ring is its order, so a group holding the right ranks in the wrong
    sequence is exactly as wrong as one holding the wrong ranks -- and it is
    the shape a set comparison would wave through.
    """
    _patch_groups(monkeypatch, rank=6, pcp=[0, 1, 2, 3, 4, 5, 6, 7])

    with pytest.raises(RuntimeError, match="did not adopt the requested"):
        order_mod.verify_pcp_topology_order(
            {"pcp": [[0, 1, 6, 7, 4, 5, 2, 3]]})


def test_verify_raises_when_the_rank_is_in_no_resolved_group(monkeypatch):
    """A rank missing from the mesh means the layout does not cover the world;
    verifying only the ranks that happen to appear would hide it."""
    _patch_groups(monkeypatch, rank=9, pcp=[0, 1, 6, 7, 4, 5, 2, 3])

    with pytest.raises(RuntimeError, match="does not appear in the pcp"):
        order_mod.verify_pcp_topology_order(
            {"pcp": [[0, 1, 6, 7, 4, 5, 2, 3]]})


def test_verify_raises_for_an_axis_it_cannot_read_back(monkeypatch):
    """Replacing an axis whose group has no accessor would apply an order that
    is never checked, which is the state this module refuses to be in."""
    _patch_groups(monkeypatch, rank=0, pcp=[0, 1])

    with pytest.raises(RuntimeError, match="No group accessor"):
        order_mod.verify_pcp_topology_order({"ep": [[0, 1]]})
