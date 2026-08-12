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

from vllm_torchtpu.distributed import pcp_rank_order as order_mod


def _config(**kwargs):
    parallel = SimpleNamespace(
        prefill_context_parallel_size=kwargs.pop("pcp", 8),
        tensor_parallel_size=kwargs.pop("tp", 1),
        data_parallel_size=kwargs.pop("dp", 1),
        nnodes=kwargs.pop("nnodes", 1),
        pipeline_parallel_size=kwargs.pop("pp", 1),
    )
    return SimpleNamespace(
        parallel_config=parallel,
        kv_transfer_config=kwargs.pop("kv_transfer_config", None),
    )


# --- expected_groups / mesh_group_orders / match_groups --------------------


def test_expected_groups_mirrors_vllms_arange_layout():
    """tp varies fastest, then pcp: the layout parallel_state builds."""
    got = order_mod.expected_groups(1, 1, 4, 2)

    assert sorted(sorted(g) for g in got["tp"]) == [[0, 1], [2, 3], [4, 5],
                                                    [6, 7]]
    assert sorted(sorted(g) for g in got["pcp"]) == [[0, 2, 4, 6],
                                                     [1, 3, 5, 7]]


def test_mesh_group_orders_slices_both_axes():
    """mesh[d,:,t] is a PCP ring; mesh[d,p,:] is a TP lane."""
    import numpy as np

    mesh = np.array([[[0, 1], [6, 7], [4, 5], [2, 3]]])

    got = order_mod.mesh_group_orders(mesh, 1, 4, 2)

    assert got["pcp"] == [[0, 6, 4, 2], [1, 7, 5, 3]]
    assert got["tp"] == [[0, 1], [6, 7], [4, 5], [2, 3]]


def test_match_groups_pairs_rows_by_membership():
    groups = [[0, 2, 4, 6], [1, 3, 5, 7]]
    rows = [[1, 7, 5, 3], [0, 6, 4, 2]]

    assert order_mod.match_groups(groups, rows) == [[0, 6, 4, 2], [1, 7, 5, 3]]


def test_match_groups_rejects_a_different_split():
    groups = [[0, 2, 4, 6], [1, 3, 5, 7]]

    assert order_mod.match_groups(groups, [[0, 1, 2, 3], [4, 5, 6, 7]]) is None


# --- resolve_pcp_topology_order -------------------------------------------


def test_no_order_when_pcp_is_disabled():
    assert order_mod.resolve_pcp_topology_order(_config(pcp=1)) is None


def test_no_order_when_a_kv_transfer_config_is_present():
    """Scoped off for P/D.

    The reshard planner derives the prefill rank as pcp_rank*tp_size+tp_rank
    and Raiden names work units from the same composite, so reordering would
    silently mis-address transfers rather than fail.
    """
    cfg = _config(kv_transfer_config=SimpleNamespace())

    assert order_mod.resolve_pcp_topology_order(cfg) is None


def test_no_order_on_multihost(monkeypatch):
    monkeypatch.setattr(order_mod.envs, "TPU_MULTIHOST_BACKEND", "ray")

    assert order_mod.resolve_pcp_topology_order(_config()) is None


def test_no_order_when_multiple_nodes(monkeypatch):
    """A ring crossing hosts would need multi-slice ordering over DCN."""
    monkeypatch.setattr(order_mod.envs, "TPU_MULTIHOST_BACKEND", "")

    assert order_mod.resolve_pcp_topology_order(_config(nnodes=2)) is None


def test_abandons_both_axes_when_the_mesh_regroups(monkeypatch):
    """A mesh that splits ranks differently must not be applied at all.

    vLLM builds TP before PCP, so accepting one axis and rejecting the other
    would leave TP on the arange layout while PCP follows the mesh -- the
    split that desynchronises TP partners.
    """
    import numpy as np
    import torch

    # tp lanes paired across chips instead of within one
    mesh = np.array([[[0, 2], [6, 4], [1, 3], [7, 5]]])
    monkeypatch.setattr(torch.tpu,
                        "topology_aware_mesh",
                        lambda _shape: mesh,
                        raising=False)
    monkeypatch.setattr(order_mod.envs, "TPU_MULTIHOST_BACKEND", "")

    assert order_mod.resolve_pcp_topology_order(_config(pcp=4, tp=2)) is None


def test_returns_both_axes_for_a_matching_mesh(monkeypatch):
    import numpy as np
    import torch

    mesh = np.array([[[0, 1], [6, 7], [4, 5], [2, 3]]])
    monkeypatch.setattr(torch.tpu,
                        "topology_aware_mesh",
                        lambda _shape: mesh,
                        raising=False)
    monkeypatch.setattr(order_mod.envs, "TPU_MULTIHOST_BACKEND", "")

    got = order_mod.resolve_pcp_topology_order(_config(pcp=4, tp=2))

    assert got == {
        "pcp": [[0, 6, 4, 2], [1, 7, 5, 3]],
        "tp": [[0, 1], [6, 7], [4, 5], [2, 3]],
    }


def test_falls_back_to_none_when_the_mesh_call_raises(monkeypatch):
    """A failed mesh call costs a few hops; it must not fail startup."""
    import torch

    def _boom(_shape):
        raise RuntimeError("no PjRtClient")

    monkeypatch.setattr(torch.tpu, "topology_aware_mesh", _boom, raising=False)
    monkeypatch.setattr(order_mod.envs, "TPU_MULTIHOST_BACKEND", "")

    assert order_mod.resolve_pcp_topology_order(_config()) is None


def test_rejects_a_result_that_is_not_a_permutation(monkeypatch):
    """Applying a partial mapping would place two ranks on one ring slot."""
    import torch

    class _Fake:

        def flatten(self):
            return self

        def tolist(self):
            return [0, 1, 2, 3, 4, 5, 6, 6]

    monkeypatch.setattr(torch.tpu,
                        "topology_aware_mesh",
                        lambda _shape: _Fake(),
                        raising=False)
    monkeypatch.setattr(order_mod.envs, "TPU_MULTIHOST_BACKEND", "")

    assert order_mod.resolve_pcp_topology_order(_config()) is None


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


def test_context_manager_restores_the_original_function():
    """The patch must not outlive group construction."""
    from vllm.distributed import parallel_state

    before = parallel_state.init_model_parallel_group
    with order_mod.pcp_topology_order({"pcp": [[0, 1, 2, 3, 6, 7, 4, 5]]}):
        assert parallel_state.init_model_parallel_group is not before
    assert parallel_state.init_model_parallel_group is before


def test_context_manager_is_inert_without_orders():
    from vllm.distributed import parallel_state

    before = parallel_state.init_model_parallel_group
    with order_mod.pcp_topology_order(None):
        assert parallel_state.init_model_parallel_group is before
