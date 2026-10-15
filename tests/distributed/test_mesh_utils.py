# SPDX-License-Identifier: Apache-2.0
"""Which process groups a pipeline worker makes standalone."""

from types import SimpleNamespace

import pytest
import torch
from torch_tpu._internal.distributed import spmd_util
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import mesh_utils

_KINDS = ("tp", "dcp", "pcp", "dp", "ep", "eplb", "pp")


@pytest.fixture
def made(monkeypatch):
    """The device groups passed to make_standalone, in call order."""
    calls = []
    monkeypatch.setattr(spmd_util, "make_standalone", calls.append, raising=False)
    return calls


def _install(monkeypatch, stages, world_size, sizes):
    """vLLM groups of the given sizes; a kind missing from `sizes` raises like
    an uninitialized group. Each device group is named by its kind."""
    monkeypatch.setattr(mesh_utils, "get_pipeline_stage", lambda: (0, stages))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a: world_size)
    for kind in _KINDS:

        def get(kind=kind):
            assert kind in sizes, f"{kind} group is not initialized"
            return SimpleNamespace(world_size=sizes[kind], device_group=kind)

        monkeypatch.setattr(parallel_state, f"get_{kind}_group", get)


def test_stages_of_several_devices_make_their_groups_standalone(monkeypatch, made):
    # PP4 x PCP2 of a MoE model on 8 ranks: the PCP and EP groups are a stage,
    # the pipeline group one device of each stage.
    _install(
        monkeypatch,
        stages=4,
        world_size=8,
        sizes={"tp": 1, "dcp": 1, "pcp": 2, "dp": 1, "ep": 2, "pp": 4},
    )
    mesh_utils.make_stage_groups_standalone()
    assert made == ["pcp", "ep", "pp"]


def test_stages_of_one_device_make_nothing_standalone(monkeypatch, made):
    # PP8: every group but the pipeline group, which is the world, is one rank.
    _install(
        monkeypatch,
        stages=8,
        world_size=8,
        sizes={"tp": 1, "dcp": 1, "pcp": 1, "dp": 1, "pp": 8},
    )
    mesh_utils.make_stage_groups_standalone()
    assert made == []


def test_no_pipeline_makes_nothing_standalone(monkeypatch, made):
    # PCP2 x TP4 without pipeline parallelism keeps every group as it is.
    _install(
        monkeypatch,
        stages=1,
        world_size=8,
        sizes={"tp": 4, "dcp": 1, "pcp": 2, "dp": 1, "ep": 8, "pp": 1},
    )
    mesh_utils.make_stage_groups_standalone()
    assert made == []
