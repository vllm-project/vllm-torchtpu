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
"""Single-host multi-port DP (additional_config["tpu_multiport_dp"]): each duty
that a DPSupervisor child needs, and that nothing changes with the option off."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.engine.core import EngineCoreProc

import vllm_torchtpu as plugin
from vllm_torchtpu import patch_registry
from vllm_torchtpu.distributed import tpu_multiport_dp as mpdp
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

_BOOTSTRAP_KEYS = (
    "TORCH_TPU_DP_MASTER_PORT",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_BASE_PORT",
    "TORCH_TPU_MP_RENDEZVOUS_PORT",
    "TPU_VISIBLE_CHIPS",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in _BOOTSTRAP_KEYS:
        # Track absent keys too: the module writes os.environ directly.
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)


ON = {"tpu_multiport_dp": {"enabled": True}}


def _vllm_config(pc, additional=None):
    return SimpleNamespace(
        parallel_config=pc, additional_config={} if additional is None else additional
    )


def _child_config(rank, *, dp=4, tp=2, external_lb=True, rpc_port=29550):
    """ParallelConfig fields of one multi-port child (DPSupervisor shape)."""
    return SimpleNamespace(
        data_parallel_size=dp,
        data_parallel_size_local=1,
        data_parallel_rank=rank,
        data_parallel_external_lb=external_lb,
        data_parallel_rpc_port=rpc_port,
        nnodes=1,
        world_size=tp,
        world_size_across_dp=dp * tp,
        _data_parallel_master_port_list=[11111, 22222],
    )


def _bootstrap():
    return (
        os.environ.get("TORCH_TPU_DP_MASTER_PORT"),
        os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES"),
    )


# --- off by default -----------------------------------------------------------


@pytest.mark.parametrize(
    "additional",
    [{}, None, {"tpu_multiport_dp": {}}, {"tpu_multiport_dp": {"enabled": False}}],
)
def test_off_changes_nothing(monkeypatch, additional):
    pc = _child_config(2)
    cfg = _vllm_config(pc, additional)
    assert mpdp.local_dp_size(cfg) == 0
    assert mpdp.prepare_multiport_dp_env(cfg) is False
    assert _bootstrap() == (None, None)
    assert pc._data_parallel_master_port_list == [11111, 22222]
    kwargs = {"vllm_config": cfg, "dp_rank": 2, "local_dp_rank": 0}
    assert mpdp.engine_core_kwargs(kwargs) is kwargs
    assert pc.data_parallel_size_local == 1


@pytest.mark.parametrize(
    "additional,match",
    [
        ({"tpu_multiport_dp": {"local_size": 4}}, "Unknown key"),
        ({"tpu_multiport_dp": {"enabled": 1}}, "must be a bool"),
        ({"tpu_multiport_dp": True}, "must be a JSON object"),
    ],
)
def test_malformed_option_is_rejected(additional, match):
    with pytest.raises(ValueError, match=match):
        mpdp.options(additional)


def test_option_parses_from_json_text():
    assert mpdp.options('{"tpu_multiport_dp": {"enabled": true}}').enabled


# --- duty 1: shared slice bootstrap in every child ----------------------------


def test_children_derive_one_shared_bootstrap(monkeypatch):
    seen = []
    for rank in range(4):
        for key in ("TORCH_TPU_DP_MASTER_PORT", "TORCH_TPU_SLICEBUILDER_ADDRESSES"):
            monkeypatch.delenv(key, raising=False)
        pc = _child_config(rank)
        assert mpdp.local_dp_size(_vllm_config(pc, ON)) == 4
        assert mpdp.prepare_multiport_dp_env(_vllm_config(pc, ON)) is True
        seen.append((_bootstrap(), tuple(pc._data_parallel_master_port_list)))
    assert len(set(seen)) == 1
    (master_port, sb), ports = seen[0]
    # The multi-host DP rendezvous port (rpc + 1), unused on this path.
    assert master_port == "29551"
    # One address per device of the whole DP4 x TP2 slice.
    assert sb == ",".join(f"localhost:{8070 + i}" for i in range(8))
    # vLLM's DP group ports, as prepare_mp_multihost_dp_env sets them.
    assert ports == (29650, 29651, 29652, 29653, 29654)


def test_bootstrap_honours_overrides(monkeypatch):
    monkeypatch.setenv("TORCH_TPU_BASE_PORT", "9000")
    monkeypatch.setenv("TORCH_TPU_MP_RENDEZVOUS_PORT", "9500")
    mpdp.prepare_multiport_dp_env(_vllm_config(_child_config(1), ON))
    assert _bootstrap() == (
        "9500",
        ",".join(f"localhost:{9000 + i}" for i in range(8)),
    )

    # An operator-set master port and a full-width list are kept.
    monkeypatch.setenv("TORCH_TPU_DP_MASTER_PORT", "7777")
    full = ",".join(f"h:{i}" for i in range(8))
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", full)
    mpdp.prepare_multiport_dp_env(_vllm_config(_child_config(1), ON))
    assert _bootstrap() == ("7777", full)

    # A list sized for one engine (2 devices) is replaced.
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "h:0,h:1")
    mpdp.prepare_multiport_dp_env(_vllm_config(_child_config(1), ON))
    assert len(os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == 8


def test_internal_lb_server_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="not a rank of"):
        mpdp.prepare_multiport_dp_env(
            _vllm_config(_child_config(0, external_lb=False), ON)
        )
    assert _bootstrap() == (None, None)


def test_multi_node_is_rejected(monkeypatch):
    pc = _child_config(0)
    pc.nnodes = 2
    with pytest.raises(NotImplementedError, match="nnodes"):
        mpdp.prepare_multiport_dp_env(_vllm_config(pc, ON))


# --- duty 2: real local DP rank and local DP size in each engine core ---------


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_engine_core_gets_its_host_slot(monkeypatch, rank):
    pc = _child_config(rank)
    kwargs = {
        "vllm_config": _vllm_config(pc, ON),
        "dp_rank": rank,
        "local_dp_rank": 0,
        "log_stats": True,
    }
    out = mpdp.engine_core_kwargs(kwargs)
    assert out["local_dp_rank"] == rank
    assert out["dp_rank"] == rank and out["log_stats"] is True
    assert pc.data_parallel_size_local == 4
    assert kwargs["local_dp_rank"] == 0  # the caller's dict is not mutated


def test_engine_core_of_internal_lb_is_left_alone(monkeypatch):
    pc = _child_config(3, external_lb=False)
    kwargs = {"vllm_config": _vllm_config(pc, ON), "dp_rank": 3, "local_dp_rank": 3}
    assert mpdp.engine_core_kwargs(kwargs) is kwargs
    assert pc.data_parallel_size_local == 1


def _stub_engine_core_entry(monkeypatch):
    import vllm_torchtpu.core.pp_chunk_scheduler as pp_chunks
    import vllm_torchtpu.distributed.pp_engine as pp_engine

    monkeypatch.setattr(patch_registry, "apply", lambda stage: None)
    monkeypatch.setattr(pp_engine, "patch_executor_for_pp_buffered", lambda cfg: None)
    monkeypatch.setattr(pp_chunks, "patch_engine_core_for_pp_chunks", lambda cfg: None)
    run = Mock()
    monkeypatch.setattr(
        EngineCoreProc, "_tpu_original_run_engine_core", run, raising=False
    )
    return run


@pytest.mark.parametrize("flag,expected", [(True, 3), (False, 0)])
def test_engine_core_entry_point_rewrites_only_under_the_flag(
    monkeypatch, flag, expected
):
    """Through vllm-torchtpu's own run_engine_core wrapper, the path a spawned
    engine core actually takes."""
    run = _stub_engine_core_entry(monkeypatch)
    pc = _child_config(3)
    cfg = _vllm_config(pc, ON if flag else {})
    plugin._run_engine_core_with_tpu_patches(
        vllm_config=cfg, dp_rank=3, local_dp_rank=0
    )
    run.assert_called_once()
    assert run.call_args.kwargs["local_dp_rank"] == expected
    assert run.call_args.kwargs["dp_rank"] == 3
    assert pc.data_parallel_size_local == (4 if flag else 1)


# --- duty 3: the supervisor's --device-ids (TPU core indices) -----------------


def _resolve_device_ids(device_ids, additional):
    import vllm.engine.arg_utils as arg_utils

    return arg_utils.EngineArgs._resolve_device_ids(
        SimpleNamespace(
            device_ids=device_ids,
            distributed_executor_backend=None,
            additional_config=additional,
        )
    )


@pytest.mark.parametrize("flag,raises", [(True, False), (False, True)])
def test_registration_lets_core_indexed_device_ids_through(monkeypatch, flag, raises):
    """DPSupervisor hands rank 2 of DP4 x TP2 --device-ids [4, 5]; with four
    chips in TPU_VISIBLE_CHIPS, vLLM rejects that unless the platform stops
    consulting the variable."""
    import vllm.engine.arg_utils as arg_utils

    from vllm_torchtpu.layers import register_layers

    monkeypatch.setattr(TpuPlatform, "device_control_env_var", "TPU_VISIBLE_CHIPS")
    monkeypatch.setattr(arg_utils, "current_platform", TpuPlatform)
    monkeypatch.setattr(
        arg_utils.EngineArgs,
        "_resolve_device_ids",
        arg_utils.EngineArgs._resolve_device_ids,
    )
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0,1,2,3")

    register_layers()

    if raises:
        with pytest.raises(ValueError, match="out of range"):
            _resolve_device_ids([4, 5], {})
        assert TpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"
    else:
        assert _resolve_device_ids([4, 5], ON) == [4, 5]
        assert TpuPlatform.device_control_env_var == mpdp.DEVICE_CONTROL_PLACEHOLDER


def test_device_ids_patch_is_installed_once(monkeypatch):
    import vllm.engine.arg_utils as arg_utils

    monkeypatch.setattr(
        arg_utils.EngineArgs,
        "_resolve_device_ids",
        arg_utils.EngineArgs._resolve_device_ids,
    )
    mpdp.install_device_ids_patch()
    first = arg_utils.EngineArgs._resolve_device_ids
    mpdp.install_device_ids_patch()
    assert arg_utils.EngineArgs._resolve_device_ids is first
    assert first._tpu_multiport_dp_patch
