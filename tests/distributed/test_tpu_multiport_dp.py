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
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import vllm.logger as vllm_logger
from vllm.config.logging import LoggingConfig
from vllm.platforms.interface import Platform
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
    # engine_core_kwargs records the logging config it applies.
    monkeypatch.setattr(
        vllm_logger,
        "_last_configured_logging_config",
        vllm_logger._last_configured_logging_config,
    )


ON = {"tpu_multiport_dp": {"enabled": True}}


def _vllm_config(pc, additional=None):
    return SimpleNamespace(
        parallel_config=pc,
        additional_config={} if additional is None else additional,
        # Leaves this process's logging as it is.
        logging_config=LoggingConfig(configure_logging=False),
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
    configure = Mock()
    monkeypatch.setattr(vllm_logger, "configure_logging_if_needed", configure)
    pc = _child_config(2)
    cfg = _vllm_config(pc, additional)
    assert mpdp.local_dp_size(cfg) == 0
    assert mpdp.prepare_multiport_dp_env(cfg) is False
    assert _bootstrap() == (None, None)
    assert pc._data_parallel_master_port_list == [11111, 22222]
    kwargs = {"vllm_config": cfg, "dp_rank": 2, "local_dp_rank": 0}
    assert mpdp.engine_core_kwargs(kwargs) is kwargs
    assert pc.data_parallel_size_local == 1
    # The engine core's logging stays vLLM's to configure.
    configure.assert_not_called()


@pytest.mark.parametrize(
    "additional,match",
    [
        ({"tpu_multiport_dp": {"local_size": 4}}, "unknown keys"),
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
    # --data-parallel-rpc-port + 1.
    assert master_port == "29551"
    # One address per device of the whole DP4 x TP2 slice.
    assert sb == ",".join(f"localhost:{8070 + i}" for i in range(8))
    # vLLM's DP group ports, as prepare_mp_multihost_dp_env sets them.
    assert ports == (29650, 29651, 29652, 29653, 29654)


def test_dp_group_ports_come_from_the_multi_host_dp_helper(monkeypatch):
    """vLLM's DP group ports come from tpu_mp_multihost.dp_master_port_list,
    the helper prepare_mp_multihost_dp_env calls (tested there), so the two
    bootstraps cannot drift apart."""
    from vllm_torchtpu.distributed import tpu_mp_multihost

    calls = []

    def ports(rpc_port):
        calls.append(rpc_port)
        return [rpc_port, -1]

    monkeypatch.setattr(tpu_mp_multihost, "dp_master_port_list", ports)
    pc = _child_config(1)
    assert mpdp.prepare_multiport_dp_env(_vllm_config(pc, ON)) is True
    assert calls == [29550]
    assert pc._data_parallel_master_port_list == [29550, -1]


def test_bootstrap_honours_overrides(monkeypatch):
    monkeypatch.setenv("TORCH_TPU_BASE_PORT", "9000")
    # The multi-host rendezvous port does not move the DP master port.
    monkeypatch.setenv("TORCH_TPU_MP_RENDEZVOUS_PORT", "9500")
    mpdp.prepare_multiport_dp_env(_vllm_config(_child_config(1), ON))
    assert _bootstrap() == (
        "29551",
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
        mpdp.validate(_vllm_config(_child_config(0, external_lb=False), ON))


def test_multi_node_is_rejected(monkeypatch):
    pc = _child_config(0)
    pc.nnodes = 2
    with pytest.raises(NotImplementedError, match="nnodes"):
        mpdp.validate(_vllm_config(pc, ON))


def test_ray_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    with pytest.raises(NotImplementedError, match="TPU_MULTIHOST_BACKEND=ray"):
        mpdp.validate(_vllm_config(_child_config(0), ON))


def test_validation_is_a_no_op_with_the_option_off(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    pc = _child_config(0, external_lb=False)
    pc.nnodes = 2
    mpdp.validate(_vllm_config(pc, {}))
    mpdp.validate(_vllm_config(_child_config(0), ON | {"tpu_multiport_dp": {}}))


# --- duty 2: real local DP rank and local DP size in each engine core ---------


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_engine_core_gets_its_host_slot(monkeypatch, rank):
    configure = Mock()
    monkeypatch.setattr(vllm_logger, "configure_logging_if_needed", configure)
    pc = _child_config(rank)
    cfg = _vllm_config(pc, ON)
    kwargs = {
        "vllm_config": cfg,
        "dp_rank": rank,
        "local_dp_rank": 0,
        "log_stats": True,
    }
    out = mpdp.engine_core_kwargs(kwargs)
    configure.assert_called_once_with(cfg.logging_config)
    assert out["local_dp_rank"] == rank
    assert out["dp_rank"] == rank and out["log_stats"] is True
    assert pc.data_parallel_size_local == 4
    assert kwargs["local_dp_rank"] == 0  # the caller's dict is not mutated


def test_engine_core_of_internal_lb_is_left_alone(monkeypatch):
    pc = _child_config(3, external_lb=False)
    kwargs = {"vllm_config": _vllm_config(pc, ON), "dp_rank": 3, "local_dp_rank": 3}
    assert mpdp.engine_core_kwargs(kwargs) is kwargs
    assert pc.data_parallel_size_local == 1


# The engine-core entry point in a fresh interpreter, as vLLM's spawned engine
# core runs it: vLLM has not configured logging yet there.
_FRESH_ENGINE_CORE = textwrap.dedent(
    """
    from types import SimpleNamespace

    import vllm_torchtpu as plugin
    import vllm_torchtpu.core.pp_chunk_scheduler as pp_chunks
    import vllm_torchtpu.distributed.pp_engine as pp_engine
    from vllm.config.logging import LoggingConfig
    from vllm.v1.engine.core import EngineCoreProc
    from vllm_torchtpu import patch_registry

    patch_registry.apply = lambda stage: None
    pp_engine.patch_executor_for_pp_buffered = lambda cfg: None
    pp_chunks.patch_engine_core_for_pp_chunks = lambda cfg: None
    seen = {}
    EngineCoreProc._tpu_original_run_engine_core = lambda *a, **k: seen.update(k)
    pc = SimpleNamespace(
        data_parallel_size=2, data_parallel_size_local=1, data_parallel_rank=1,
        data_parallel_external_lb=True,
    )
    cfg = SimpleNamespace(
        parallel_config=pc,
        additional_config={"tpu_multiport_dp": {"enabled": True}},
        logging_config=LoggingConfig(),
    )
    plugin._run_engine_core_with_tpu_patches(
        vllm_config=cfg, dp_rank=1, local_dp_rank=0
    )
    print("LOCAL_DP_RANK", seen["local_dp_rank"], flush=True)
    """
)


def test_engine_core_rewrite_is_logged_in_a_fresh_engine_process():
    """The rewrite's log line is the launch's evidence that each engine core
    took its own slot; vLLM configures logging only inside run_engine_core,
    after this wrapper, so the line must not depend on an earlier setup."""
    env = dict(
        os.environ,
        VLLM_CONFIGURE_LOGGING="1",
        VLLM_LOGGING_LEVEL="INFO",
        VLLM_LOGGING_STREAM="ext://sys.stdout",
        JAX_PLATFORMS="cpu",
    )
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_ENGINE_CORE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "LOCAL_DP_RANK 1" in out.stdout
    assert (
        "TPU multi-port DP: engine core dp_rank=1 -> local_dp_rank=1, "
        "data_parallel_size_local=2"
    ) in out.stdout


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


@pytest.fixture
def isolated_registry(monkeypatch):
    """A registry with nothing applied yet, as in a freshly spawned process."""
    monkeypatch.setattr(patch_registry, "_applied", set())
    monkeypatch.setattr(patch_registry, "_applying", set())
    monkeypatch.setattr(patch_registry, "_active_stages", set())


@pytest.mark.parametrize("flag", [True, False])
def test_registration_lets_core_indexed_device_ids_through(
    monkeypatch, isolated_registry, flag
):
    """DPSupervisor hands rank 2 of DP4 x TP2 --device-ids [4, 5]; with four
    chips in TPU_VISIBLE_CHIPS, vLLM rejects that unless the platform stops
    consulting the variable. The general plugin installs the patch through the
    registry's general_plugin stage."""
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

    assert arg_utils.EngineArgs._resolve_device_ids._tpu_multiport_dp_patch
    assert (
        "vllm_torchtpu.distributed.tpu_multiport_dp:install_device_ids_patch"
        in patch_registry._applied
    )
    additional = ON if flag else {}
    if flag:
        assert _resolve_device_ids([4, 5], additional) == [4, 5]
        assert (
            TpuPlatform.device_control_env_var
            == Platform.device_control_env_var
            == "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
        )
    else:
        with pytest.raises(ValueError, match="out of range"):
            _resolve_device_ids([4, 5], additional)
        assert TpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


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
