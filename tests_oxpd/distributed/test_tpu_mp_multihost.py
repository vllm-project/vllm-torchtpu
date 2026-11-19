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

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from vllm_torchtpu.distributed import tpu_mp_multihost as mp
from vllm_torchtpu.platforms import tpu_platform


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCH_TPU_MP_RENDEZVOUS_PORT",
        "TORCH_TPU_BASE_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_XPROF_SESSION_ID",
        "TPU_NUM_HOSTS",
        "NODE_RANK",
        mp._DONE_SENTINEL,
        mp._DP_DONE_SENTINEL,
    ):
        # Track absent keys too, since the module writes os.environ directly.
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)
    monkeypatch.setattr(mp, "_keepalive_stores", [])


@pytest.mark.parametrize("override,expected", [(None, 9001), ("9100", 9100)])
def test_rendezvous_port(monkeypatch, override, expected):
    if override is not None:
        monkeypatch.setenv("TORCH_TPU_MP_RENDEZVOUS_PORT", override)
    assert mp._rendezvous_port(9000) == expected


@pytest.mark.parametrize("rank", [0, 1])
def test_rendezvous_orders_peers_and_shares_session(monkeypatch, rank):
    values = {
        "test_ip_0": b"10.0.0.1",
        "test_ip_1": b"10.0.0.2",
        "test_xprof_session_id": b"12345",
    }
    store = Mock()
    store.get.side_effect = values.__getitem__
    create_store = Mock(return_value=store)
    monkeypatch.setattr(mp.dist, "TCPStore", create_store)
    monkeypatch.setattr(mp, "get_ip", lambda: f"10.0.0.{rank + 1}")
    monkeypatch.setattr(mp.time, "time_ns", lambda: 12345)
    assert mp._rendezvous_host_ips(
        num_peers=2,
        peer_index=rank,
        master_addr="10.0.0.1",
        port=9001,
        key_prefix="test",
    ) == (["10.0.0.1", "10.0.0.2"], "12345")
    create_store.assert_called_once_with(
        host_name="10.0.0.1",
        port=9001,
        world_size=2,
        is_master=rank == 0,
        timeout=timedelta(seconds=300),
    )
    expected_sets = [call(f"test_ip_{rank}", f"10.0.0.{rank + 1}")]
    if rank == 0:
        expected_sets.append(call("test_xprof_session_id", "12345"))
    assert store.set.call_args_list == expected_sets
    store.wait.assert_called_once_with(list(values), timedelta(seconds=300))
    assert mp._keepalive_stores == ([store] if rank == 0 else [])


@pytest.fixture
def bootstrap(monkeypatch):
    rendezvous = Mock(return_value=(["10.0.0.1", "10.0.0.2"], "session"))
    topology = Mock(return_value="1,2,2,2")
    monkeypatch.setattr(mp, "_rendezvous_host_ips", rendezvous)
    monkeypatch.setattr(tpu_platform, "get_tpu_multihost_topology", topology)
    return rendezvous, topology


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("base_port", [None, "8500"])
def test_mp_bootstrap_and_reentry(monkeypatch, bootstrap, rank, base_port):
    rendezvous, topology = bootstrap
    if base_port is not None:
        monkeypatch.setenv("TORCH_TPU_BASE_PORT", base_port)
    config = SimpleNamespace(
        nnodes=2,
        node_rank=rank,
        local_world_size=4,
        master_addr="10.0.0.1",
        master_port=9000,
    )
    mp.prepare_mp_multihost_env(config)
    rendezvous.assert_called_once_with(
        num_peers=2,
        peer_index=rank,
        master_addr="10.0.0.1",
        port=9001,
        key_prefix="tpu_mp_nnodes",
    )
    topology.assert_called_once_with(8)
    ports = "8500,8501,8502,8503" if base_port else "8070,8071,8072,8073"
    expected_addresses = [
        f"{ip}:{port}" for ip in ["10.0.0.1", "10.0.0.2"] for port in ports.split(",")
    ]
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == ",".join(
        expected_addresses
    )
    assert os.environ["TORCH_TPU_TOPOLOGY"] == "1,2,2,2"
    assert os.environ["TPU_NUM_HOSTS"] == "2"
    assert os.environ["NODE_RANK"] == str(rank)
    assert os.environ["TORCH_TPU_XPROF_SESSION_ID"] == "session"
    assert os.environ["MASTER_ADDR"] == config.master_addr
    assert os.environ["MASTER_PORT"] == str(config.master_port)
    mp.prepare_mp_multihost_env(config)
    assert rendezvous.call_count == 1
    assert os.environ["MASTER_PORT"] == str(config.master_port)


def test_single_host_skips_rendezvous(bootstrap):
    mp.prepare_mp_multihost_env(SimpleNamespace(nnodes=1))
    bootstrap[0].assert_not_called()
    bootstrap[1].assert_not_called()
    assert mp._DONE_SENTINEL not in os.environ


@pytest.mark.parametrize("local_size", [0, -1, 4, 8])
def test_dp_without_remote_replicas_skips_rendezvous(bootstrap, local_size):
    config = SimpleNamespace(data_parallel_size=4, data_parallel_size_local=local_size)
    assert not mp.prepare_mp_multihost_dp_env(config, 8)
    bootstrap[0].assert_not_called()


def test_dp_rejects_uneven_host_partition(bootstrap):
    config = SimpleNamespace(data_parallel_size=5, data_parallel_size_local=2)
    with pytest.raises(ValueError, match="must be a multiple"):
        mp.prepare_mp_multihost_dp_env(config, 10)
    bootstrap[0].assert_not_called()


@pytest.mark.parametrize("start_rank,peer", [(0, 0), (2, 1)])
def test_dp_bootstrap_sets_slice_and_reapplies_master_ports(
    bootstrap, start_rank, peer
):
    rendezvous, topology = bootstrap
    config = SimpleNamespace(
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_rank=start_rank,
        data_parallel_rpc_port=9000,
        data_parallel_master_ip="10.0.0.1",
        world_size=2,
    )
    assert mp.prepare_mp_multihost_dp_env(config, 8)
    rendezvous.assert_called_once_with(
        num_peers=2,
        peer_index=peer,
        master_addr="10.0.0.1",
        port=9001,
        key_prefix="tpu_mp_dp",
    )
    topology.assert_called_once_with(8)
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == (
        "10.0.0.1:8070,10.0.0.1:8071,10.0.0.1:8072,10.0.0.1:8073,"
        "10.0.0.2:8070,10.0.0.2:8071,10.0.0.2:8072,10.0.0.2:8073"
    )
    assert os.environ["TORCH_TPU_TOPOLOGY"] == "1,2,2,2"
    assert os.environ["TORCH_TPU_XPROF_SESSION_ID"] == "session"
    assert config._data_parallel_master_port_list == [9100, 9101, 9102, 9103, 9104]
    config._data_parallel_master_port_list = [42]
    assert mp.prepare_mp_multihost_dp_env(config, 8)
    assert config._data_parallel_master_port_list == [9100, 9101, 9102, 9103, 9104]
    assert rendezvous.call_count == 1


def test_failed_bootstrap_can_retry(bootstrap):
    rendezvous, topology = bootstrap
    rendezvous.side_effect = RuntimeError("peer timeout")
    config = SimpleNamespace(
        nnodes=2,
        node_rank=0,
        local_world_size=4,
        master_addr="10.0.0.1",
        master_port=9000,
    )
    with pytest.raises(RuntimeError, match="peer timeout"):
        mp.prepare_mp_multihost_env(config)
    assert mp._DONE_SENTINEL not in os.environ
    topology.assert_not_called()
    rendezvous.side_effect = None
    mp.prepare_mp_multihost_env(config)
    assert os.environ[mp._DONE_SENTINEL] == "2"
