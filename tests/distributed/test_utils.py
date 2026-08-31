from vllm_torchtpu import envs
from vllm_torchtpu.distributed import utils


def test_kv_transfer_names_default(monkeypatch):
    monkeypatch.delenv("TPU_KV_TRANSFER_NAMESPACE", raising=False)

    assert utils.get_ipc_socket_path(0) == "ipc:///tmp/tpu_conn_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_node0_0"


def test_kv_transfer_names_are_namespaced(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "prefill")

    assert utils.get_ipc_socket_path(
        0) == "ipc:///tmp/tpu_conn_prefill_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_prefill_node0_0"


def test_kv_transfer_namespace_is_sanitized(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "decode:9400")

    assert utils.get_ipc_socket_path(
        2) == "ipc:///tmp/tpu_conn_decode_9400_node2_0.sock"
    assert utils.get_shm_name(2) == "tpu_conn_kv_decode_9400_node2_0"


def test_ray_multihost_get_kv_ips_and_ports_empty(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setattr(utils, "_NODES_KV_IP_PORT", dict())

    assert utils.get_kv_ips() == []
    assert utils.get_kv_ports() == []


def test_ray_multihost_get_kv_ips_and_ports_sparse_node_ids(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setattr(utils, "_NODES_KV_IP_PORT", dict())

    # Regression test for KeyError when node_ids are not 0-indexed contiguous (e.g. node_id=1)
    utils.set_node_kv_ip_port((1, "10.36.48.210", 9101))

    assert utils.get_kv_ips() == ["10.36.48.210"]
    assert utils.get_kv_ports() == [9101]


def test_ray_multihost_get_kv_ips_and_ports_sorted(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setattr(utils, "_NODES_KV_IP_PORT", dict())

    # Insert out of order: node 2, then node 0, then node 1
    utils.set_node_kv_ip_port((2, "10.0.0.3", 9102))
    utils.set_node_kv_ip_port((0, "10.0.0.1", 9100))
    utils.set_node_kv_ip_port((1, "10.0.0.2", 9101))

    assert utils.get_kv_ips() == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert utils.get_kv_ports() == [9100, 9101, 9102]


def test_non_ray_get_kv_ips_and_ports(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MULTIHOST_BACKEND", "")
    monkeypatch.setattr(utils, "get_host_ip", lambda: "192.168.1.100")
    monkeypatch.setenv("TPU_KV_TRANSFER_PORT", "9200")

    assert utils.get_kv_ips() == "192.168.1.100"
    assert utils.get_kv_ports() == "9200"


def test_get_node_id(monkeypatch):
    # When ray backend is active, get_node_id() returns 0
    monkeypatch.setattr(envs, "TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("TPU_NODE_ID", "3")
    assert utils.get_node_id() == 0

    # When ray backend is not active, get_node_id() reads TPU_NODE_ID
    monkeypatch.setattr(envs, "TPU_MULTIHOST_BACKEND", "")
    monkeypatch.setenv("TPU_NODE_ID", "3")
    assert utils.get_node_id() == 3

    monkeypatch.delenv("TPU_NODE_ID", raising=False)
    assert utils.get_node_id() == 0
