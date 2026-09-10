import pytest

from vllm_torchtpu.distributed import utils

pytestmark = pytest.mark.cpu_test


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
