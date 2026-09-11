import contextlib
import logging

import pytest

from vllm_torchtpu.distributed import utils

pytestmark = pytest.mark.cpu_test


@contextlib.contextmanager
def _captured_warnings(logger_name: str = "vllm"):
    messages: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: messages.append(record.getMessage())
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)


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


def test_get_kv_transfer_ips_defaults(monkeypatch):
    monkeypatch.delenv("TPU_KV_TRANSFER_IPS", raising=False)

    with monkeypatch.context() as m:
        m.setattr(utils, "get_host_ip", lambda: "192.168.1.10")
        assert utils.get_kv_transfer_ips() == ["192.168.1.10"]


def test_get_kv_transfer_ips_env(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_IPS", "10.0.0.1, 10.0.0.2")
    assert utils.get_kv_transfer_ips() == ["10.0.0.1", "10.0.0.2"]


def test_get_kv_transfer_ips_configured_ips(monkeypatch):
    monkeypatch.delenv("TPU_KV_TRANSFER_IPS", raising=False)

    # From string
    assert utils.get_kv_transfer_ips(
        configured_ips="10.30.0.1, 10.30.0.2") == ["10.30.0.1", "10.30.0.2"]

    # From list
    assert utils.get_kv_transfer_ips(
        configured_ips=["10.30.0.3", "10.30.0.4"]) == [
            "10.30.0.3", "10.30.0.4"
        ]

    # Explicit env vars take precedence over configured_ips
    monkeypatch.setenv("TPU_KV_TRANSFER_IPS", "10.99.0.1")
    assert utils.get_kv_transfer_ips(configured_ips=["10.30.0.3"]) == [
        "10.99.0.1"
    ]


def test_get_kv_transfer_ip_direct_resolution(monkeypatch):
    monkeypatch.delenv("TPU_KV_TRANSFER_IPS", raising=False)

    # 1. Single IP via TPU_KV_TRANSFER_IPS
    monkeypatch.setenv("TPU_KV_TRANSFER_IPS", "10.10.0.1")
    assert utils.get_kv_transfer_ip() == "10.10.0.1"

    # 2. NUMA-aware direct device resolution via LOCAL_RANK
    ip_to_numa = {
        "10.10.0.2": 0,
        "10.190.0.2": 1,
    }
    monkeypatch.setattr(utils, "_get_local_ip_to_numa_map", lambda: ip_to_numa)
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_device_numa_node",
        lambda dev_id: None
        if dev_id is None or dev_id < 0 else (0 if dev_id < 4 else 1),
    )
    monkeypatch.setenv("TPU_KV_TRANSFER_IPS", "10.10.0.2,10.190.0.2")

    # Device on NUMA 0 -> 10.10.0.2; Device on NUMA 1 -> 10.190.0.2
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"
    monkeypatch.setenv("LOCAL_RANK", "4")
    assert utils.get_kv_transfer_ip() == "10.190.0.2"
    monkeypatch.setenv("LOCAL_RANK", "6")
    assert utils.get_kv_transfer_ip() == "10.190.0.2"
    monkeypatch.setenv("LOCAL_RANK", "7")
    assert utils.get_kv_transfer_ip() == "10.190.0.2"

    # 3. Fallback to rank 0 when LOCAL_RANK is unset or invalid
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    monkeypatch.setenv("LOCAL_RANK", "not_an_int")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    monkeypatch.setenv("LOCAL_RANK", "-5")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    # 4. Fallback without NUMA map emits warning
    monkeypatch.setattr(utils, "_get_local_ip_to_numa_map", dict)
    with _captured_warnings() as warnings:
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert utils.get_kv_transfer_ip() == "10.10.0.2"
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert utils.get_kv_transfer_ip() == "10.190.0.2"
    assert any("Unable to resolve NUMA-aware KV transfer IP" in w
               for w in warnings)

    # 5. Fallback to get_host_ip() when no env set
    monkeypatch.delenv("TPU_KV_TRANSFER_IPS", raising=False)
    monkeypatch.setattr(utils, "get_host_ip", lambda: "192.168.1.99")
    assert utils.get_kv_transfer_ip() == "192.168.1.99"


def test_get_kv_transfer_ip_zero_arguments():
    import inspect
    sig = inspect.signature(utils.get_kv_transfer_ip)
    assert len(sig.parameters) == 0
    with pytest.raises(TypeError):
        utils.get_kv_transfer_ip(0)  # type: ignore
    with pytest.raises(TypeError):
        utils.get_kv_transfer_ip(device_id=0)  # type: ignore
    with pytest.raises(TypeError):
        utils.get_kv_transfer_ip(configured_ips=["10.0.0.1"])  # type: ignore


def test_get_kv_transfer_ip_platform_without_numa_method(monkeypatch):
    ip_to_numa = {"10.10.0.2": 0, "10.190.0.2": 1}
    monkeypatch.setattr(utils, "_get_local_ip_to_numa_map", lambda: ip_to_numa)
    monkeypatch.setenv("TPU_KV_TRANSFER_IPS", "10.10.0.2,10.190.0.2")

    class DummyPlatform:
        pass

    monkeypatch.setattr("vllm.platforms.current_platform", DummyPlatform())
    with _captured_warnings() as warnings:
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert utils.get_kv_transfer_ip() == "10.10.0.2"
    assert any("Unable to resolve NUMA-aware KV transfer IP" in w
               for w in warnings)

    class ErrorPlatform:

        def get_device_numa_node(self, dev_id):
            raise RuntimeError("Hardware sysfs unavailable")

    monkeypatch.setattr("vllm.platforms.current_platform", ErrorPlatform())
    with _captured_warnings() as warnings:
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert utils.get_kv_transfer_ip() == "10.190.0.2"
    assert any("Unable to resolve NUMA-aware KV transfer IP" in w
               for w in warnings)


def test_get_kv_transfer_ip_defensive_local_rank_parsing(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_IPS", "10.10.0.2,10.190.0.2")

    # 1. Fallback mode without NUMA map
    monkeypatch.setattr(utils, "_get_local_ip_to_numa_map", dict)

    monkeypatch.setenv("LOCAL_RANK", "1")
    assert utils.get_kv_transfer_ip() == "10.190.0.2"

    monkeypatch.setenv("LOCAL_RANK", "-1")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    monkeypatch.setenv("LOCAL_RANK", "invalid_rank")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    # 2. Active NUMA resolution mode
    ip_to_numa = {"10.10.0.2": 0, "10.190.0.2": 1}
    monkeypatch.setattr(utils, "_get_local_ip_to_numa_map", lambda: ip_to_numa)
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_device_numa_node",
        lambda dev_id: 0 if dev_id == 0 else 1,
    )

    monkeypatch.setenv("LOCAL_RANK", "invalid_number")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    monkeypatch.setenv("LOCAL_RANK", "-99")
    assert utils.get_kv_transfer_ip() == "10.10.0.2"

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert utils.get_kv_transfer_ip() == "10.10.0.2"


def test_get_local_ip_to_numa_map_json(monkeypatch):
    import json
    import subprocess

    fake_output = json.dumps([
        {
            "ifname": "lo",
            "addr_info": [{
                "local": "127.0.0.1"
            }]
        },
        {
            "ifname": "eth0",
            "addr_info": [{
                "local": "10.0.0.2"
            }]
        },
        {
            "ifname": "eth1",
            "addr_info": [{
                "local": "10.0.1.2"
            }]
        },
    ])

    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **kw: fake_output)
    monkeypatch.setattr(utils, "_read_interface_numa_node", lambda ifname: 0
                        if ifname == "eth0" else 1)

    mapping = utils._get_local_ip_to_numa_map()
    assert mapping == {"10.0.0.2": 0, "10.0.1.2": 1}

    # Error handling fallback to empty dict
    def _raise_error(*a, **kw):
        raise subprocess.CalledProcessError(1, "ip")

    monkeypatch.setattr(subprocess, "check_output", _raise_error)
    assert utils._get_local_ip_to_numa_map() == {}
