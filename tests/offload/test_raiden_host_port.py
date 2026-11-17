# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.offload.raiden_store import _host_port, _store_server_ip


def test_ipv4_and_hostname_unchanged():
    assert _host_port("127.0.0.1", 27800) == "127.0.0.1:27800"
    assert _host_port("head-0.svc", 27800) == "head-0.svc:27800"


def test_ipv6_literal_is_bracketed():
    assert _host_port("::1", 27800) == "[::1]:27800"
    assert _host_port("fd00::5", 1) == "[fd00::5]:1"
    assert _host_port("[::1]", 27800) == "[::1]:27800"


def test_store_server_ip_brackets_ipv6_for_the_controller_bind():
    # RaidenController binds store_server_ip + ":" + port; a bare "::1" gives
    # "::1:27800", which gRPC cannot bind.
    assert _store_server_ip("::1") == "[::1]"
    assert _store_server_ip("[::1]") == "[::1]"
    assert _store_server_ip("fd00::5") == "[fd00::5]"
    assert _store_server_ip("10.0.0.1") == "10.0.0.1"
    assert _store_server_ip("") == "127.0.0.1"
