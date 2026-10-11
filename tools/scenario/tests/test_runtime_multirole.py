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
"""Prefill, decode and a driver on this machine: remote heads, the router, the client.

The in-process tests run the driver here against stub servers the test
starts; the multiprocess ones run every part as its own `scenario run`.
"""

import dataclasses
import json
import pathlib
from collections.abc import Iterator
from typing import Any

import pytest
from plan_fixtures import build
from runtime_helpers import (
    FAST,
    REPO_ROOT,
    LocalHost,
    free_ports,
    gone,
    kill_group,
    move_api_port,
    no_stray_processes,
    records,
    stub_env,
    tiny,
)
from scenario.plan import codec, plumbing
from scenario.plan.types import Plan
from scenario.runtime import bringup, control, driver, exits, net, proc
from scenario.runtime.control import StatusServer
from scenario.runtime.placement import LIBTPU_BLOCK, PlacementFlags
from scenario.runtime.proc import Proc

LOOPBACK = "127.0.0.1"
ROLE_HOSTS = "prefill=127.0.0.1;decode=127.0.0.1;client=127.0.0.1"


def pd1(driver_role: str = "client", hold_s: float = 0) -> dict[str, Any]:
    """tiny as one prefill and one decode host, a router, and `driver_role`."""
    data = tiny()
    data["name"] = "pd1"
    server = data["server"]
    hardware = {key: server.pop(key) for key in ("topology", "hosts")}
    del server["accelerator"], server["chips_per_host"]
    kv = {"connector": "TPURaidenConnector"}
    one_chip = {"accelerator": "tpu7x", "chips_per_host": 1, **hardware}
    server["roles"] = {
        "prefill": {**one_chip, "kv": {**kv, "role": "producer"}},
        "decode": {**one_chip, "kv": {**kv, "role": "consumer"}},
    }
    server["router"] = {
        "type": "toy-proxy",
        "command": ["python3", "examples/disagg/toy_proxy_server.py"],
    }
    perf = data["tasks"]["perf"]
    perf |= {"concurrency": [1], "warmup_runs": 0}
    del perf["baseline"]
    data["tasks"] = {
        "driver": driver_role,
        "perf": perf,
        "custom": [
            {"name": "hold", "command": ["sleep", str(hold_s)], "timeout": "1m"}
        ],
    }
    return data


@pytest.fixture
def ports(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Free ports for the Plans built here; decode gets its own API port."""
    api, decode, router, status = free_ports(4)
    monkeypatch.setattr(plumbing, "API_PORT", api)
    monkeypatch.setattr(plumbing, "ROUTER_PORT", router)
    return {"prefill": api, "decode": decode, "router": router, "status": status}


def localhost_plan(data: dict[str, Any], ports: dict[str, int]) -> Plan:
    return move_api_port(build(data), "decode", ports["decode"])


# The driver in this process ----------------------------------------------------


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch, ports: dict[str, int]) -> None:
    monkeypatch.setattr(control, "STATUS_PORT", ports["status"])
    monkeypatch.setattr(control, "POLL_S", 0.02)
    monkeypatch.setattr(driver, "OWN_SERVER_CHECK_S", 0.02)
    monkeypatch.setattr(bringup, "POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(bringup, "REMOTE_POLL_S", 0.02)
    monkeypatch.setattr(proc, "CHECK_EVERY_S", 0.02)


@pytest.fixture
def heads(tmp_path: pathlib.Path, ports: dict[str, int]) -> Iterator[dict[str, Proc]]:
    """Stub prefill and decode servers, standing in for those roles' hosts."""
    env = stub_env(tmp_path)
    started = {
        role: Proc(
            ["vllm", "serve", "test/tiny", "--port", str(ports[role])], env=env
        ).start()
        for role in ("prefill", "decode")
    }
    yield started
    for server in started.values():
        server.stop()


def nothing_left(env: dict[str, str]) -> bool:
    """Whether every process the driver started has gone; the heads are the test's."""
    started = [r for r in records(env) if r["program"] != "vllm serve"]
    return bool(started) and all(gone(found["pid"]) for found in started)


def client_options(tmp_path: pathlib.Path, env: dict[str, str]) -> driver.RunOptions:
    return driver.RunOptions(
        env=env,
        flags=PlacementFlags(role="client", role_hosts=ROLE_HOSTS),
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
    )


@pytest.mark.usefixtures("quick", "heads")
def test_a_client_drives_a_pair_through_the_router(
    tmp_path: pathlib.Path, ports: dict[str, int]
) -> None:
    env = stub_env(tmp_path)
    plan = localhost_plan(pd1(), ports)
    assert driver.run(plan, client_options(tmp_path, env)) == exits.OK

    (routed,) = records(env, "router")
    assert routed["argv"] == [
        "--host",
        "0.0.0.0",
        "--port",
        str(ports["router"]),
        "--prefiller-hosts",
        LOOPBACK,
        "--prefiller-ports",
        str(ports["prefill"]),
        "--decoder-hosts",
        LOOPBACK,
        "--decoder-ports",
        str(ports["decode"]),
    ]
    # The router and the tasks of a chipless driver cannot see libtpu.
    (bench,) = records(env, "vllm bench serve")
    for found in (routed, bench):
        assert found["env"]["PYTHONPATH"] == f"{LIBTPU_BLOCK}:{env['PYTHONPATH']}"
        assert found["env"]["VLLM_PLUGINS"] == ""
        assert found["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert bench["argv"][bench["argv"].index("--port") + 1] == str(ports["router"])
    run_dir = tmp_path / "results" / "pd1"
    assert "INFO stub router: on" in (run_dir / "router.log").read_text()
    assert not (run_dir / "server.log").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert list(summary["phases"]) == [
        "placement",
        "roles-ready",
        "router-ready",
        "exit",
    ]
    assert [(t["name"], t["status"]) for t in summary["tasks"]] == [
        ("perf", "passed"),
        ("hold", "passed"),
    ]
    assert nothing_left(env)


@pytest.mark.usefixtures("quick", "heads")
def test_a_dead_router_stops_the_tasks_with_5(
    tmp_path: pathlib.Path, ports: dict[str, int]
) -> None:
    env = stub_env(tmp_path, STUB_ROUTER_DIE_AFTER="0.3")
    data = pd1(hold_s=0.4)
    data["tasks"]["custom"] *= 2
    data["tasks"]["custom"][1] = {**data["tasks"]["custom"][1], "name": "after"}
    del data["tasks"]["perf"]
    plan = localhost_plan(data, ports)
    assert driver.run(plan, client_options(tmp_path, env)) == exits.SERVER_DIED
    summary = json.loads((tmp_path / "results/pd1/summary.json").read_text())
    # The router died while the first task ran; the check after it stops the rest.
    assert [(t["name"], t["status"], t.get("reason")) for t in summary["tasks"]] == [
        ("hold", "failed", "the router died"),
        ("after", "skipped", "the router died"),
    ]
    assert nothing_left(env)


@pytest.mark.usefixtures("quick", "heads")
def test_a_router_that_fails_to_start_is_4(
    tmp_path: pathlib.Path, ports: dict[str, int], capsys: pytest.CaptureFixture[str]
) -> None:
    env = stub_env(tmp_path, STUB_ROUTER_EXIT="1")
    plan = localhost_plan(pd1(), ports)
    assert driver.run(plan, client_options(tmp_path, env)) == exits.NOT_READY
    text = capsys.readouterr().out
    assert "[router] ERROR stub router: failed to start" in text
    assert "the router died while starting (exit 1)" in text
    assert records(env, "vllm bench serve") == []


@pytest.mark.usefixtures("quick")
def test_a_head_that_never_answers_is_4(
    tmp_path: pathlib.Path, ports: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(net, "RESOLVE_TIMEOUT_S", 0)
    plan = localhost_plan(pd1(), ports)
    roles = tuple(
        dataclasses.replace(
            role, readiness=dataclasses.replace(role.readiness, timeout_s=0)
        )
        for role in plan.roles
    )
    env = stub_env(tmp_path)
    code = driver.run(
        dataclasses.replace(plan, roles=roles), client_options(tmp_path, env)
    )
    assert code == exits.NOT_READY
    assert records(env) == []


@pytest.mark.usefixtures("quick")
def test_a_host_waiting_for_its_head_leaves_with_the_driver(
    tmp_path: pathlib.Path, ports: dict[str, int]
) -> None:
    data = pd1()
    data["server"]["multihost"] = "mp"
    decode = data["server"]["roles"]["decode"]
    decode |= {"topology": "2x2x2", "hosts": 2, "chips_per_host": 4}
    decode["args"] = {"data-parallel-size": 2}
    plan = build(data)
    status = StatusServer(codec.digest(plan), control.followers(plan)).start()
    status.set("done", 0)
    env = stub_env(tmp_path)
    hosts = "prefill=127.0.0.1;decode=head.invalid,127.0.0.1;client=127.0.0.1"
    options = driver.RunOptions(
        env=env,
        flags=PlacementFlags("decode", 1, None, LOOPBACK, hosts),
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
    )
    try:
        # Its own head's name never resolves; the driver's end ends the wait.
        assert driver.run(plan, options) == exits.OK
    finally:
        status.stop()
    assert records(env) == []


# Every part its own process -------------------------------------------------------


@pytest.fixture
def parts(tmp_path: pathlib.Path, ports: dict[str, int]) -> Iterator[list[LocalHost]]:
    started: list[LocalHost] = []
    yield started
    for host in started:
        host.kill()
    for found in records(stub_env(tmp_path)):
        kill_group(found["pid"])


def start(
    tmp_path: pathlib.Path,
    plan: Plan,
    role: str,
    status_port: int,
    **knobs: str,
) -> LocalHost:
    env = {**stub_env(tmp_path, **knobs), "STUB_HOST": f"{role}-0"}
    hosts = ROLE_HOSTS if plan.driver.chipless else ROLE_HOSTS.rsplit(";", 1)[0]
    args = ["--role", role, "--host-index", "0", "--advertise-ip", LOOPBACK]
    args += ["--role-hosts", hosts, "--results-dir", str(tmp_path / "results")]
    patch = {**FAST, "scenario.runtime.control.STATUS_PORT": status_port}
    return LocalHost(f"{role}-0", plan, args, env, tmp_path, patch)


def outputs(*hosts: LocalHost) -> str:
    return "\n".join(f"== {host.name}\n{host.output()}" for host in hosts)


@pytest.mark.multiprocess
def test_a_client_and_a_pair_end_together(
    tmp_path: pathlib.Path, ports: dict[str, int], parts: list[LocalHost]
) -> None:
    plan = localhost_plan(pd1(), ports)
    parts += [
        start(tmp_path, plan, role, ports["status"])
        for role in ("prefill", "decode", "client")
    ]
    codes = [host.wait() for host in parts]
    assert codes == [0, 0, 0], outputs(*parts)
    env = stub_env(tmp_path)
    served = {r["env"]["STUB_HOST"]: r for r in records(env, "vllm serve")}
    assert sorted(served) == ["decode-0", "prefill-0"]
    # Each head advertises its own address to the KV connector.
    for found in served.values():
        assert found["env"]["TPU_RAIDEN_ADVERTISE_HOST"] == LOOPBACK
        assert found["env"]["VLLM_HOST_IP"] == LOOPBACK
    (routed,) = records(env, "router")
    assert routed["env"]["STUB_HOST"] == "client-0"
    assert routed["env"]["PYTHONPATH"].startswith(f"{LIBTPU_BLOCK}:")
    client = parts[2].output()
    assert client.index("Role decode is ready") < client.index("router-ready")
    for role in ("prefill", "decode"):
        part = tmp_path / "results" / "pd1" / f"{role}-0" / "summary.json"
        assert json.loads(part.read_text())["exit"] == 0
        assert "Leaving with exit 0: the driver is done (exit 0)" in outputs(*parts)
    assert no_stray_processes(env)


@pytest.mark.multiprocess
def test_a_role_head_drives_its_pair(
    tmp_path: pathlib.Path, ports: dict[str, int], parts: list[LocalHost]
) -> None:
    plan = localhost_plan(pd1(driver_role="prefill"), ports)
    parts += [
        start(tmp_path, plan, role, ports["status"]) for role in ("decode", "prefill")
    ]
    assert [host.wait() for host in parts] == [0, 0], outputs(*parts)
    env = stub_env(tmp_path)
    (routed,) = records(env, "router")
    assert routed["env"]["STUB_HOST"] == "prefill-0"
    # A driver on a TPU host keeps its own platform: no libtpu block.
    assert not routed["env"].get("PYTHONPATH", "").startswith(str(LIBTPU_BLOCK))
    assert no_stray_processes(env)


@pytest.mark.multiprocess
def test_a_dead_router_ends_the_run_with_5(
    tmp_path: pathlib.Path, ports: dict[str, int], parts: list[LocalHost]
) -> None:
    data = pd1(hold_s=1)
    del data["tasks"]["perf"]
    data["tasks"]["custom"] *= 2
    data["tasks"]["custom"][1] = {**data["tasks"]["custom"][1], "name": "after"}
    plan = localhost_plan(data, ports)
    parts += [
        start(tmp_path, plan, role, ports["status"]) for role in ("prefill", "decode")
    ]
    parts.append(
        start(tmp_path, plan, "client", ports["status"], STUB_ROUTER_DIE_AFTER="0.5")
    )
    # The driver ends the run with 5; the hosts that served leave with it.
    assert [host.wait() for host in parts] == [0, 0, 5], outputs(*parts)
    assert "the router died (exit 1)" in parts[2].output()
    assert no_stray_processes(stub_env(tmp_path))
