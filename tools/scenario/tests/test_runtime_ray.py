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
"""A Ray role's bring-up against the stub `ray`: the head's node count, workers
joining, env before `ray start`, `ray stop` at the end."""

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
    kill_group,
    no_stray_processes,
    records,
    stub_env,
    tiny,
)
from scenario.plan import codec, plumbing
from scenario.plan.types import Plan
from scenario.runtime import bringup, control, driver, exits, proc
from scenario.runtime.control import StatusServer
from scenario.runtime.placement import PlacementFlags

LOOPBACK = "127.0.0.1"


def ray2() -> dict[str, Any]:
    """tiny over two Ray hosts."""
    data = tiny()
    data["name"] = "ray2"
    data["server"] |= {"topology": "2x2x2", "hosts": 2, "chips_per_host": 4}
    data["server"]["multihost"] = "ray"
    data["tasks"] = {"custom": [{"name": "hold", "command": ["true"], "timeout": "1m"}]}
    return data


@pytest.fixture
def ports(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    api, ray, status = free_ports(3)
    monkeypatch.setattr(plumbing, "API_PORT", api)
    monkeypatch.setattr(plumbing, "RAY_PORT", ray)
    monkeypatch.setattr(control, "STATUS_PORT", status)
    return {"api": api, "ray": ray, "status": status}


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch, ports: dict[str, int]) -> None:
    monkeypatch.setattr(control, "POLL_S", 0.02)
    monkeypatch.setattr(control, "LINGER_S", 0.2)
    monkeypatch.setattr(control, "ABORT_LINGER_S", 0.2)
    monkeypatch.setattr(driver, "OWN_SERVER_CHECK_S", 0.02)
    monkeypatch.setattr(bringup, "POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(bringup, "RAY_CLUSTER_POLL_S", 0.05)
    monkeypatch.setattr(bringup, "RAY_JOIN_PROBE_S", 0.05)
    monkeypatch.setattr(proc, "CHECK_EVERY_S", 0.02)


def options(
    tmp_path: pathlib.Path, env: dict[str, str], index: int
) -> driver.RunOptions:
    flags = PlacementFlags("server", index, LOOPBACK, LOOPBACK)
    return driver.RunOptions(
        env=env,
        flags=flags,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
    )


def programs(env: dict[str, str]) -> list[str]:
    return [found["program"] for found in records(env)]


@pytest.mark.usefixtures("quick")
@pytest.mark.parametrize("tpus", ["4", "0"])
def test_a_head_without_every_node_with_tpus_is_exit_3(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tpus: str,
) -> None:
    monkeypatch.setattr(bringup, "RAY_CLUSTER_TIMEOUT_S", 0.1)
    env = stub_env(tmp_path, STUB_RAY_TPU=tpus)
    assert driver.run(build(ray2()), options(tmp_path, env, 0)) == exits.BRINGUP
    text = capsys.readouterr().out
    alive = 1 if tpus == "4" else 0
    assert f"  {alive}/2 Ray nodes with TPUs" in text
    assert f"the Ray cluster had {alive} of 2 nodes with TPUs after 0s" in text
    # The head's server never starts; `ray status` explains, `ray stop` ends it.
    assert programs(env) == ["ray start", "raylet", "ray status", "ray stop"]
    summary = json.loads((tmp_path / "results/ray2/summary.json").read_text())
    assert (summary["exit"], summary["commands_ok"]) == (3, False)
    assert no_stray_processes(env)


@pytest.mark.usefixtures("quick")
def test_a_failed_ray_start_is_exit_3(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_RAY_START_EXIT="1")
    assert driver.run(build(ray2()), options(tmp_path, env, 0)) == exits.BRINGUP
    assert programs(env) == ["ray start"]


@pytest.mark.usefixtures("quick")
def test_a_worker_waiting_for_its_head_leaves_with_the_driver(
    tmp_path: pathlib.Path,
) -> None:
    plan = build(ray2())
    status = StatusServer(codec.digest(plan), control.followers(plan)).start()
    status.set("done", 4)
    env = stub_env(tmp_path)
    try:
        # No head ever listens; the driver's end still ends the wait.
        assert driver.run(plan, options(tmp_path, env, 1)) == exits.OK
    finally:
        status.stop()
    assert records(env) == []


@pytest.fixture
def hosts(tmp_path: pathlib.Path, ports: dict[str, int]) -> Iterator[list[LocalHost]]:
    started: list[LocalHost] = []
    yield started
    for host in started:
        host.kill()
    for found in records(stub_env(tmp_path)):
        kill_group(found["pid"])


def start(
    tmp_path: pathlib.Path, plan: Plan, index: int, status_port: int
) -> LocalHost:
    env = {**stub_env(tmp_path), "STUB_HOST": f"server-{index}"}
    args = ["--role", "server", "--host-index", str(index)]
    args += ["--head-address", LOOPBACK, "--advertise-ip", LOOPBACK]
    args += ["--results-dir", str(tmp_path / "results")]
    patch = {
        **FAST,
        "scenario.runtime.control.STATUS_PORT": status_port,
        "scenario.runtime.bringup.RAY_JOIN_PROBE_S": 0.05,
        "scenario.runtime.bringup.RAY_CLUSTER_POLL_S": 0.05,
    }
    return LocalHost(f"server-{index}", plan, args, env, tmp_path, patch)


@pytest.mark.multiprocess
def test_a_ray_role_serves_once_every_host_is_a_node(
    tmp_path: pathlib.Path, ports: dict[str, int], hosts: list[LocalHost]
) -> None:
    plan = build(ray2())
    hosts += [start(tmp_path, plan, i, ports["status"]) for i in (1, 0)]
    worker, head = hosts
    assert (head.wait(), worker.wait()) == (0, 0), head.output() + worker.output()
    env = stub_env(tmp_path)

    # Every raylet has the scenario's env and the tool's, which `ray start`
    # passed on, so the Ray workers it starts have them too.
    raylets = records(env, "raylet")
    assert sorted(r["node"] for r in raylets) == ["server-0", "server-1"]
    for raylet in raylets:
        assert raylet["env"]["TINY_KNOB"] == "1"
        assert raylet["env"]["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] == "1"
        assert raylet["env"]["TPU_MULTIHOST_BACKEND"] == "ray"
        assert raylet["env"]["JAX_PLATFORMS"] == ""
    starts = {r["env"]["STUB_HOST"]: r["argv"] for r in records(env, "ray start")}
    assert starts == {
        "server-0": ["start", "--head", f"--port={ports['ray']}"],
        "server-1": ["start", f"--address={LOOPBACK}:{ports['ray']}"],
    }
    # Only the head serves, and only once both hosts are nodes.
    (served,) = records(env, "vllm serve")
    assert served["env"]["STUB_HOST"] == "server-0"
    text = head.output()
    assert text.index("Ray cluster complete: 2 nodes") < text.index("server-spawn")
    assert "ray-joined" in worker.output()
    stops = sorted(r["env"]["STUB_HOST"] for r in records(env, "ray stop"))
    assert stops == ["server-0", "server-1"]
    assert no_stray_processes(env)
