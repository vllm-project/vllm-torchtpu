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
"""The hosts of one mp role on this machine: the driver, its followers, pre commands.

Tests marked multiprocess run every host as its own `scenario run` process
(`pytest -m multiprocess`); the others run one follower in this process
against a status endpoint standing in for the driver.
"""

import dataclasses
import json
import pathlib
import signal
import threading
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
    state,
    stub_env,
    tiny,
    wait_for,
)
from scenario.plan import codec, plumbing
from scenario.plan.types import Command, HostPlan, Plan
from scenario.runtime import bringup, control, driver, exits
from scenario.runtime.control import StatusServer
from scenario.runtime.exits import RunError
from scenario.runtime.log import Log
from scenario.runtime.placement import Placement, PlacementFlags, Refs

LOOPBACK = "127.0.0.1"


def mp2(hold_s: float = 0) -> dict[str, Any]:
    """tiny over two mp hosts, with remote code; its one task sleeps `hold_s`."""
    data = tiny()
    data["name"] = "mp2"
    server = data["server"]
    server |= {"topology": "2x2x2", "hosts": 2, "chips_per_host": 4}
    server["multihost"] = "mp"
    server["model"] = {"name": "test/tiny", "path": "gs://bucket/tiny"}
    server["args"] |= {
        "load-format": "runai_streamer",
        "data-parallel-size": 2,
        "trust-remote-code": True,
        "api-server-count": 1,
    }
    data["tasks"] = {
        "custom": [{"name": "hold", "command": ["sleep", str(hold_s)], "timeout": "1m"}]
    }
    return data


@pytest.fixture
def ports(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Free ports for the API and DP coordination in Plans built here, and the
    driver status port for the hosts."""
    api, dp, status = free_ports(3)
    monkeypatch.setattr(plumbing, "API_PORT", api)
    monkeypatch.setattr(plumbing, "DP_RPC_PORT", dp)
    return {"api": api, "dp": dp, "status": status}


class Localhost:
    """The hosts of one run on this machine, and what is left of them at the end."""

    def __init__(self, tmp_path: pathlib.Path, status_port: int) -> None:
        self.tmp_path = tmp_path
        self.env = stub_env(tmp_path)
        self.results = tmp_path / "results"
        self.patch = {**FAST, "scenario.runtime.control.STATUS_PORT": status_port}
        self.hosts: list[LocalHost] = []

    def start(
        self,
        plan: Plan,
        role: str,
        index: int,
        *,
        env: dict[str, str] | None = None,
        patch: dict[str, Any] | None = None,
        role_hosts: str | None = None,
    ) -> LocalHost:
        args = [
            "--role",
            role,
            "--host-index",
            str(index),
            "--head-address",
            LOOPBACK,
            "--advertise-ip",
            LOOPBACK,
            "--results-dir",
            str(self.results),
        ]
        if role_hosts is not None:
            args += ["--role-hosts", role_hosts]
        host = LocalHost(
            f"{role}-{index}",
            plan,
            args,
            {**self.env, **(env or {})},
            self.tmp_path,
            {**self.patch, **(patch or {})},
        )
        self.hosts.append(host)
        return host

    def run_dir(self, plan: Plan) -> pathlib.Path:
        return self.results / plan.scenario_name

    def summary(self, plan: Plan, part: str | None = None) -> pathlib.Path:
        found = self.run_dir(plan)
        if part is not None:
            found /= part
        return found / "summary.json"

    def served(self, headless: bool) -> dict[str, Any]:
        (found,) = [
            r
            for r in records(self.env, "vllm serve")
            if ("--headless" in r["argv"]) == headless
        ]
        return found

    def close(self) -> None:
        for host in self.hosts:
            host.kill()
        for found in records(self.env):
            kill_group(found["pid"])


@pytest.fixture
def localhost(tmp_path: pathlib.Path, ports: dict[str, int]) -> Iterator[Localhost]:
    run = Localhost(tmp_path, ports["status"])
    yield run
    run.close()


def outputs(*hosts: LocalHost) -> str:
    return "\n".join(f"== {host.name}\n{host.output()}" for host in hosts)


# Several processes ----------------------------------------------------------


@pytest.mark.multiprocess
def test_two_hosts_end_together_when_the_driver_is_done(
    localhost: Localhost, ports: dict[str, int]
) -> None:
    plan = build(mp2())
    head = localhost.start(plan, "server", 0)
    # The worker's first poll must find the status server up, so the driver
    # has heard from it and waits for it at the end.
    wait_for(
        lambda: "Serving the driver status on port" in head.output(),
        "the head to serve its status",
    )
    # The worker polls rarely, so it learns of the end from its own engine,
    # which dies with the head's: one more request finds the driver done.
    worker = localhost.start(
        plan, "server", 1, patch={"scenario.runtime.control.POLL_S": 30.0}
    )
    assert (head.wait(), worker.wait()) == (0, 0), outputs(head, worker)

    headless = localhost.served(headless=True)
    assert headless["argv"][-3:] == ["--headless", "--data-parallel-start-rank", "1"]
    assert "--port" not in headless["argv"]
    assert "--api-server-count" not in headless["argv"]
    address = headless["argv"].index("--data-parallel-address")
    assert headless["argv"][address + 1] == LOOPBACK
    assert headless["env"]["TPU_MULTIHOST_BACKEND"] == "mp"
    assert headless["env"]["VLLM_HOST_IP"] == LOOPBACK
    assert headless["env"]["TPU_SKIP_MDS_QUERY"] == "1"
    # The remote-code warm ran on the worker before its server.
    programs = [r["program"] for r in records(localhost.env)]
    warm = programs.index("transformers AutoTokenizer")
    assert warm < programs.index("vllm serve", warm)
    assert "--api-server-count" in localhost.served(headless=False)["argv"]

    text = worker.output()
    assert "The driver answered: starting" in text
    assert "ERROR stub: lost the data-parallel head" in text
    assert "This host's server exited (exit 1)" in text
    assert "Leaving with exit 0: the driver is done (exit 0)" in text
    assert "Serving the driver status on port" in head.output()
    assert "Not heard from" not in head.output()
    part = json.loads(localhost.summary(plan, "server-1").read_text())
    assert (part["part"], part["state"], part["exit"]) == ("server/1", "done", 0)
    assert list(part["phases"]) == ["placement", "pre-done", "server-spawn", "exit"]
    done = json.loads(localhost.summary(plan).read_text())
    assert (done["state"], done["exit"]) == ("done", 0)
    assert no_stray_processes(localhost.env)


def _serving(localhost: Localhost, plan: Plan) -> tuple[LocalHost, LocalHost]:
    """Both hosts, once the worker has seen the driver serving."""
    head = localhost.start(plan, "server", 0)
    worker = localhost.start(plan, "server", 1)
    wait_for(
        lambda: "The driver is serving" in worker.output(),
        "the worker to see the driver serving",
    )
    return head, worker


@pytest.mark.multiprocess
def test_an_aborted_driver_ends_its_followers_with_7(localhost: Localhost) -> None:
    plan = build(mp2(hold_s=60))
    head, worker = _serving(localhost, plan)
    head.send(signal.SIGTERM)
    assert (head.wait(), worker.wait()) == (143, 7), outputs(head, worker)
    assert "the driver aborted" in worker.output()
    assert no_stray_processes(localhost.env)


@pytest.mark.multiprocess
def test_a_lost_driver_ends_its_followers_with_7(localhost: Localhost) -> None:
    plan = build(mp2(hold_s=60))
    head, worker = _serving(localhost, plan)
    head.kill()
    # What the killed driver started lives on, as on a host that lost it.
    assert worker.wait() == exits.DRIVER_LOST, outputs(worker)
    assert "it was lost before it finished" in worker.output()
    kill_group(localhost.served(headless=False)["pid"])
    assert no_stray_processes(localhost.env)


@pytest.mark.multiprocess
def test_a_follower_whose_server_dies_starting_exits_4(localhost: Localhost) -> None:
    plan = build(mp2())
    head = localhost.start(plan, "server", 0)
    worker = localhost.start(plan, "server", 1, env={"STUB_SERVE_EXIT": "1"})
    assert worker.wait() == exits.NOT_READY, outputs(head, worker)
    assert "this host's server died while starting" in worker.output()
    # The head never has every host, so it waits until stopped.
    assert state(localhost.summary(plan)) == "starting"
    head.send(signal.SIGTERM)
    assert head.wait() == 143
    assert no_stray_processes(localhost.env)


@pytest.mark.multiprocess
def test_a_follower_whose_server_dies_serving_exits_5(localhost: Localhost) -> None:
    plan = build(mp2(hold_s=1))
    head, worker = _serving(localhost, plan)
    kill_group(localhost.served(headless=True)["pid"])
    assert worker.wait() == exits.SERVER_DIED, outputs(head, worker)
    assert "this host's server died" in worker.output()
    # The stub head does not notice; the driver ends as it would have.
    assert head.wait() == 0, outputs(head)
    assert "Not heard from since the end: server/1" in head.output()
    assert no_stray_processes(localhost.env)


@pytest.mark.multiprocess
def test_a_follower_of_another_plan_exits_2(
    localhost: Localhost, ports: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build(mp2())
    monkeypatch.setattr(plumbing, "DP_RPC_PORT", ports["dp"] + 1)
    other = build(mp2())
    assert codec.digest(other) != codec.digest(plan)
    head = localhost.start(plan, "server", 0)
    worker = localhost.start(other, "server", 1)
    assert worker.wait() == exits.USAGE, outputs(head, worker)
    assert "they belong to different runs" in worker.output()
    head.send(signal.SIGTERM)
    assert head.wait() == 143
    assert no_stray_processes(localhost.env)


# One follower in this process -------------------------------------------------


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch, ports: dict[str, int]) -> None:
    from scenario.runtime import proc

    monkeypatch.setattr(control, "STATUS_PORT", ports["status"])
    monkeypatch.setattr(control, "POLL_S", 0.02)
    monkeypatch.setattr(driver, "OWN_SERVER_CHECK_S", 0.02)
    monkeypatch.setattr(proc, "CHECK_EVERY_S", 0.02)


def follower_options(
    tmp_path: pathlib.Path, env: dict[str, str], **flags: Any
) -> driver.RunOptions:
    placement = {
        "role": "server",
        "host_index": 1,
        "head_address": LOOPBACK,
        "advertise_ip": LOOPBACK,
        **flags,
    }
    return driver.RunOptions(
        env=env,
        flags=PlacementFlags(**placement),
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
    )


@pytest.mark.usefixtures("quick")
def test_a_follower_starts_at_once_and_leaves_with_the_driver(
    tmp_path: pathlib.Path,
) -> None:
    plan = build(mp2())
    env = stub_env(tmp_path)
    status = StatusServer(codec.digest(plan), control.followers(plan)).start()
    found: list[int] = []
    try:
        thread = threading.Thread(
            target=lambda: found.append(
                driver.run(plan, follower_options(tmp_path, env))
            )
        )
        thread.start()
        # It serves while the driver is still starting.
        wait_for(lambda: records(env, "vllm serve"), "the follower's server")
        status.set("serving")
        status.set("done", 0)
        thread.join(30)
    finally:
        status.stop()
    assert found == [exits.OK]
    part = tmp_path / "results" / "mp2" / "server-1"
    assert json.loads((part / "summary.json").read_text())["exit"] == 0
    assert "INFO stub: shutting down" in (part / "server.log").read_text()
    assert not (tmp_path / "results" / "mp2" / "config.json").exists()
    assert no_stray_processes(env)


@pytest.mark.usefixtures("quick")
def test_a_follower_without_a_driver_gives_up_with_3(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "FIRST_CONTACT_S", 0.3)
    env = stub_env(tmp_path)
    code = driver.run(build(mp2()), follower_options(tmp_path, env))
    assert code == exits.BRINGUP
    assert no_stray_processes(env)


def test_a_follower_needs_the_drivers_address(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    options = follower_options(tmp_path, env, head_address=None)
    assert driver.run(build(mp2()), options) == exits.USAGE
    assert records(env) == []


# Pre commands -----------------------------------------------------------------


def pre_host(*commands: Command) -> HostPlan:
    return HostPlan(1, False, commands, None, (), (), ())


def run_pre(tmp_path: pathlib.Path, host: HostPlan) -> None:
    plan = build(mp2())
    refs = Refs(
        plan,
        Placement("server", 1),
        run_dir=tmp_path,
        server_log=tmp_path / "server.log",
        env={},
    )
    log = Log("server/1", "sha", "0")
    env = stub_env(tmp_path)
    bringup.run_pre(host, refs=refs, env=env, added={}, cwd=tmp_path, log=log)


def test_pre_commands_run_in_order_within_their_timeouts(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "order"
    first = Command(("sh", "-c", f"echo one >> {out}"))
    slow = Command(("sleep", "30"), timeout_s=0, best_effort=True)
    failing = Command(("sh", "-c", f"echo two >> {out}; exit 3"), best_effort=True)
    run_pre(tmp_path, pre_host(first, slow, failing))
    assert out.read_text() == "one\ntwo\n"
    text = capsys.readouterr().out
    assert "::warning::pre[1] (sleep) did not finish in 0s; it is best effort" in text
    assert "::warning::pre[2] (sh) exited 3; it is best effort" in text


def test_a_failed_pre_command_is_exit_3(tmp_path: pathlib.Path) -> None:
    after = tmp_path / "after"
    host = pre_host(Command(("false",)), Command(("touch", str(after))))
    with pytest.raises(RunError) as error:
        run_pre(tmp_path, host)
    assert (error.value.code, error.value.message) == (
        exits.BRINGUP,
        "pre[0] (false) exited 1",
    )
    assert not after.exists()


def test_a_best_effort_warm_that_fails_does_not_stop_the_host(
    tmp_path: pathlib.Path,
) -> None:
    (host,) = [h for h in build(mp2()).roles[0].host_plans if h.pre]
    warm = dataclasses.replace(host.pre[0], argv=("false",))
    run_pre(tmp_path, dataclasses.replace(host, pre=(warm,)))
