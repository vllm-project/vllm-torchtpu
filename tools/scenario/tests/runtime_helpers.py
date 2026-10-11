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
"""Helpers for runtime tests: stub programs on PATH, their records, a small scenario.

Imported by name, like plan_fixtures.py. The stubs (stubs/stubs.py) need no
network, TPU or vLLM.
"""

import contextlib
import dataclasses
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

from scenario.plan import codec
from scenario.plan.types import Command, Concat, Plan, Readiness, Ref

STUBS = pathlib.Path(__file__).resolve().parent / "stubs" / "stubs.py"
# `scenario` with runtime constants replaced from $STUB_PATCH.
RUN_SCENARIO = pathlib.Path(__file__).resolve().parent / "stubs" / "run_scenario.py"
PROGRAMS = ("vllm", "python3", "lm_eval", "evalscope", "ray")
# A stand-in vllm package: just runai_utils.ObjectStorageModel.
SITE = pathlib.Path(__file__).resolve().parent / "stubs" / "site"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def free_ports(count: int) -> list[int]:
    """`count` distinct free ports."""
    with contextlib.ExitStack() as stack:
        socks = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in socks:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in socks]


def stub_env(tmp_path: pathlib.Path, **knobs: str) -> dict[str, str]:
    """The env a run gets: stub programs first on PATH, records under tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for program in PROGRAMS:
        shim = bin_dir / program
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{STUBS}" {program} "$@"\n'
        )
        shim.chmod(0o755)
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SCENARIO_", "STUB_"))
    }
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_RECORDS"] = str(records)
    env["PYTHONPATH"] = str(SITE)
    env["STUB_PULLED_DIR"] = str(tmp_path / "pulled")
    env["STUB_RAY_STATE"] = str(tmp_path / "ray-state.json")
    env.update(knobs)
    return env


def records(env: dict[str, str], program: str | None = None) -> list[dict[str, Any]]:
    """Every recorded call in order, or those of one program."""
    found = [
        json.loads(path.read_text())
        for path in sorted(pathlib.Path(env["STUB_RECORDS"]).glob("*.json"))
    ]
    return [r for r in found if program is None or r["program"] == program]


def gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def no_stray_processes(env: dict[str, str]) -> bool:
    """Whether every recorded process and engine child has gone."""
    pids = []
    for found in records(env):
        pids.append(found["pid"])
        pids += [found[key] for key in ("engine_pid", "daemon_pid") if key in found]
    return all(gone(pid) for pid in pids)


PERCENTILES = ["ttft", "tpot", "itl", "e2el"]


def tiny() -> dict[str, Any]:
    """One chip, perf and eval, with the values defaults.yaml adds."""
    return {
        "schema": 1,
        "name": "tiny",
        "server": {
            "model": {"name": "test/tiny"},
            "accelerator": "tpu7x",
            "topology": "1x1x1",
            "hosts": 1,
            "chips_per_host": 1,
            "startup_timeout": "10m",
            "args": {
                "tensor-parallel-size": 1,
                "data-parallel-size": 1,
                "max-model-len": 4096,
                "max-num-batched-tokens": 1024,
                "max-num-seqs": 64,
                "async-scheduling": True,
                "kv-cache-dtype": "fp8",
            },
            "env": {"TINY_KNOB": "1"},
        },
        "tasks": {
            "perf": {
                "dataset": {"name": "random"},
                "isl_osl": [[128, 128]],
                "concurrency": [1, 4],
                "num_prompts": {"per_concurrency": 2},
                "random_range_ratio": 0.5,
                "range_ratio_style": "min",
                "warmup_runs": 1,
                "temperature": 0,
                "percentile_metrics": PERCENTILES,
                "client_args": {"backend": "vllm", "ignore-eos": True, "seed": 42},
                "tolerance": 0.05,
                "baseline": {
                    "isl128_osl128_c1": {
                        "completed": 2,
                        "median_tpot_ms": 10.0,
                        "output_token_throughput": 500.0,
                    },
                    "isl128_osl128_c4": {"completed": 8, "median_tpot_ms": 10.0},
                },
            },
            "eval": {
                "lm_eval": {
                    "tasks": [
                        "mmlu_pro",
                        {
                            "name": "gsm8k",
                            "limit": 5,
                            "env": {"HF_ALLOW_CODE_EVAL": "1"},
                        },
                    ],
                    "model": "local-chat-completions",
                    "model_args": {"num_concurrent": 8},
                    "seed": "0,1234,None,1234",
                    "limit": 10,
                    "apply_chat_template": True,
                    "fewshot_as_multiturn": True,
                    "gen_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
                },
                "tolerance": 0.02,
                "baseline": {
                    "mmlu_pro": {"exact_match,custom-extract": 0.8},
                    "gsm8k": {"exact_match,custom-extract": 0.8},
                },
            },
        },
    }


def replace_program(plan: Plan, old: str, new: str) -> Plan:
    """`plan` with every command that starts `old` starting `new` instead.

    For programs at fixed paths, such as InferenceX's baked interpreter.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, Command) and value.argv[:1] == (old,):
            return dataclasses.replace(value, argv=(new, *value.argv[1:]))
        if isinstance(value, tuple):
            return tuple(walk(item) for item in value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            changes = {
                f.name: walk(getattr(value, f.name)) for f in dataclasses.fields(value)
            }
            return dataclasses.replace(value, **changes)
        return value

    return walk(plan)


def move_api_port(plan: Plan, role: str, port: int) -> Plan:
    """`plan` with `role` serving its API on `port`, so roles share a machine.

    The role's readiness and its first host's --port change, and so does
    every port that follows a Ref to its head: the router's
    --prefiller-ports or --decoder-ports, and the :port of a URL.
    """
    old = str(plan.role(role).readiness.port)
    head = Ref("role_head", role=role)

    def values(items: tuple[Any, ...]) -> tuple[Any, ...]:
        found = list(items)
        for index, item in enumerate(found):
            after_head = index >= 2 and found[index - 2] == head
            after_port = index >= 1 and found[index - 1] == "--port"
            url_port = (
                isinstance(item, str)
                and item.startswith(f":{old}")
                and index >= 1
                and found[index - 1] == head
            )
            if item == old and (after_head or (after_port and in_role)):
                found[index] = str(port)
            elif url_port:
                found[index] = f":{port}" + item[len(old) + 1 :]
        return tuple(found)

    def walk(value: Any) -> Any:
        if isinstance(value, Concat):
            return Concat(values(value.parts))
        if isinstance(value, Readiness) and in_role:
            return dataclasses.replace(value, port=port)
        if isinstance(value, tuple):
            return values(tuple(walk(item) for item in value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            changes = {
                f.name: walk(getattr(value, f.name)) for f in dataclasses.fields(value)
            }
            return dataclasses.replace(value, **changes)
        return value

    in_role = False
    roles = []
    for found in plan.roles:
        in_role = found.name == role
        roles.append(walk(found))
    in_role = False
    return dataclasses.replace(plan, roles=tuple(roles), driver=walk(plan.driver))


def signal_when_serving(summary: pathlib.Path, signum: int) -> threading.Thread:
    """Send `signum` to this process once the run's summary says it serves."""

    def watch() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if json.loads(summary.read_text())["state"] == "serving":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.02)
        os.kill(os.getpid(), signum)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


def serve_only(data: dict[str, Any]) -> dict[str, Any]:
    return {**data, "tasks": {}}


# Waits short enough for hosts on one machine; the status port is per test.
FAST: dict[str, Any] = {
    "scenario.runtime.control.POLL_S": 0.05,
    "scenario.runtime.control.GONE_S": 1.0,
    "scenario.runtime.control.LINGER_S": 1.0,
    "scenario.runtime.control.ABORT_LINGER_S": 0.5,
    "scenario.runtime.driver.OWN_SERVER_CHECK_S": 0.05,
    "scenario.runtime.bringup.POLL_INTERVAL_S": 0.05,
    "scenario.runtime.bringup.REMOTE_POLL_S": 0.05,
    "scenario.runtime.proc.CHECK_EVERY_S": 0.05,
}


class LocalHost:
    """One host of a run on this machine: a `scenario run` process.

    It runs the encoded Plan from SCENARIO_PLAN, as a pod does, with its
    placement as flags. Its output goes to <log_dir>/<name>.log.
    """

    def __init__(
        self,
        name: str,
        plan: Plan,
        args: Sequence[str],
        env: dict[str, str],
        log_dir: pathlib.Path,
        patch: dict[str, Any],
    ) -> None:
        self.name = name
        self.log_path = log_dir / f"{name}.log"
        env = {
            **env,
            "SCENARIO_PLAN": codec.encode(plan),
            "STUB_PATCH": json.dumps(patch),
        }
        with self.log_path.open("w") as out:
            self.popen = subprocess.Popen(
                [sys.executable, str(RUN_SCENARIO), "run", *args],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def wait(self, timeout: float = 60) -> int:
        try:
            return self.popen.wait(timeout)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"{self.name} still runs:\n{self.output()}") from None

    def send(self, signum: int) -> None:
        self.popen.send_signal(signum)

    def output(self) -> str:
        return self.log_path.read_text()

    def kill(self) -> None:
        if self.popen.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.popen.pid, signal.SIGKILL)
            self.popen.wait()


def wait_for(predicate: Any, what: str, within: float = 30) -> None:
    deadline = time.monotonic() + within
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


def state(summary: pathlib.Path) -> str | None:
    try:
        return json.loads(summary.read_text())["state"]
    except (OSError, ValueError):
        return None


def kill_group(pid: int) -> None:
    """SIGKILL what is left of a process group, such as a killed host's server."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
