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
"""A single-host run against the stub server: files, phases, exit codes, teardown."""

import dataclasses
import json
import os
import pathlib
import re
import signal
import socket
import time
from typing import Any

import pytest
from plan_fixtures import Data, build
from runtime_helpers import (
    REPO_ROOT,
    no_stray_processes,
    records,
    serve_only,
    signal_when_serving,
    stub_env,
    tiny,
)
from scenario.plan import codec
from scenario.plan.types import Command, Plan
from scenario.runtime import exits
from scenario.runtime.context import TaskContext, TaskOutcome
from scenario.runtime.driver import RunOptions, run
from scenario.runtime.proc import Proc

pytestmark = pytest.mark.usefixtures("api_port", "fast_polls")


def options(tmp_path: pathlib.Path, env: dict[str, str], **kwargs: Any) -> RunOptions:
    data = kwargs.pop("resolved_data", tiny())
    return RunOptions(
        env=env,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
        resolved={"point": None, "scenario": data},
        **kwargs,
    )


def run_dir(tmp_path: pathlib.Path, name: str = "tiny") -> pathlib.Path:
    return tmp_path / "results" / name


def summary(tmp_path: pathlib.Path) -> dict[str, Any]:
    return json.loads((run_dir(tmp_path) / "summary.json").read_text())


def with_readiness(plan: Plan, timeout_s: int) -> Plan:
    role = plan.roles[0]
    readiness = dataclasses.replace(role.readiness, timeout_s=timeout_s)
    return dataclasses.replace(
        plan, roles=(dataclasses.replace(role, readiness=readiness),)
    )


class Calls:
    """Executors that record what they saw and do as told."""

    def __init__(self, outcomes: dict[str, Any] | None = None) -> None:
        self.seen: list[str] = []
        self.outcomes = outcomes or {}

    def executor(
        self, ctx: TaskContext, task: Any, entry: dict[str, Any]
    ) -> TaskOutcome:
        self.seen.append(task.kind)
        action = self.outcomes.get(task.kind, "pass")
        if callable(action):
            return action(ctx, task, entry)
        return TaskOutcome("passed" if action == "pass" else "failed")

    def table(self) -> dict[str, Any]:
        return {"perf": self.executor, "eval": self.executor, "custom": self.executor}


def run_benchmarks_config_keys() -> list[str]:
    """The config.json keys run_benchmarks.sh writes, in order."""
    script = (REPO_ROOT / "scripts/vllm/benchmarking/run_benchmarks.sh").read_text()
    block = script.split('cat > "$RESULTS_DIR/config.json" << EOF', 1)[1]
    block = block.split("\nEOF", 1)[0]
    return re.findall(r'^\s*"([a-z_]+)":', block, flags=re.MULTILINE)


def test_a_run_serves_drives_and_tears_down(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    plan = build(tiny())
    calls = Calls()
    assert run(plan, options(tmp_path, env), calls.table()) == exits.OK
    assert calls.seen == ["perf", "eval"]

    out = run_dir(tmp_path)
    (served,) = records(env, "vllm serve")
    assert served["argv"] == list(plan.roles[0].host_plans[0].serve.argv[2:])
    assert served["cwd"] == str(REPO_ROOT)
    # The server gets the scenario's env and the tool's, over what it inherits.
    assert served["env"]["TINY_KNOB"] == "1"
    assert served["env"]["VLLM_ENGINE_READY_TIMEOUT_S"] == "600"
    assert served["env"]["TPU_ACCELERATOR_TYPE"] == "tpu7x"
    assert "INFO stub: serving test/tiny" in (out / "server.log").read_text()
    assert no_stray_processes(env)

    done = summary(tmp_path)
    assert done["state"] == "done" and done["exit"] == 0 and done["commands_ok"]
    assert list(done["phases"]) == ["placement", "server-spawn", "ready", "exit"]
    assert [(t["name"], t["status"]) for t in done["tasks"]] == [
        ("perf", "passed"),
        ("eval", "passed"),
    ]
    ran = json.loads((out / "scenario_run.json").read_text())
    assert ran["scenario"] == {"ref": "tiny", "name": "tiny", "point": None}
    assert ran["plan_sha"] == codec.digest(plan)
    assert codec.from_data(ran["plan"]) == plan
    assert ran["resolved"]["scenario"]["name"] == "tiny"
    assert ran["target"] == {"name": "none", "commit": None}
    # tiny names no image and no target set one.
    assert ran["placement"]["role"] == "server" and ran["image"] == ""


def test_config_json_has_todays_shape(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    run(build(tiny()), options(tmp_path, env), Calls().table())
    config = json.loads((run_dir(tmp_path) / "config.json").read_text())
    assert list(config) == run_benchmarks_config_keys()
    assert config | {"timestamp": "T"} == {
        "model": "test/tiny",
        "model_uri": None,
        "tensor_parallelism": 1,
        "data_parallelism": 1,
        "enable_ep": False,
        "quantization": "none",
        "isl_osl_configs": "128:128",
        "dataset_name": "random",
        "concurrency_options": "1 4",
        "num_prompts": {"per_concurrency": 2},
        "random_range_ratio": 0.5,
        "range_ratio_style": "min",
        "benchmark_warmup_runs": 1,
        "benchmark_temperature": 0,
        "benchmark_extra_body": None,
        "max_model_len": 4096,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 64,
        "async_scheduling": True,
        "kv_cache_dtype": "fp8",
        "capture_profile": 0,
        "profile_dir": None,
        "profile_gcs_dir": None,
        "timestamp": "T",
        "mmlu_pro_disable_multiturn_args": False,
        "lm_eval_tasks": "mmlu_pro gsm8k",
        "extra_lm_eval_model_args": "num_concurrent=8",
        "lm_eval_gen_kwargs": '{"chat_template_kwargs":{"enable_thinking":false}}',
        "perf_tolerance": 0.05,
        "eval_tolerance": 0.02,
    }
    assert re.fullmatch(r"\d{8}-\d{6}", config["timestamp"])


def test_the_child_env_adds_only_plan_names(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    plan = build(tiny())
    run(plan, options(tmp_path, env), Calls().table())
    (served,) = records(env, "vllm serve")
    added = {name for name, value in served["env"].items() if env.get(name) != value}
    host = plan.roles[0].host_plans[0]
    planned = {name for name, _ in (*host.scenario_env, *host.tool_env)}
    # The shell shim and the interpreter add a few of their own.
    assert added - planned <= {"PWD", "SHLVL", "_", "OLDPWD", "__CF_USER_TEXT_ENCODING"}


def test_a_server_that_dies_starting_is_exit_4(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_SERVE_EXIT="3")
    calls = Calls()
    assert run(build(tiny()), options(tmp_path, env), calls.table()) == exits.NOT_READY
    assert calls.seen == []
    done = summary(tmp_path)
    assert done["exit"] == 4 and not done["commands_ok"]
    assert "engine failed to start" in (run_dir(tmp_path) / "server.log").read_text()


def test_a_server_never_ready_is_exit_4(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = stub_env(tmp_path, STUB_NEVER_READY="1")
    plan = with_readiness(build(tiny()), timeout_s=1)
    assert run(plan, options(tmp_path, env), Calls().table()) == exits.NOT_READY
    assert "not ready within 1s" in capsys.readouterr().out
    assert no_stray_processes(env)


def test_a_taken_port_is_exit_3(tmp_path: pathlib.Path, api_port: int) -> None:
    env = stub_env(tmp_path)
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", api_port))
        squatter.listen()
        assert run(build(tiny()), options(tmp_path, env), Calls().table()) == 3
    assert records(env, "vllm serve") == []


def test_a_server_dying_mid_run_is_exit_5(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_DIE_AFTER="0.3")

    def wait_for_death(
        ctx: TaskContext, task: Any, entry: dict[str, Any]
    ) -> TaskOutcome:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ctx.alive()
            time.sleep(0.02)
        return TaskOutcome("passed")

    calls = Calls({"perf": wait_for_death})
    assert (
        run(build(tiny()), options(tmp_path, env), calls.table()) == exits.SERVER_DIED
    )
    done = summary(tmp_path)
    assert [(t["name"], t["status"], t.get("reason")) for t in done["tasks"]] == [
        ("perf", "failed", "the server died"),
        ("eval", "skipped", "the server died"),
    ]
    assert not done["commands_ok"]
    assert no_stray_processes(env)


def test_a_failed_task_does_not_stop_the_next(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    calls = Calls({"perf": "fail"})
    assert (
        run(build(tiny()), options(tmp_path, env), calls.table()) == exits.TASK_FAILED
    )
    assert calls.seen == ["perf", "eval"]


def test_skip_rest_skips_the_remaining_tasks(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)

    def stop_here(ctx: TaskContext, task: Any, entry: dict[str, Any]) -> TaskOutcome:
        return TaskOutcome("failed", skip_rest="perf said so")

    calls = Calls({"perf": stop_here})
    assert run(build(tiny()), options(tmp_path, env), calls.table()) == 1
    assert calls.seen == ["perf"]
    assert summary(tmp_path)["tasks"][1] == {
        "kind": "eval",
        "name": "eval",
        "status": "skipped",
        "reason": "perf said so",
    }


def test_sigterm_stops_everything(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)

    def terminate(ctx: TaskContext, task: Any, entry: dict[str, Any]) -> TaskOutcome:
        finished = ctx.run(
            "sleeper",
            Command(("sh", "-c", f"kill -TERM {os.getpid()}; exec sleep 60")),
        )
        raise AssertionError(f"not interrupted: {finished}")

    calls = Calls({"perf": terminate})
    assert run(build(tiny()), options(tmp_path, env), calls.table()) == exits.SIGTERM
    done = summary(tmp_path)
    assert done["state"] == "aborted" and done["exit"] == 143
    assert done["tasks"][0]["reason"] == "interrupted"
    assert no_stray_processes(env)


def test_up_serves_until_stopped(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    signal_when_serving(run_dir(tmp_path) / "summary.json", signal.SIGINT)
    assert run(build(tiny()), options(tmp_path, env, mode="up"), {}) == exits.OK
    assert summary(tmp_path)["tasks"] == []
    assert no_stray_processes(env)


def test_a_run_without_tasks_serves(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    signal_when_serving(run_dir(tmp_path) / "summary.json", signal.SIGINT)
    plan = build(serve_only(tiny()))
    assert run(plan, options(tmp_path, env), {}) == exits.OK
    assert no_stray_processes(env)


def test_a_url_run_drives_a_running_server(
    tmp_path: pathlib.Path, api_port: int
) -> None:
    env = stub_env(tmp_path)
    server = Proc(
        ["vllm", "serve", "test/tiny", "--port", str(api_port)], env=env
    ).start()
    try:
        url = f"http://127.0.0.1:{api_port}"
        for _ in range(200):
            if _ready(url):
                break
            time.sleep(0.05)
        calls = Calls()
        code = run(
            build(tiny()), options(tmp_path, env, mode="url", url=url), calls.table()
        )
        assert code == exits.OK
        assert calls.seen == ["perf", "eval"]
        assert len(records(env, "vllm serve")) == 1
    finally:
        server.stop()
    calls = Calls()
    code = run(
        build(tiny()), options(tmp_path, env, mode="url", url=url), calls.table()
    )
    assert code == exits.NOT_READY and calls.seen == []


def _ready(url: str) -> bool:
    from scenario.runtime import net

    return net.http_status(f"{url}/health") == 200


def test_several_hosts_wait_for_a_later_version(
    tmp_path: pathlib.Path, variants: dict[str, Data]
) -> None:
    env = stub_env(tmp_path)
    plan = build(variants["single"])
    assert run(plan, options(tmp_path, env), {}) == exits.USAGE
    assert not (tmp_path / "results").exists()
