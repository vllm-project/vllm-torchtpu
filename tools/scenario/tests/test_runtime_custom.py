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
"""Custom tasks (env contract, timeout, skip_rest_if_exit_ge, order) and AgentX."""

import dataclasses
import json
import pathlib
from typing import Any

import pytest
from plan_fixtures import Data, build
from runtime_helpers import REPO_ROOT, records, replace_program, stub_env, tiny
from scenario.plan.plumbing import AGENTX_PYTHON
from scenario.plan.types import CustomTaskPlan, Plan
from scenario.runtime import exits
from scenario.runtime.driver import RunOptions, run

pytestmark = pytest.mark.usefixtures("api_port", "fast_polls")


def go(tmp_path: pathlib.Path, plan: Plan, env: dict[str, str], data: Data) -> int:
    options = RunOptions(
        env=env,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
        resolved={"point": None, "scenario": data},
    )
    return run(plan, options)


def out(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "results" / "tiny"


def tasks(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    return json.loads((out(tmp_path) / "summary.json").read_text())["tasks"]


def custom(name: str, script: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "command": ["sh", "-c", script], "timeout": "1h", **extra}


DUMP = 'env > "$SCENARIO_RESULTS_DIR/$TASK.env"'


def test_custom_tasks_get_the_env_contract(
    tmp_path: pathlib.Path, api_port: int
) -> None:
    env = stub_env(tmp_path)
    data = tiny()
    data["tasks"] = {
        "custom": [
            custom(
                "check",
                f"echo checking; {DUMP}",
                env={
                    "TASK": "check",
                    "LOG_DIR": "${SCENARIO_RESULTS_DIR}/check",
                    "COST": "$$5",
                },
            )
        ]
    }
    assert go(tmp_path, build(data), env, data) == exits.OK
    seen = dict(
        line.split("=", 1)
        for line in (out(tmp_path) / "check.env").read_text().splitlines()
        if "=" in line
    )
    run_dir = str(out(tmp_path).resolve())
    assert seen["SCENARIO_RESULTS_DIR"] == run_dir
    assert seen["SCENARIO_SERVED_MODEL"] == "test/tiny"
    assert seen["SCENARIO_API_PORT"] == str(api_port)
    assert seen["SCENARIO_SERVER_LOG"] == f"{run_dir}/server.log"
    assert seen["LOG_DIR"] == f"{run_dir}/check"
    assert seen["COST"] == "$5"
    # The host's env reaches its tasks, as a sourced config's exports did.
    assert seen["TINY_KNOB"] == "1"
    assert (out(tmp_path) / "custom" / "check.log").read_text() == "checking\n"
    assert tasks(tmp_path)[0] | {"started": 0, "ended": 0} == {
        "kind": "custom",
        "name": "check",
        "status": "passed",
        "exit": 0,
        "started": 0,
        "ended": 0,
    }


@pytest.mark.parametrize(
    ("first", "statuses"),
    [
        (2, ["failed", "skipped", "skipped"]),
        (1, ["failed", "passed", "passed"]),
        (0, ["passed", "passed", "passed"]),
    ],
)
def test_skip_rest_if_exit_ge(
    tmp_path: pathlib.Path, first: int, statuses: list[str]
) -> None:
    env = stub_env(tmp_path)
    data = tiny()
    data["tasks"].pop("perf")
    data["tasks"]["custom"] = [
        custom("check-completions", f"exit {first}", skip_rest_if_exit_ge=2),
        custom("correctness", "exit 0", skip_rest_if_exit_ge=2),
    ]
    code = go(tmp_path, build(data), env, data)
    assert code == (exits.OK if first == 0 else exits.TASK_FAILED)
    assert [t["status"] for t in tasks(tmp_path)] == statuses
    assert len(records(env, "lm_eval")) == (0 if first >= 2 else 2)
    if first >= 2:
        assert tasks(tmp_path)[2]["reason"] == (
            "check-completions exited 2, at least its skip_rest_if_exit_ge 2"
        )


def test_tasks_run_perf_custom_eval(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = tiny()
    data["tasks"]["perf"]["isl_osl"] = [[128, 128]]
    data["tasks"]["perf"]["concurrency"] = [1]
    data["tasks"]["perf"]["baseline"] = {"isl128_osl128_c1": {"completed": 2}}
    data["tasks"]["custom"] = [custom("between", "echo between")]
    assert go(tmp_path, build(data), env, data) == exits.OK
    assert [t["name"] for t in tasks(tmp_path)] == ["perf", "between", "eval"]
    order = [t["started"] for t in tasks(tmp_path)]
    assert order == sorted(order)


def test_a_custom_task_times_out(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = tiny()
    data["tasks"] = {"custom": [custom("slow", "sleep 30", skip_rest_if_exit_ge=100)]}
    plan = build(data)
    (task,) = plan.driver.tasks
    assert isinstance(task, CustomTaskPlan)
    quick = dataclasses.replace(
        task, timeout_s=1, command=dataclasses.replace(task.command, timeout_s=1)
    )
    plan = dataclasses.replace(
        plan, driver=dataclasses.replace(plan.driver, tasks=(quick,))
    )
    assert go(tmp_path, plan, env, data) == exits.TASK_FAILED
    (entry,) = tasks(tmp_path)
    assert (entry["status"], entry["exit"], entry["reason"]) == (
        "failed",
        124,
        "timed out",
    )
    summary = json.loads((out(tmp_path) / "summary.json").read_text())
    assert not summary["commands_ok"]


def agentx(**extra: Any) -> Data:
    data = tiny()
    data["tasks"] = {
        "inferencex": {
            "model_prefix": "tiny",
            "precision": "bf16",
            "framework": "vllm",
            "spec_decoding": "none",
            "kv_offloading": "none",
        },
        "agentx": {
            "concurrency": 4,
            "duration": "1m",
            "timeout": "5m",
            "env": {"AIPERF_MAX_CONTEXT_LENGTH": "4096"},
            **extra,
        },
    }
    return data


def test_agentx_replays_one_concurrency(tmp_path: pathlib.Path, api_port: int) -> None:
    env = stub_env(tmp_path)
    data = agentx()
    plan = replace_program(build(data), AGENTX_PYTHON, "python3")
    assert go(tmp_path, plan, env, data) == exits.OK
    (call,) = records(env, "infx agentic")
    seen = call["env"]
    assert seen["CONC"] == "4" and seen["DURATION"] == "60"
    assert seen["AIPERF_SERVER_URL"] == f"http://127.0.0.1:{api_port}"
    assert seen["RESULT_DIR"] == str(out(tmp_path).resolve())
    assert seen["AIPERF_MAX_CONTEXT_LENGTH"] == "4096"
    (entry,) = tasks(tmp_path)
    assert entry["results"] == ["agentx_c4.json"]
    assert entry["submission_valid"] is True
    assert "gate" not in entry


def test_agentx_records_an_invalid_run(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """aiperf's verdict is recorded and warned about, not gated on."""
    env = stub_env(tmp_path, STUB_AGENTX_VALID="false")
    data = agentx()
    plan = replace_program(build(data), AGENTX_PYTHON, "python3")
    assert go(tmp_path, plan, env, data) == exits.OK
    (entry,) = tasks(tmp_path)
    assert (entry["status"], entry["submission_valid"]) == ("passed", False)
    assert "aiperf marked the run invalid" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("knobs", "reason"),
    [
        ({"STUB_AGENTX_EXIT": "3"}, "exited 3"),
        ({"STUB_AGENTX_RESULT": "0"}, "no agentx_c4* result in"),
    ],
)
def test_agentx_failures(
    tmp_path: pathlib.Path, knobs: dict[str, str], reason: str
) -> None:
    env = stub_env(tmp_path, **knobs)
    data = agentx()
    plan = replace_program(build(data), AGENTX_PYTHON, "python3")
    assert go(tmp_path, plan, env, data) == exits.TASK_FAILED
    (entry,) = tasks(tmp_path)
    assert entry["reason"].startswith(reason)
