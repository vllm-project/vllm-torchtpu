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
"""After-tasks: when they fire, what they get, and how they end a run."""

import dataclasses
import json
import os
import pathlib
import re
from typing import Any

import pytest
from plan_fixtures import Data, build, scenario
from runtime_helpers import REPO_ROOT, no_stray_processes, stub_env, tiny
from scenario.plan import codec
from scenario.plan.build import build_plan
from scenario.plan.types import Plan
from scenario.runtime import exits, publish
from scenario.runtime.driver import RunOptions, run
from scenario.target import AfterTask, TargetData, TargetInfo

pytestmark = pytest.mark.usefixtures("api_port", "fast_polls")

CELLS = ["isl128_osl128_c1", "isl128_osl128_c4"]


def notes(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "notes.txt"


def noting(name: str, *words: str, **fields: Any) -> AfterTask:
    """An after-task that appends `name` and its filled words to notes.txt."""
    script = 'printf "%s" "$NAME" >> "$NOTES"; printf " %s" "$@" >> "$NOTES"'
    script += '; echo >> "$NOTES"'
    env = (("NAME", name), ("NOTES", "{run_dir}/../../notes.txt"))
    return AfterTask(name, ("sh", "-c", script, "_", *words), env=env, **fields)


def failing(name: str, code: int = 1, **fields: Any) -> AfterTask:
    return AfterTask(name, ("sh", "-c", f"exit {code}"), **fields)


def target(*tasks: AfterTask) -> TargetData:
    return TargetData(TargetInfo("test", "abc", None), tasks)


def go(
    tmp_path: pathlib.Path,
    plan: Plan,
    env: dict[str, str],
    data: Data,
    after: TargetData | None,
) -> int:
    options = RunOptions(
        env=env,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
        resolved={"point": None, "scenario": data},
        target=after,
    )
    return run(plan, options)


def perf_only() -> Data:
    """Two cells, enough to see after-tasks fire per cell."""
    data = tiny()
    data["tasks"].pop("eval")
    perf = data["tasks"]["perf"]
    perf["isl_osl"] = [[128, 128]]
    perf["baseline"] = {key: perf["baseline"][key] for key in CELLS}
    return data


def lines(tmp_path: pathlib.Path) -> list[list[str]]:
    return [line.split(" ") for line in notes(tmp_path).read_text().splitlines()]


def summary(tmp_path: pathlib.Path) -> dict[str, Any]:
    return json.loads((tmp_path / "results" / "tiny" / "summary.json").read_text())


def test_result_and_end_tasks_fire_with_their_values(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, SCENARIO_IMAGE="registry.example/img@sha256:1")
    data = perf_only()
    after = target(
        noting("artifact", "{file}", "{label}", when="result"),
        noting("rows", "{file}", when="result", results="cells", require="result_ok"),
        noting("ok-rows", "{label}", when="result", require="result_ok"),
        noting("end", "{run_dir}", "{image}", "{run_stamp}", "{unpublished}", "{x}"),
    )
    assert go(tmp_path, build(data), env, data, after) == exits.OK
    run_dir = str(tmp_path / "results" / "tiny")
    found = lines(tmp_path)
    # Per cell: the always task, then the result_ok one; cells-only tasks
    # never fire on product cells.
    assert found[:4] == [
        entry
        for cell in CELLS
        for entry in (
            ["artifact", f"{run_dir}/{cell}.json", cell],
            ["ok-rows", cell],
        )
    ]
    name, run_dir_seen, image, stamp, unpublished, other = found[4]
    assert (name, run_dir_seen, image, other) == (
        "end",
        run_dir,
        "registry.example/img@sha256:1",
        "{x}",
    )
    assert re.fullmatch(r"\d{8}-\d{6}", stamp)
    handed = unpublished.split(";")
    assert f"{run_dir}/config.json" in handed and f"{run_dir}/summary.json" in handed
    assert f"{run_dir}/isl128_osl128_c1.warmup1.json" in handed
    assert not any(f"{run_dir}/{cell}.json" in handed for cell in CELLS)
    statuses = [(t["name"], t["status"]) for t in summary(tmp_path)["after_tasks"]]
    assert statuses.count(("artifact", "ok")) == 2 and ("end", "ok") in statuses
    assert no_stray_processes(env)


def test_the_failed_request_check_sits_between_the_stages(
    tmp_path: pathlib.Path,
) -> None:
    env = stub_env(tmp_path, STUB_BENCH_FAILED="1")
    data = perf_only()
    plan = build(data)
    (task,) = plan.driver.tasks
    limited = dataclasses.replace(task, max_failed_requests=0)
    plan = dataclasses.replace(
        plan, driver=dataclasses.replace(plan.driver, tasks=(limited,))
    )
    after = target(
        noting("artifact", "{label}", when="result"),
        noting("rows", "{label}", when="result", require="result_ok"),
    )
    assert go(tmp_path, plan, env, data, after) == exits.TASK_FAILED
    assert lines(tmp_path) == [["artifact", "isl128_osl128_c1"]]


def test_commands_ok_tasks_skip_after_a_failed_command(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_BENCH_FAIL="isl128_osl128_c4.json")
    data = perf_only()
    after = target(
        noting("artifacts", "{run_dir}"),
        noting("bigquery", "{run_dir}", require="commands_ok", fatal=False),
    )
    assert go(tmp_path, build(data), env, data, after) == exits.TASK_FAILED
    assert [line[0] for line in lines(tmp_path)] == ["artifacts"]
    assert summary(tmp_path)["after_tasks"][-1] == {
        "name": "bigquery",
        "when": "end",
        "status": "skipped",
        "reason": "a command failed",
    }


@pytest.mark.parametrize(
    ("fail_bench", "fatal", "code"),
    [
        (False, True, exits.AFTER_TASK),
        (True, True, exits.TASK_FAILED),
        (False, False, exits.OK),
    ],
)
def test_a_failed_after_task(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    fail_bench: bool,
    fatal: bool,
    code: int,
) -> None:
    knobs = {"STUB_BENCH_FAIL": "isl128_osl128_c1.json"} if fail_bench else {}
    env = stub_env(tmp_path, **knobs)
    data = perf_only()
    after = target(failing("upload", 3, fatal=fatal))
    assert go(tmp_path, build(data), env, data, after) == code
    out = capsys.readouterr().out
    if fatal:
        assert "::error::after-task upload exited 3" in out
    else:
        assert "::warning::after-task upload exited 3; not failing the run" in out
    assert summary(tmp_path)["exit"] == code


def test_an_after_task_times_out(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = perf_only()
    after = target(AfterTask("slow", ("sleep", "30"), timeout_s=1))
    assert go(tmp_path, build(data), env, data, after) == exits.AFTER_TASK
    assert summary(tmp_path)["after_tasks"] == [
        {"name": "slow", "when": "end", "status": "timeout", "exit": 124}
    ]


def test_experiments_skip_what_asks_to(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = perf_only()
    plan = build_plan(
        scenario(data), ref="tiny", overrides=["server.args.max-num-seqs=32"]
    )
    after = target(
        noting("artifacts", "{run_dir}"),
        noting("bigquery", "{run_dir}", skip_on_experiment=True),
    )
    assert go(tmp_path, plan, env, data, after) == exits.OK
    assert [line[0] for line in lines(tmp_path)] == ["artifacts"]


def test_a_bring_up_failure_still_hands_on_its_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    env = stub_env(tmp_path, STUB_SERVE_EXIT="1")
    data = perf_only()
    after = target(
        noting("artifacts", "{unpublished}"),
        noting("bigquery", "{run_dir}", require="commands_ok"),
    )
    assert go(tmp_path, build(data), env, data, after) == exits.NOT_READY
    ((name, handed),) = lines(tmp_path)
    assert name == "artifacts" and "server.log" in handed


def test_a_signal_runs_no_after_tasks(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = perf_only()
    data["tasks"] = {
        "custom": [
            {
                "name": "stop",
                "command": ["sh", "-c", f"kill -TERM {os.getpid()}; sleep 30"],
                "timeout": "1m",
            }
        ]
    }
    after = target(noting("artifacts", "{run_dir}"))
    assert go(tmp_path, build(data), env, data, after) == exits.SIGTERM
    assert not notes(tmp_path).exists()
    assert summary(tmp_path)["state"] == "aborted"


def test_pods_read_the_target_data_from_env(tmp_path: pathlib.Path) -> None:
    data = perf_only()
    data["tasks"] = {}
    data["tasks"]["custom"] = [{"name": "noop", "command": ["true"], "timeout": "1m"}]
    after = target(noting("end", "{run_dir}"))
    env = stub_env(tmp_path, SCENARIO_TARGET_DATA=codec.encode_data(after.to_data()))
    assert go(tmp_path, build(data), env, data, None) == exits.OK
    assert lines(tmp_path) == [["end", str(tmp_path / "results" / "tiny")]]


def test_placeholders() -> None:
    values = {"run_dir": "r", "file": "r/a.json"}
    assert (
        publish.fill("{run_dir}/{file}:{label}{{x}}", values)
        == "r/r/a.json:{label}{{x}}"
    )
