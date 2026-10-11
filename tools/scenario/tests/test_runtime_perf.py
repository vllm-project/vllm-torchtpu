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
"""The perf executor against the stub server and clients."""

import dataclasses
import json
import pathlib
from typing import Any

import pytest
from plan_fixtures import Data, build
from runtime_helpers import REPO_ROOT, no_stray_processes, records, stub_env, tiny
from scenario.plan.types import PerfTaskPlan, Plan
from scenario.runtime import exits
from scenario.runtime.driver import RunOptions, run
from scenario.runtime.placement import Placement, Refs

pytestmark = pytest.mark.usefixtures("api_port", "fast_polls")

CELLS = ["isl128_osl128_c1", "isl128_osl128_c4"]


def perf_only(data: Data | None = None) -> Data:
    data = data or tiny()
    data["tasks"].pop("eval")
    return data


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


def perf_entry(tmp_path: pathlib.Path) -> dict[str, Any]:
    summary = json.loads((out(tmp_path) / "summary.json").read_text())
    return summary["tasks"][0]


def filled(plan: Plan, tmp_path: pathlib.Path) -> list[list[str]]:
    """Every perf client argv of the Plan, its Refs filled as the run fills them."""
    refs = Refs(
        plan,
        Placement("server", 0),
        run_dir=out(tmp_path),
        server_log=out(tmp_path) / "server.log",
        env={},
        pulled_dir=lambda: None,
    )
    task = plan.driver.tasks[0]
    assert isinstance(task, PerfTaskPlan)
    argvs = []
    for cell in task.cells:
        for command in (*cell.warmups, cell.measured):
            argvs.append(refs.argv(command.argv))
    return argvs


def test_cells_run_with_todays_files(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = perf_only()
    plan = build(data)
    assert go(tmp_path, plan, env, data) == exits.OK
    calls = records(env, "vllm bench serve")
    assert [["vllm", "bench", "serve", *c["argv"]] for c in calls] == filled(
        plan, tmp_path
    )
    # num_prompts {per_concurrency: 2} and the min-style lengths, as planned.
    measured = calls[1]["argv"]
    assert measured[measured.index("--num-prompts") + 1] == "2"
    assert measured[measured.index("--random-input-len") + 1] == "96"
    assert calls[3]["argv"][calls[3]["argv"].index("--num-prompts") + 1] == "8"
    names = sorted(path.name for path in out(tmp_path).iterdir())
    for cell in CELLS:
        for suffix in (".json", ".log", ".warmup1.json", ".warmup1.log"):
            assert cell + suffix in names
    log = (out(tmp_path) / "benchmark.log").read_text()
    assert "MODEL: test/tiny" in log and "CONFIG: tiny" in log
    assert "TP: 1  DP: 1  EP: false" in log and "EXIT CODE: 0" in log
    assert "100%|" in (out(tmp_path) / "isl128_osl128_c1.log").read_text()
    entry = perf_entry(tmp_path)
    assert entry["status"] == "passed"
    assert [(c["key"], c["status"], c["result"]) for c in entry["cells"]] == [
        (cell, "ok", f"{cell}.json") for cell in CELLS
    ]
    assert no_stray_processes(env)


def test_a_failed_cell_does_not_stop_the_next(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_BENCH_FAIL="isl128_osl128_c4.json")
    data = perf_only()
    assert go(tmp_path, build(data), env, data) == exits.TASK_FAILED
    assert len(records(env, "vllm bench serve")) == 4
    entry = perf_entry(tmp_path)
    assert [c["status"] for c in entry["cells"]] == ["ok", "failed"]
    assert entry["cells"][1]["reason"] == "the client exited 1"
    summary = json.loads((out(tmp_path) / "summary.json").read_text())
    assert not summary["commands_ok"]
    assert "EXIT CODE: 1" in (out(tmp_path) / "benchmark.log").read_text()


def test_a_failed_warmup_skips_its_measured_run(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_BENCH_FAIL="isl128_osl128_c1.warmup1.json")
    data = perf_only()
    assert go(tmp_path, build(data), env, data) == exits.TASK_FAILED
    files = [pathlib.Path(c["argv"][-1]).name for c in records(env, "vllm bench serve")]
    assert "isl128_osl128_c1.json" not in files
    assert "isl128_osl128_c4.json" in files
    assert perf_entry(tmp_path)["cells"][0]["reason"] == "warmup 1 exited 1"


def test_a_dead_server_stops_the_task(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_BENCH_KILL="isl128_osl128_c4.json")
    data = perf_only()
    assert go(tmp_path, build(data), env, data) == exits.SERVER_DIED
    assert len(records(env, "vllm bench serve")) == 4
    entry = perf_entry(tmp_path)
    assert entry["status"] == "failed" and entry["reason"] == "the server died"
    assert [c["status"] for c in entry["cells"]] == ["ok", "failed"]
    assert no_stray_processes(env)


def object_storage(data: Data) -> Data:
    data["server"]["model"]["path"] = "gs://bucket/models/tiny"
    data["server"]["args"]["load-format"] = "runai_streamer"
    return data


@pytest.mark.parametrize("pulled", [True, False])
def test_the_client_tokenizes_from_the_pulled_weights(
    tmp_path: pathlib.Path, pulled: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    env = stub_env(tmp_path, **({"STUB_PULL": "1"} if pulled else {}))
    data = object_storage(perf_only())
    assert go(tmp_path, build(data), env, data) == exits.OK
    for call in records(env, "vllm bench serve"):
        tokenizer = call["argv"][call["argv"].index("--tokenizer") + 1]
        assert tokenizer == (env["STUB_PULLED_DIR"] if pulled else "test/tiny")
    if not pulled:
        assert "has no config.json; the client resolves test/tiny" in (
            capsys.readouterr().out
        )


def test_the_inferencex_client(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = perf_only()
    perf = data["tasks"]["perf"]
    perf["client"] = "inferencex"
    for key in ("client_args", "range_ratio_style"):
        perf.pop(key)
    perf["num_warmups"] = {"per_concurrency": 2}
    plan = build(data)
    assert go(tmp_path, plan, env, data) == exits.OK
    calls = records(env, "infx benchmark_serving")
    assert len(calls) == 4
    first = calls[0]
    assert first["env"]["PYTHONPATH"] == "/opt/infx/InferenceX/inferencex-e2e"
    assert first["argv"][first["argv"].index("--result-dir") + 1] == str(
        out(tmp_path).resolve()
    )
    assert (out(tmp_path) / "isl128_osl128_c4.json").is_file()


def with_perf(plan: Plan, **changes: Any) -> Plan:
    (task,) = plan.driver.tasks
    driver = dataclasses.replace(
        plan.driver, tasks=(dataclasses.replace(task, **changes),)
    )
    return dataclasses.replace(plan, driver=driver)


def test_too_many_failed_requests_stop_the_task(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_BENCH_FAILED="1")
    data = perf_only()
    plan = with_perf(build(data), max_failed_requests=0)
    assert go(tmp_path, plan, env, data) == exits.TASK_FAILED
    entry = perf_entry(tmp_path)
    assert [c["status"] for c in entry["cells"]] == ["failed", "skipped"]
    assert entry["cells"][0]["reason"] == "1 failed requests, over the 0 allowed"
    assert entry["cells"][1]["reason"] == (
        "isl128_osl128_c1 had 1 failed requests, over the 0 allowed"
    )
