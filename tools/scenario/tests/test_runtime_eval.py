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
"""The eval executor: lm_eval runs and their gates, the request cache,
evalscope, and InferenceX's eval client."""

import dataclasses
import json
import pathlib
import re
from typing import Any

import pytest
from plan_fixtures import Data, build
from runtime_helpers import REPO_ROOT, records, replace_program, stub_env, tiny
from scenario.plan.plumbing import AGENTX_PYTHON
from scenario.plan.types import (
    Command,
    EvalRunPlan,
    EvalTaskPlan,
    GatePlan,
    Plan,
    Ref,
    concat,
)
from scenario.runtime import exec_eval, exits
from scenario.runtime.driver import RunOptions, run
from scenario.runtime.placement import Placement, Refs

pytestmark = pytest.mark.usefixtures("api_port", "fast_polls")


def eval_only(data: Data | None = None) -> Data:
    data = data or tiny()
    data["tasks"].pop("perf")
    return data


def go(
    tmp_path: pathlib.Path, plan: Plan, env: dict[str, str], data: Data, **kwargs: Any
) -> int:
    options = RunOptions(
        env=env,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
        resolved={"point": None, "scenario": data},
        **kwargs,
    )
    return run(plan, options)


def out(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "results" / "tiny"


def summary(tmp_path: pathlib.Path) -> dict[str, Any]:
    return json.loads((out(tmp_path) / "summary.json").read_text())


def runs(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    return summary(tmp_path)["tasks"][-1]["runs"]


def test_lm_eval_runs_each_task_and_gates_it(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = eval_only()
    plan = build(data)
    assert go(tmp_path, plan, env, data) == exits.OK
    calls = records(env, "lm_eval")
    run_dir = str(out(tmp_path).resolve())
    refs = Refs(
        plan,
        Placement("server", 0),
        run_dir=out(tmp_path),
        server_log=out(tmp_path) / "server.log",
        env={},
    )
    (task,) = plan.driver.tasks
    assert isinstance(task, EvalTaskPlan)
    assert [["lm_eval", *c["argv"]] for c in calls] == [
        refs.argv(r.command.argv) for r in task.runs
    ]
    assert [c["argv"][:3] for c in calls] == [
        ["--tasks", "mmlu_pro", "--seed"],
        ["--tasks", "gsm8k", "--seed"],
    ]
    assert calls[0]["argv"][calls[0]["argv"].index("--output_path") + 1] == run_dir
    model_args = calls[0]["argv"][calls[0]["argv"].index("--model_args") + 1]
    assert re.fullmatch(
        r"model=test/tiny,base_url=http://127\.0\.0\.1:\d+/v1/chat/completions,num_concurrent=8",
        model_args,
    )
    # Per-task env reaches that task only; no cache root, no cache.
    assert calls[1]["env"]["HF_ALLOW_CODE_EVAL"] == "1"
    assert "HF_ALLOW_CODE_EVAL" not in calls[0]["env"]
    assert all("--cache_requests" not in c["argv"] for c in calls)
    assert all("LM_HARNESS_CACHE_PATH" not in c["env"] for c in calls)
    assert [(r["key"], r["status"], r["gate"]["status"]) for r in runs(tmp_path)] == [
        ("mmlu_pro", "ok", "pass"),
        ("gsm8k", "ok", "pass"),
    ]
    for key in ("mmlu_pro", "gsm8k"):
        assert (
            "PASS: all evaluated tasks"
            in (out(tmp_path) / f"eval_check_{key}.md").read_text()
        )
    assert "Requesting API: 100%|" in (out(tmp_path) / "lm_eval_gsm8k.log").read_text()


def test_a_failed_eval_does_not_stop_the_next(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_LM_EVAL_FAIL="mmlu_pro")
    data = eval_only()
    assert go(tmp_path, build(data), env, data) == exits.TASK_FAILED
    assert [c["argv"][1] for c in records(env, "lm_eval")] == ["mmlu_pro", "gsm8k"]
    first, second = runs(tmp_path)
    assert first["status"] == "failed" and first["reason"] == "lm_eval exited 1"
    assert first["gate"] == {"status": "skipped", "reason": "the eval failed"}
    assert second["gate"]["status"] == "pass"
    assert not summary(tmp_path)["commands_ok"]


def test_one_gate_failing_leaves_the_others(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_LM_EVAL_SCORES='{"gsm8k": 0.5}')
    data = eval_only()
    assert go(tmp_path, build(data), env, data) == exits.TASK_FAILED
    assert [r["gate"]["status"] for r in runs(tmp_path)] == ["pass", "fail"]
    assert summary(tmp_path)["commands_ok"]


def test_the_request_cache(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = eval_only()
    cache = tmp_path / "cache"
    for attempt in ("first", "second"):
        assert (
            go(tmp_path / attempt, build(data), env, data, lm_eval_cache=str(cache))
            == 0
        )
    calls = records(env, "lm_eval")
    paths = [c["env"]["LM_HARNESS_CACHE_PATH"] for c in calls]
    assert all(c["argv"][-2:] == ["--cache_requests", "true"] for c in calls)
    # Keyed by content: the same task gets the same directory on every run.
    assert paths[0] == paths[2] and paths[1] == paths[3] and paths[0] != paths[1]
    assert all(re.fullmatch(re.escape(str(cache)) + r"/[0-9a-f]{16}", p) for p in paths)
    assert (
        pathlib.Path(paths[0]) / "requests.txt"
    ).read_text() == "mmlu_pro\nmmlu_pro\n"


def test_the_target_supplies_the_cache_root(tmp_path: pathlib.Path) -> None:
    from scenario.target import TargetData, TargetInfo

    env = stub_env(tmp_path)
    data = eval_only()
    target = TargetData(
        TargetInfo("t", None, None), lm_eval_cache_root=str(tmp_path / "c")
    )
    assert go(tmp_path, build(data), env, data, target=target) == exits.OK
    assert all(
        c["env"]["LM_HARNESS_CACHE_PATH"].startswith(str(tmp_path / "c"))
        for c in records(env, "lm_eval")
    )


BASE = [
    "lm_eval",
    "--tasks",
    "humaneval_plus_tpu",
    "--seed",
    "0,1234,None,1234",
    "--output_path",
    "/runs/a",
    "--include_path",
    "tasks",
    "--model",
    "local-chat-completions",
    "--model_args",
    "model=m,base_url=http://10.0.0.1:8000/v1/chat/completions,num_concurrent=8,timeout=60",
    "--gen_kwargs",
    '{"x":1}',
]


def key(tmp_path: pathlib.Path, argv: list[str], version: str = "0.4.9") -> str:
    return exec_eval.request_key(argv, version, tmp_path)


def test_the_request_key_follows_what_decides_requests(tmp_path: pathlib.Path) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "humaneval.yaml").write_text("task: humaneval_plus_tpu\n")
    first = key(tmp_path, BASE)
    assert exec_eval.request_spec(BASE) == [
        "--tasks",
        "humaneval_plus_tpu",
        "--seed",
        "0,1234,None,1234",
        "--include_path",
        "tasks",
        "--model",
        "local-chat-completions",
        "--gen_kwargs",
        '{"x":1}',
        "extra_model_args=timeout=60",
    ]

    def changed(index: int, value: str) -> list[str]:
        argv = list(BASE)
        argv[index] = value
        return argv

    # Where results go, the endpoint, the served name and the concurrency
    # change no request.
    assert key(tmp_path, changed(6, "/runs/b")) == first
    elsewhere = "model=n,base_url=http://other:9/v1/chat/completions,num_concurrent=64,timeout=60"
    assert key(tmp_path, changed(12, elsewhere)) == first
    # The arguments, the version and the task files do.
    assert key(tmp_path, changed(14, '{"x":2}')) != first
    assert key(tmp_path, [*BASE, "--limit", "5"]) != first
    assert key(tmp_path, BASE, version="0.5.0") != first
    (tasks / "humaneval.yaml").write_text("task: humaneval_plus_tpu\nnum_fewshot: 1\n")
    assert key(tmp_path, BASE) != first
    # A tokenizer setting makes the served name count.
    tokenized = changed(12, "model=m,base_url=u,tokenizer=t")
    assert exec_eval.request_spec(tokenized)[-2:] == [
        "extra_model_args=tokenizer=t",
        "model=m",
    ]


def test_a_concurrency_sweep_runs_and_gates_per_key(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_LM_EVAL_SCORES='{"mmlu_pro": 0.5}')
    data = eval_only()
    lm_eval = data["tasks"]["eval"]["lm_eval"]
    lm_eval["concurrency"] = [1, 2]
    lm_eval["tasks"] = ["mmlu_pro"]
    lm_eval.pop("model_args")
    data["tasks"]["eval"]["baseline"] = {
        "mmlu_pro_c2": {"exact_match,custom-extract": 0.8}
    }
    assert go(tmp_path, build(data), env, data) == exits.TASK_FAILED
    outputs = [
        c["argv"][c["argv"].index("--output_path") + 1] for c in records(env, "lm_eval")
    ]
    assert [pathlib.Path(p).name for p in outputs] == ["mmlu_pro_c1", "mmlu_pro_c2"]
    assert [(r["key"], r["gate"]["status"]) for r in runs(tmp_path)] == [
        ("mmlu_pro_c1", "skipped"),
        ("mmlu_pro_c2", "fail"),
    ]


def with_eval(plan: Plan, task: EvalTaskPlan) -> Plan:
    driver = dataclasses.replace(plan.driver, tasks=(task,))
    return dataclasses.replace(plan, driver=driver)


def evalscope_task(gated: bool) -> EvalTaskPlan:
    work = concat(Ref("results_dir"), "/mmmu_pro_evalscope")
    argv = (
        "evalscope",
        "eval",
        "--model",
        "test/tiny",
        "--datasets",
        "mmmu_pro",
        "--work-dir",
        work,
    )
    gate = (
        GatePlan(0.02, (("mmmu_pro", (("acc,none", 0.6),)),), "inline")
        if gated
        else None
    )
    return EvalTaskPlan(
        (EvalRunPlan("mmmu_pro", "mmmu_pro", "evalscope", Command(argv), gate),)
    )


def test_evalscope_reports_become_lm_eval_results(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path)
    data = eval_only()
    stale = out(tmp_path) / "mmmu_pro_evalscope" / "old.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier run")
    plan = with_eval(build(data), evalscope_task(gated=True))
    assert go(tmp_path, plan, env, data) == exits.OK
    assert json.loads((out(tmp_path) / "results_mmmu_pro.json").read_text()) == {
        "results": {"mmmu_pro": {"acc,none": 0.6}}
    }
    assert not stale.exists()
    assert runs(tmp_path)[0]["gate"]["status"] == "pass"


def test_a_missing_evalscope_report_fails(tmp_path: pathlib.Path) -> None:
    env = stub_env(tmp_path, STUB_EVALSCOPE_REPORT="0")
    data = eval_only()
    plan = with_eval(build(data), evalscope_task(gated=False))
    assert go(tmp_path, plan, env, data) == exits.TASK_FAILED
    assert runs(tmp_path)[0]["reason"] == "no evalscope report"


def inferencex_eval() -> Data:
    data = tiny()
    data["tasks"] = {
        "inferencex": {
            "model_prefix": "tiny",
            "precision": "bf16",
            "framework": "vllm",
            "spec_decoding": "none",
            "kv_offloading": "none",
        },
        "eval": {
            "client": "inferencex",
            "framework": "lm-eval",
            "concurrency": 4,
            "timeout": "1h",
            "env": {"EVAL_LIMIT": "5"},
        },
    }
    return data


@pytest.mark.parametrize(
    ("knobs", "code", "gate"),
    [
        ({}, exits.OK, "pass"),
        ({"STUB_IX_FLOOR_EXIT": "1"}, exits.TASK_FAILED, "fail"),
        ({"STUB_IX_EVAL_EXIT": "2"}, exits.TASK_FAILED, "skipped"),
    ],
)
def test_the_inferencex_eval_client_and_its_floor(
    tmp_path: pathlib.Path, knobs: dict[str, str], code: int, gate: str
) -> None:
    env = stub_env(tmp_path, **knobs)
    data = inferencex_eval()
    plan = replace_program(build(data), AGENTX_PYTHON, "python3")
    assert go(tmp_path, plan, env, data) == code
    (call,) = records(env, "infx eval")
    assert call["env"]["EVAL_ONLY"] == "true" and call["env"]["EVAL_LIMIT"] == "5"
    stage = out(tmp_path).resolve() / "inferencex_eval"
    assert call["argv"][call["argv"].index("--stage-to") + 1] == str(stage)
    floors = records(env, "infx validate_scores")
    assert len(floors) == (0 if gate == "skipped" else 1)
    if floors:
        assert floors[0]["argv"][:2] == ["--meta-env", str(stage / "meta_env.json")]
    (entry,) = runs(tmp_path)
    assert entry["gate"]["status"] == gate
    assert summary(tmp_path)["commands_ok"] == ("STUB_IX_EVAL_EXIT" not in knobs)
