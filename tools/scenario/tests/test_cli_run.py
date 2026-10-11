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
"""scenario run, up, perf and eval end to end, against the stub server and clients."""

import json
import pathlib
import signal
import time
from collections.abc import Iterator
from typing import Any

import pytest
from click.testing import CliRunner, Result
from runtime_helpers import (
    REPO_ROOT,
    no_stray_processes,
    records,
    signal_when_serving,
    stub_env,
)
from scenario.cli import main
from scenario.plan import codec
from scenario.plan.build import plan, resolved_payload
from scenario.resolve import resolve
from scenario.runtime import exits, net
from scenario.runtime.proc import Proc
from scenario.target import AfterTask, Target, TargetData, TargetInfo
from scenario.targets.local import LocalTarget

TESTS = pathlib.Path(__file__).resolve().parent
TINY = TESTS / "data" / "runtime" / "scenarios" / "smoke" / "tiny.yaml"
K3_PD = TESTS / "data" / "repo" / "scenarios" / "kimi-k3" / "tp8-dp4-ep-1p1d.yaml"
CELLS = ("isl128_osl128_c1", "isl128_osl128_c4")

pytestmark = pytest.mark.usefixtures("api_port", "fast_polls")


@pytest.fixture
def env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Stub programs on PATH for the CLI, run from the repository root."""
    found = stub_env(tmp_path)
    for name in ("PATH", "PYTHONPATH", "STUB_RECORDS", "STUB_PULLED_DIR"):
        monkeypatch.setenv(name, found[name])
    monkeypatch.chdir(REPO_ROOT)
    return found


def invoke(*args: Any) -> Result:
    return CliRunner().invoke(main, [str(arg) for arg in args])


def out(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "results" / "tiny"


def summary(run_dir: pathlib.Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text())


def test_run_serves_runs_and_gates(tmp_path: pathlib.Path, env: dict[str, str]) -> None:
    result = invoke("run", TINY, "--results-dir", tmp_path / "results")
    assert result.exit_code == exits.OK, result.output
    run_dir = out(tmp_path)
    names = {path.name for path in run_dir.iterdir()}
    for cell in CELLS:
        assert {f"{cell}.json", f"{cell}.log", f"{cell}.warmup1.json"} <= names
    assert {
        "config.json",
        "scenario_run.json",
        "summary.json",
        "server.log",
        "benchmark.log",
        "regression_check.md",
        "eval_check_mmlu_pro.md",
        "eval_check_gsm8k.md",
    } <= names
    config = json.loads((run_dir / "config.json").read_text())
    assert config["model"] == "test/tiny" and config["isl_osl_configs"] == "128:128"
    assert config["lm_eval_tasks"] == "mmlu_pro gsm8k"
    done = summary(run_dir)
    assert (done["state"], done["exit"], done["commands_ok"]) == ("done", 0, True)
    gates = [t["gate"]["status"] for t in done["tasks"] if "gate" in t]
    gates += [r["gate"]["status"] for r in done["tasks"][1]["runs"]]
    assert gates == ["pass", "pass", "pass"]
    assert done["after_tasks"] == []
    ran = json.loads((run_dir / "scenario_run.json").read_text())
    assert ran["target"]["name"] == "local"
    assert ran["scenario"] == {"ref": "smoke/tiny", "name": "tiny", "point": None}
    assert ran["resolved"]["scenario"]["tasks"]["perf"]["warmup_runs"] == 1
    assert "--- [scenario]" in result.output and "server/0 exit 0" in result.output
    assert no_stray_processes(env)


@pytest.mark.parametrize(
    ("knobs", "code"),
    [
        ({"STUB_BENCH_METRICS": '{"median_tpot_ms": 20.0}'}, exits.TASK_FAILED),
        ({"STUB_BENCH_KILL": "isl128_osl128_c1.json"}, exits.SERVER_DIED),
        ({"STUB_SERVE_EXIT": "1"}, exits.NOT_READY),
    ],
)
def test_run_exit_codes(
    tmp_path: pathlib.Path,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    knobs: dict[str, str],
    code: int,
) -> None:
    for name, value in knobs.items():
        monkeypatch.setenv(name, value)
    result = invoke("run", TINY, "--results-dir", tmp_path / "results")
    assert result.exit_code == code, result.output
    assert summary(out(tmp_path))["exit"] == code
    assert no_stray_processes(env)


def test_run_only_and_experiments(
    tmp_path: pathlib.Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_BENCH_METRICS", '{"median_tpot_ms": 20.0}')
    result = invoke(
        "run",
        TINY,
        "--results-dir",
        tmp_path / "results",
        "--set",
        "server.args.max-num-seqs=32",
        "--only",
        "perf",
    )
    assert result.exit_code == exits.OK, result.output
    assert records(env, "lm_eval") == []
    (perf,) = summary(out(tmp_path))["tasks"]
    assert perf["gate"] == {"status": "skipped", "reason": "experiment: gates are off"}


def test_a_pod_runs_from_its_env(
    tmp_path: pathlib.Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (resolved,) = resolve(TINY)
    planned = plan(resolved)
    note = tmp_path / "uploaded.txt"
    target = TargetData(
        TargetInfo("pods", "abc", str(tmp_path / "pod-results")),
        (AfterTask("upload", ("sh", "-c", f'echo "$0" > {note}', "{run_dir}")),),
    )
    monkeypatch.setenv("SCENARIO_PLAN", codec.encode(planned))
    monkeypatch.setenv(
        "SCENARIO_RESOLVED", codec.encode_data(resolved_payload(resolved))
    )
    monkeypatch.setenv("SCENARIO_TARGET_DATA", codec.encode_data(target.to_data()))
    monkeypatch.setenv("SCENARIO_HOST_INDEX", "0")
    monkeypatch.setenv("SCENARIO_IMAGE", "registry.example/img@sha256:2")
    result = invoke("run", "--role", "server", "--only", "perf")
    assert result.exit_code == exits.USAGE
    result = invoke("run", "--role", "server")
    assert result.exit_code == exits.OK, result.output
    run_dir = tmp_path / "pod-results" / "tiny"
    assert note.read_text().strip() == str(run_dir)
    ran = json.loads((run_dir / "scenario_run.json").read_text())
    assert ran["target"] == {"name": "pods", "commit": "abc"}
    assert ran["image"] == "registry.example/img@sha256:2"
    assert json.loads((run_dir / "config.json").read_text())["max_num_seqs"] == 64
    assert summary(run_dir)["after_tasks"][0]["status"] == "ok"
    dry = invoke("run", "--dry-run", "--role", "server")
    assert dry.exit_code == 0 and "vllm serve test/tiny" in dry.output


@pytest.mark.parametrize(
    ("args", "env_plan", "message"),
    [
        ((), None, "run needs a scenario FILE, or SCENARIO_PLAN"),
        (("--point", "x"), "", "need a scenario FILE"),
        ((), "not a plan", "SCENARIO_PLAN: not an encoded Plan"),
    ],
)
def test_pod_usage(
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
    env_plan: str | None,
    message: str,
) -> None:
    if env_plan is None:
        monkeypatch.delenv("SCENARIO_PLAN", raising=False)
    else:
        monkeypatch.setenv(
            "SCENARIO_PLAN", env_plan or codec.encode(plan(resolve(TINY)[0]))
        )
    result = invoke("run", *args)
    assert result.exit_code == exits.USAGE
    assert message in result.stderr


def test_several_roles_wait_for_the_next_version(env: dict[str, str]) -> None:
    result = invoke("run", K3_PD, "--role", "prefill", "--host-index", "0")
    assert result.exit_code == exits.USAGE
    assert "this version runs single-host, single-role plans" in result.output


def test_up_serves_until_stopped(tmp_path: pathlib.Path, env: dict[str, str]) -> None:
    signal_when_serving(out(tmp_path) / "summary.json", signal.SIGINT)
    result = invoke("up", TINY, "--results-dir", tmp_path / "results")
    assert result.exit_code == exits.OK, result.output
    assert "Serving test/tiny at http://127.0.0.1:" in result.output
    assert records(env, "vllm bench serve") == []
    assert no_stray_processes(env)


@pytest.fixture
def server(env: dict[str, str], api_port: int) -> Iterator[str]:
    """A stub server that is already up, as perf --url and eval --url expect."""
    proc = Proc(
        ["vllm", "serve", "test/tiny", "--port", str(api_port)], env=env
    ).start()
    url = f"http://127.0.0.1:{api_port}"
    deadline = time.monotonic() + 10
    while net.http_status(f"{url}/health") != 200 and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        yield url
    finally:
        proc.stop()


def test_perf_against_a_running_server(
    tmp_path: pathlib.Path, env: dict[str, str], server: str
) -> None:
    result = invoke(
        "perf", TINY, "--url", server, "--results-dir", tmp_path / "results"
    )
    assert result.exit_code == exits.OK, result.output
    calls = records(env, "vllm bench serve")
    assert len(calls) == 4
    port = server.rsplit(":", 1)[1]
    assert all(c["argv"][c["argv"].index("--port") + 1] == port for c in calls)
    assert len(records(env, "vllm serve")) == 1
    assert records(env, "lm_eval") == []
    assert [t["name"] for t in summary(out(tmp_path))["tasks"]] == ["perf"]


def test_eval_against_a_running_server(
    tmp_path: pathlib.Path, env: dict[str, str], server: str
) -> None:
    result = invoke(
        "eval",
        TINY,
        "--url",
        server,
        "--results-dir",
        tmp_path / "results",
        "--lm-eval-cache",
        tmp_path / "cache",
    )
    assert result.exit_code == exits.OK, result.output
    calls = records(env, "lm_eval")
    assert [c["argv"][1] for c in calls] == ["mmlu_pro", "gsm8k"]
    assert all(
        c["env"]["LM_HARNESS_CACHE_PATH"].startswith(str(tmp_path / "cache"))
        for c in calls
    )
    assert records(env, "vllm bench serve") == []


@pytest.mark.parametrize("url", ["https://h:1", "http://h:1/v1", "h:1"])
def test_a_url_is_http_host_port(env: dict[str, str], url: str) -> None:
    result = invoke("perf", TINY, "--url", url)
    assert result.exit_code == exits.USAGE
    assert "--url is http://HOST:PORT" in result.stderr


def test_without_a_server_a_url_run_is_not_ready(
    tmp_path: pathlib.Path, env: dict[str, str]
) -> None:
    result = invoke(
        "eval", TINY, "--url", "http://127.0.0.1:9", "--results-dir", tmp_path / "r"
    )
    assert result.exit_code == exits.NOT_READY


def test_the_local_target(env: dict[str, str]) -> None:
    from scenario.runtime.driver import RunOptions

    target = LocalTarget(RunOptions(env={}, repo_root=REPO_ROOT))
    assert isinstance(target, Target)
    info = target.info()
    assert info.name == "local" and info.results_root is None
    assert info.commit is None or len(info.commit) == 40
    assert target.reserved_env().names == frozenset()
